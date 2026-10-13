"""Phases 1 and 2: scan the video, pick candidates statistically, preview them."""

import collections
import math
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from . import device
from .frame_source import open_capture, use_sequential_indexing
from .measurement import (
    measure_length_pca,
    candidate_navigation_order,
    deduplicate_detections_by_mask_iou,
    draw_readable_text,
    enhance_review_frame,
    expand_bbox,
    expanded_bbox_crop,
    extract_detection_mask,
    normalize_mask_shape,
)
from .settings_ui import (
    ask_confirm,
    get_shared_tk_root,
    offer_new_analysis,
    restart_application,
)
from .telemetry import get_altitude, get_nearest_altitude, get_tilt

# Zoom margin for the candidate preview. The same value crops the mask saved
# during screening and the frame shown later, so the two line up exactly.
_PREVIEW_ZOOM_MARGIN = 0.4


class AnalysisProgress:
    """Progress window, updated by the screening loop."""

    def __init__(self, total_frames: int):
        import _tkinter
        import tkinter as tk
        from tkinter import ttk

        self._tkinter = _tkinter
        self.tk = tk
        self.cancelled = False
        self.started_at = time.monotonic()
        self.last_update = 0.0
        self.root = tk.Toplevel(get_shared_tk_root())
        self.root.title("Whale Vision — Analysis in progress")
        self.root.geometry("600x260")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        container = ttk.Frame(self.root, padding=22)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="Video screening", font=("", 18, "bold")).pack(
            anchor="w"
        )
        self.status = tk.StringVar(value="Preparing…")
        ttk.Label(container, textvariable=self.status).pack(anchor="w", pady=(8, 12))
        self.progress = ttk.Progressbar(
            container,
            maximum=max(1, total_frames),
            mode="determinate",
            length=550,
        )
        self.progress.pack(fill="x")
        self.details = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.details).pack(anchor="w", pady=(12, 4))
        self.timing = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.timing).pack(anchor="w")
        ttk.Button(
            container,
            text="Stop and use the results collected so far",
            command=self.cancel,
        ).pack(
            anchor="e", pady=(18, 0)
        )
        self._process_events()

    def cancel(self):
        if self.cancelled:
            return
        confirmed = ask_confirm(
            self.root,
            "Confirm stop analysis",
            (
                "Do you really want to stop the screening?\n\n"
                "The analysis will continue using only the candidates "
                "collected so far."
            ),
            yes_label="Stop and keep the candidates collected so far",
            no_label="No, keep scanning",
        )
        if not confirmed:
            self.status.set("Stop cancelled: analysis in progress…")
            return
        self.cancelled = True
        self.status.set("Stopping…")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(
        self,
        completed: int,
        total: int,
        frame_idx: int,
        video_second: int,
        candidates: int,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and now - self.last_update < 0.1:
            return not self.cancelled
        self.last_update = now
        elapsed = now - self.started_at
        fraction = completed / total if total else 1.0
        eta = elapsed * (total - completed) / completed if completed else 0.0
        self.progress["maximum"] = max(1, total)
        self.progress["value"] = completed
        self.status.set(f"Progress: {fraction * 100:.1f}%")
        self.details.set(
            f"Video frame: {frame_idx}  •  Second: {video_second}  •  "
            f"Valid candidates: {candidates}"
        )
        eta_text = self._format_duration(eta) if completed else "computing…"
        self.timing.set(
            f"Elapsed: {self._format_duration(elapsed)}  •  "
            f"Estimated time remaining: {eta_text}"
        )
        self._process_events()
        return not self.cancelled

    def _process_events(self):
        try:
            for _event_index in range(24):
                event_processed = self.root.tk.dooneevent(
                    self._tkinter.DONT_WAIT
                )
                if not event_processed:
                    break
        except self.tk.TclError:
            self.cancelled = True

    def close(self):
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


class CandidatePreviewWindow:
    """Reusable candidate preview and timeline window.

    It is opened several times in one session: if the manual review does not
    reach the requested quota, the user comes back here to look for more
    frames. Building a fresh `tk.Tk()` each round proved fragile (invalid image
    names, controls that stopped responding after the first pass), so this
    window is built once and then reused with `withdraw`/`deiconify`, the same
    way `ReviewActionPanel` and `AnalysisProgress` do elsewhere in this package.

    Navigation stays quick with many candidates, or with an arrow key held
    down, because it moves only the highlight on the current point instead of
    redrawing the whole graph (`_refresh_current_marker` rather than
    `_draw_graph_full`), reuses the same `PhotoImage` when the size has not
    changed, and avoids an expensive video seek when the next frame is close to
    the current one.
    """

    # _screening_quartile value for a candidate that measured fine but is a
    # length outlier, or falls outside the requested size quartiles (see
    # run_screening, PHASE 2). These used to be dropped from the final pool
    # without the user ever seeing them. They now stay selectable like the
    # rest, just drawn in a different colour.
    OUTLIER_QUARTILE = 5

    # Fixed-size image panel, never sized to its content. Sizing it from
    # image_holder.winfo_width/height made it change with every candidate
    # (crops differ), which shuffled the graph and controls below on each frame.
    _PREVIEW_PANEL_SIZE = (1000, 580)

    _REASON_ORDER = (
        "altitude",
        "tilt",
        "yolo_none",
        "yolo_low_conf",
        "yolo_extra_low_conf",
        "yolo_mask",
        "off_center",
        "unusual_shape",
        "no_sample",
    )
    # Every colour here must stay distinct from the other reasons AND from
    # everything else on the same graph: quartile_colors below (Q1 #3498db,
    # Q2/accepted #2ecc71, Q3 #f39c12, Q4 #9b59b6), the "already discarded"
    # grey #7f8c8d and the axis line #aeb6bf.
    #
    # Two real clashes were fixed here: "tilt" used to be the exact same hex as
    # Q4, and "off_center" was the same grey as "already discarded" and all
    # but invisible against the axis line right above the tick strip. A second
    # rejected for being off-centre (the most common way a measurement is lost)
    # could therefore look like it had no tick at all.
    _REASON_COLORS = {
        "altitude": "#e67e22",
        "tilt": "#8e44ad",
        "yolo_none": "#e74c3c",
        "yolo_low_conf": "#1abc9c",
        # Different meaning from yolo_low_conf: not one animal rejected, but a
        # sign that ANOTHER animal was probably there next to the one already
        # handled in this frame. It can never get a whale_id, having never had a
        # box at the main confidence. Its colour is used by nothing else.
        "yolo_extra_low_conf": "#ff6fae",
        "yolo_mask": "#d35400",
        "off_center": "#00b8d4",
        "unusual_shape": "#f1c40f",
        # White, not dark grey: against the almost black graph background a
        # dark tick disappears, making it look like that second has no
        # recorded reason at all.
        "no_sample": "#ffffff",
    }
    _REASON_LABELS = {
        "altitude": "unstable altitude",
        "tilt": "excessive tilt",
        # A single "yolo" entry used to cover both "nothing detected" and "a box
        # was found but no mask came out of it", hiding the difference from the
        # user. There are now three separate reasons; see run_screening for
        # where each is assigned.
        "yolo_none": "no whale detected",
        "yolo_low_conf": "detected below the confidence threshold",
        "yolo_extra_low_conf": "possible additional individual below confidence (not tracked)",
        "yolo_mask": "detected, but the mask could not be extracted",
        "off_center": "off-center",
        "unusual_shape": "unusual shape",
        "no_sample": "excluded (unspecified reason)",
    }

    @staticmethod
    def _reason_parts(reason):
        """Split a rejection reason into its base code and optional detail.

        The base code is what _REASON_COLORS and _REASON_LABELS are keyed on;
        the detail is whatever follows ':', such as the confidence value for
        yolo_low_conf.
        """
        base, _, detail = reason.partition(":")
        return base, detail

    @classmethod
    def _reason_label(cls, reason):
        base, detail = cls._reason_parts(reason)
        label = cls._REASON_LABELS.get(base, base)
        return f"{label} ({detail})" if detail else label

    def __init__(self):
        self.root = None

    # ------------------------------------------------------------------
    # Building the widgets (done once for the whole session)
    # ------------------------------------------------------------------
    def _build(self):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self._tk = tk
        self._messagebox = messagebox

        self.root = tk.Toplevel(get_shared_tk_root())
        self.root.title("Whale Vision — Candidate preview")
        self.root.geometry("1200x950")
        self.root.minsize(1000, 700)
        # Left resizable (no resizable(False, ...) call), so the window can be
        # grown to nearly full screen. The image panel follows the WINDOW size
        # (see _on_root_configure), not the current content, which is why it
        # still stays steady from one candidate to the next.
        self._preview_ready = False
        self._resize_after_id = None
        self.root.bind("<Configure>", self._on_root_configure)

        self.current_position = tk.IntVar(value=0)
        self.navigation_mode = tk.StringVar(value="Chronological")
        self.preview_contrast = tk.DoubleVar(value=100.0)
        self.contrast_text = tk.StringVar(value="Contrast: 1.00×")
        self.include_current = tk.BooleanVar(value=True)
        self.frame_info = tk.StringVar()
        self.selected_count_text = tk.StringVar()
        self.show_yolo_box = tk.BooleanVar(value=True)
        self.show_sam_box = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_length = tk.BooleanVar(value=True)
        self.rejection_summary_text = tk.StringVar()

        # All the content sits in a scrollable canvas. On a screen shorter than
        # the window's minimum size (900x700) the controls at the bottom (start
        # review and so on) were otherwise out of reach, with no way to scroll.
        scroll_canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        content = ttk.Frame(scroll_canvas)
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor="nw")

        # Two separate one-way handlers, so they cannot retrigger each other
        # forever. An earlier single "clever" handler that also adjusted the
        # height did exactly that and pinned the CPU at 100%.
        #
        # The canvas drives only the WIDTH of `content` (needed by widgets with
        # fill="x"), never its height: `content` keeps its natural height, and
        # changing its width cannot affect that because no wraplength here
        # depends on the actual width. `content` only updates the scrollregion
        # and never calls itemconfigure, so neither handler can start the other.
        def _sync_content_width(event):
            scroll_canvas.itemconfigure(content_window, width=event.width)

        def _sync_scrollregion(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        scroll_canvas.bind("<Configure>", _sync_content_width)
        content.bind("<Configure>", _sync_scrollregion)

        def _on_mousewheel(event):
            # macOS and Windows report event.delta (its sign is the direction);
            # Linux uses Button-4/Button-5 instead. One step at a time on both:
            # that is enough for a wheel or two fingers, and treating them alike
            # avoids over-scrolling on macOS, where delta is not a multiple of
            # 120 as it is on Windows.
            if getattr(event, "num", None) == 4:
                scroll_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                scroll_canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_mousewheel(_event=None):
            self.root.bind_all("<MouseWheel>", _on_mousewheel)
            self.root.bind_all("<Button-4>", _on_mousewheel)
            self.root.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None):
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")

        # The global binding is only active while the pointer is over this
        # window, so it does not steal scrolling from other open Tk windows
        # (the review panel, for instance).
        scroll_canvas.bind("<Enter>", _bind_mousewheel)
        scroll_canvas.bind("<Leave>", _unbind_mousewheel)

        header = ttk.Frame(content, padding=(14, 12, 14, 6))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Preview of candidate frames",
            font=("", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "YOLO and SAM are unloaded from memory here. Browse the frames, "
                "exclude the useless ones, then start the detailed review."
            ),
            wraplength=1050,
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            content,
            textvariable=self.frame_info,
            font=("", 12, "bold"),
            padding=(14, 4),
        ).pack(fill="x")

        self.image_holder = ttk.Frame(content, padding=(14, 4))
        self.image_holder.pack(fill="x")
        panel_width, panel_height = self._PREVIEW_PANEL_SIZE
        # tk.Frame (not ttk) plus pack_propagate(False) keeps exactly this size
        # whatever it contains, instead of shrinking to the current image.
        self.image_stage = tk.Frame(
            self.image_holder, width=panel_width, height=panel_height, background="#111111"
        )
        self.image_stage.pack()
        self.image_stage.pack_propagate(False)
        self.image_label = ttk.Label(self.image_stage, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        ttk.Label(
            self.image_holder,
            text="Scroll over the image to zoom in/out, drag to pan, double-click to reset",
            foreground="#7f8c8d",
        ).pack(anchor="w", pady=(2, 0))
        # Manual zoom and pan on the image, separate from page scrolling. The
        # "break" return stops the event reaching the global wheel binding that
        # scrolls the whole window.
        self.image_label.bind("<MouseWheel>", self._on_preview_wheel)
        self.image_label.bind("<Button-4>", self._on_preview_wheel)
        self.image_label.bind("<Button-5>", self._on_preview_wheel)
        self.image_label.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self.image_label.bind("<B1-Motion>", self._on_preview_drag_move)
        self.image_label.bind("<Double-Button-1>", self._on_preview_reset)

        self.graph = tk.Canvas(
            content,
            height=190,
            background="#17202a",
            highlightthickness=0,
        )
        self.graph.pack(fill="x", padx=14, pady=(5, 8))

        # Always-visible summary of why frames were rejected. Before this, the
        # only way to find out was clicking a tiny tick in the graph.
        ttk.Label(
            content,
            textvariable=self.rejection_summary_text,
            font=("", 11, "bold"),
            foreground="#c0392b",
            padding=(14, 0, 14, 2),
            wraplength=1050,
        ).pack(fill="x")

        # The legend is built from real widgets (a colour swatch plus a label)
        # rather than text drawn inside the canvas, which had no readable legend.
        reason_legend = ttk.Frame(content, padding=(14, 0, 14, 6))
        reason_legend.pack(fill="x")
        ttk.Label(
            reason_legend,
            text="Excluded frames (click a tick in the graph for detail):",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 3))
        # A 3-column grid rather than one row: six entries with descriptive
        # labels ran off the side of the window.
        columns = 3
        for index, reason in enumerate(self._REASON_ORDER):
            entry = ttk.Frame(reason_legend)
            entry.grid(row=1 + index // columns, column=index % columns, sticky="w", padx=(0, 18), pady=1)
            tk.Frame(
                entry,
                width=12,
                height=12,
                background=self._REASON_COLORS[reason],
                highlightthickness=1,
                highlightbackground="#888888",
            ).pack(side="left", padx=(0, 4))
            ttk.Label(entry, text=self._REASON_LABELS[reason]).pack(side="left")

        controls = ttk.Frame(content, padding=(14, 4, 14, 12))
        controls.pack(fill="x")
        display_controls = ttk.Frame(controls)
        display_controls.pack(fill="x", pady=(0, 8))
        navigation = ttk.Frame(controls)
        navigation.pack(fill="x")

        ttk.Label(display_controls, text="Order:").pack(side="left", padx=(0, 5))
        self.order_selector = ttk.Combobox(
            display_controls,
            textvariable=self.navigation_mode,
            values=("Chronological", "Size (largest first)"),
            state="readonly",
            width=27,
        )
        self.order_selector.pack(side="left", padx=(0, 18))
        self.order_selector.bind("<<ComboboxSelected>>", self._change_navigation_mode)

        ttk.Label(display_controls, textvariable=self.contrast_text).pack(side="left", padx=(0, 6))
        self.contrast_slider = ttk.Scale(
            display_controls,
            from_=50,
            to=250,
            variable=self.preview_contrast,
            command=self._update_contrast_label,
            takefocus=False,
        )
        self.contrast_slider.pack(side="left", fill="x", expand=True)
        self.contrast_slider.bind("<ButtonRelease-1>", self._apply_preview_contrast)

        # Lets the user see the clean frame when the overlays get in the way,
        # for instance a box covering part of the outline.
        overlay_toggles = ttk.Frame(controls)
        overlay_toggles.pack(fill="x", pady=(0, 8))
        ttk.Label(overlay_toggles, text="Show:").pack(side="left", padx=(0, 5))
        ttk.Checkbutton(
            overlay_toggles, text="YOLO box", variable=self.show_yolo_box, command=self._show_current_frame
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            overlay_toggles, text="SAM box", variable=self.show_sam_box, command=self._show_current_frame
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            overlay_toggles, text="mask", variable=self.show_mask, command=self._show_current_frame
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            overlay_toggles,
            text="length line and points",
            variable=self.show_length,
            command=self._show_current_frame,
        ).pack(side="left")

        ttk.Button(navigation, text="← Previous", command=lambda: self._move(-1)).pack(side="left")
        ttk.Label(
            navigation,
            text="Use the arrow keys, C to include/exclude, or click a point in the graph",
            anchor="center",
        ).pack(side="left", fill="x", expand=True, padx=12)
        ttk.Button(navigation, text="Next →", command=lambda: self._move(1)).pack(side="left")

        selection_row = ttk.Frame(controls)
        selection_row.pack(fill="x", pady=(10, 0))
        self.include_checkbox = ttk.Checkbutton(
            selection_row,
            text="Include this frame in the detailed review (C)",
            variable=self.include_current,
            command=self._toggle_current,
        )
        self.include_checkbox.pack(side="left")
        ttk.Label(selection_row, textvariable=self.selected_count_text).pack(side="right")

        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(
            actions,
            text="Select all",
            command=lambda: self._select_everything(True),
        ).pack(side="left", padx=(0, 5))
        ttk.Button(
            actions,
            text="Deselect all",
            command=lambda: self._select_everything(False),
        ).pack(side="left", padx=5)
        ttk.Button(
            actions,
            text="Back to settings",
            command=self._request_settings,
        ).pack(side="right", padx=(5, 0))
        ttk.Button(
            actions,
            text="Start review of selected frames",
            command=self._finish_selection,
        ).pack(side="right", padx=5)

        self.root.bind("<Left>", lambda event: self._handle_arrow(event, -1))
        self.root.bind("<Right>", lambda event: self._handle_arrow(event, 1))
        self.root.bind("<c>", self._handle_space)
        self.root.bind("<C>", self._handle_space)
        self.graph.bind("<Button-1>", self._jump_from_graph)
        # Raised on hover, without needing a click: with many candidates close
        # in time the dots overlap and the one under the cursor can be hidden.
        self.graph.bind("<Motion>", self._hover_graph)
        # Without this binding the graph keeps the size the Canvas had on its
        # very first draw (often 1x1), before the window manager placed it.
        self.graph.bind("<Configure>", lambda _event: self._draw_graph_full())
        self.root.protocol("WM_DELETE_WINDOW", self._request_settings)

    # ------------------------------------------------------------------
    # Un round di anteprima
    # ------------------------------------------------------------------
    def run(
        self,
        candidates,
        video_path,
        video_fps,
        telemetry,
        start_sec,
        end_sec,
        measure_in_meters,
        config,
        decided_status=None,
        rejected_frames=None,
        subject_label=None,
        current_whale_id=None,
    ):
        """Show the preview: one frame in memory at a time and no inference.

        `subject_label` (optional, e.g. "Whale 1/2"): with several animals in
        one video the preview and review loop runs once per animal, and this
        label goes in the window title to say which one is being reviewed.

        `decided_status` (optional) maps `_candidate_id` -> "accepted" or
        "discarded" for candidates already reviewed in an earlier round. They
        are shown but cannot be selected again, so the user can look for more
        to reach the quota without reviewing the same frame twice.

        `rejected_frames` (optional) maps a frame number to why that sample
        never became a candidate ("altitude", "tilt", "yolo_none",
        "yolo_low_conf", "yolo_mask", "off_center", "unusual_shape",
        "no_sample"). It is drawn on the graph to explain missing frames.
        Keyed per frame, not per second, so a rejected sample stays visible even
        if another sample of the same second did become a candidate.
        """
        if candidates.empty:
            return candidates.copy()
        if self.root is None:
            self._build()
        self.root.title(
            "Whale Vision — Candidate preview"
            + (f" — {subject_label}" if subject_label else "")
        )

        self.candidates = candidates.reset_index(drop=True)
        self.video_fps = video_fps
        self.telemetry = telemetry
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.unit_label = "meters" if measure_in_meters else "pixels"
        self.rejected_frames = rejected_frames or {}
        self._config = config
        self.current_whale_id = current_whale_id

        decided_status = decided_status or {}
        self.locked_status_by_position = {}
        if decided_status and "_candidate_id" in self.candidates.columns:
            for position, candidate_id in enumerate(self.candidates["_candidate_id"]):
                if candidate_id in decided_status:
                    self.locked_status_by_position[position] = decided_status[candidate_id]

        self.video_capture = open_capture(video_path, sequential=use_sequential_indexing(config))
        if not self.video_capture.isOpened():
            raise RuntimeError(
                f"Impossibile aprire il video per l'anteprima: {video_path}"
            )

        self.candidate_count = len(self.candidates)
        # Print the total received here so it can be compared with the
        # "accepted (candidate)" lines printed during screening (run_screening,
        # same file). If the two disagree, the missing candidates were lost
        # between the scan and this window, not while drawing the graph.
        print(f"  > Candidate preview: received {self.candidate_count} candidate(s) to show.")
        # Everything starts unselected: the user picks what goes to review,
        # rather than having to deselect from a full list.
        self.selected_positions = set()
        # Frames with no candidate at all (YOLO found nothing) that the user
        # still sends to review to annotate by hand. Maps frame_number -> image
        # size, used to build a placeholder bbox (the whole frame) for the
        # synthetic row that ends up in `to_review`; see
        # _build_manual_candidate_rows.
        self.manual_selected_frames = {}
        self._viewing_rejected = False
        self._current_rejected_frame = None
        self._current_rejected_frame_size = None
        self._current_rejected_reason = None
        # Manual zoom and pan on the preview image (wheel and drag), reset
        # whenever the shown frame changes; see _preview_key in
        # _show_current_frame and _show_rejected_frame.
        self._preview_zoom = 1.0
        self._preview_pan = [0.5, 0.5]
        self._preview_key = None
        # The view a double click returns to for the current frame: the
        # subject's bbox for candidates, the whole frame for rejected ones. See
        # _on_preview_reset.
        self._preview_default = (1.0, [0.5, 0.5])
        self._preview_drag_origin = None
        self._preview_last_scale = 1.0
        self._preview_last_source_size = (1, 1)
        self.navigation_mode.set("Chronological")
        self.navigation_order = candidate_navigation_order(
            self.candidates,
            self.navigation_mode.get(),
        )
        self.graph_candidate_hits = []
        self.rejected_hits = []
        self.result_positions = []
        self.return_to_settings = False
        self.preview_photo = None
        self._point_items = {}
        self._point_centers = {}
        self._cross_items = {}
        self._last_shown_position = 0
        self._frame_cache = collections.OrderedDict()
        self.current_position.set(0)
        self._update_selected_count_text()

        self._draw_graph_full()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._preview_ready = True
        self.root.after(80, self._show_current_frame)
        self.root.mainloop()
        self._preview_ready = False

        self.video_capture.release()
        self.video_capture = None
        if self.return_to_settings:
            return None
        selected = self.candidates.iloc[self.result_positions].reset_index(drop=True)
        if self.manual_selected_frames:
            manual_rows = self._build_manual_candidate_rows()
            selected = (
                pd.concat([selected, manual_rows], ignore_index=True)
                if not selected.empty
                else manual_rows
            )
        return selected

    def _update_selected_count_text(self):
        manual_count = len(self.manual_selected_frames)
        text = f"{len(self.selected_positions)} frames included in the review"
        if manual_count:
            text += f" (+{manual_count} without automatic detection, manual only)"
        self.selected_count_text.set(text)

    def _build_manual_candidate_rows(self):
        """Synthetic rows for frames with no candidate at all.

        See `manual_selected_frames`. They are in the same format
        `review.run_review` reads: bbox_x1..y2 cover the whole frame as a
        placeholder (never used for inference, see `_manual_only` in
        run_review), and gsd uses the same formula as screening.
        """
        pixel_size = self._config.sensor_width_mm / self._config.image_width_px
        focal_length = self._config.focal_length_mm
        rows = []
        for frame_number, size in self.manual_selected_frames.items():
            seconds = frame_number / max(float(self.video_fps), 1e-9)
            altitude = get_nearest_altitude(self.telemetry, seconds)
            gsd = (
                (float(altitude) * pixel_size) / focal_length
                if self.unit_label == "meters"
                else 1.0
            )
            rows.append(
                {
                    "frame_pos": frame_number,
                    "video_second": int(seconds),
                    "h": altitude,
                    "gsd": gsd,
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": float(size[0]),
                    "bbox_y2": float(size[1]),
                    "auto_length": np.nan,
                    "whale_id": self.current_whale_id,
                    "_candidate_id": -int(frame_number) - 1,
                    "_manual_only": True,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Navigazione / lettura video
    # ------------------------------------------------------------------
    def _candidate_id_at(self, position):
        return self.navigation_order[int(position)]

    def _row_at(self, position):
        return self.candidates.loc[self._candidate_id_at(position)]

    def _read_frame_near(self, target_frame):
        """Read a frame, skipping the costly seek when the target is close,
        which it usually is when stepping through in time order."""
        current = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        gap = target_frame - current
        if 0 <= gap <= max(1, int(self.video_fps * 5)):
            for _ in range(gap):
                if not self.video_capture.grab():
                    return False, None
        else:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return self.video_capture.read()

    # ------------------------------------------------------------------
    # Grafico
    # ------------------------------------------------------------------
    def _draw_graph_full(self):
        try:
            self._draw_graph_full_unsafe()
        except self._tk.TclError as error:
            # Out-of-range canvas coordinates must not block the rest of the
            # interface (buttons, checkboxes); the next redraw tries again.
            print(f"  > Preview graph: drawing skipped ({error}).")
        except Exception as error:
            # Only TclError used to be caught. Any other error mid-draw (a
            # non-finite value reaching this far, say) stopped the redraw with
            # nothing printed: on screen some ticks were simply missing, with no
            # way to find out why. Now every error is reported.
            print(f"  > Preview graph: drawing failed ({type(error).__name__}: {error}).")

    def _draw_graph_full_unsafe(self):
        graph = self.graph
        graph.delete("all")
        self.graph_candidate_hits.clear()
        self.rejected_hits.clear()
        self._point_items.clear()
        self._point_centers.clear()
        self._cross_items.clear()

        # A written summary that is always visible answers "why are frames
        # missing here" without hunting for a tiny tick to click.
        reason_counts = collections.Counter(
            self._reason_parts(reason)[0] for reason in self.rejected_frames.values()
        )
        if reason_counts:
            self.rejection_summary_text.set(
                "Excluded frames: "
                + ", ".join(
                    f"{count} {self._REASON_LABELS.get(reason, reason)}"
                    for reason, count in sorted(
                        reason_counts.items(), key=lambda item: -item[1]
                    )
                )
            )
        else:
            self.rejection_summary_text.set("No seconds excluded during screening.")

        graph.update_idletasks()
        width = max(300, graph.winfo_width())
        height = max(150, graph.winfo_height())
        # A slightly larger bottom margin leaves room for the strip of ticks
        # for rejected seconds under the time axis. The colour legend is a
        # separate widget, see _build.
        left, right, top, bottom = 58, 20, 24, 44
        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)

        candidates = self.candidates
        candidate_times = (
            pd.to_numeric(candidates["frame_pos"], errors="coerce")
            .to_numpy(dtype=float)
            / max(float(self.video_fps), 1e-9)
        )
        # The time axis spans the whole analysed window (start_sec..end_sec),
        # not just first-to-last candidate. Otherwise the altitude curve, which
        # covers the entire flight, runs off the edges instead of joining start
        # to end, and candidates bunched in time make the graph look wrongly
        # zoomed in.
        time_min, time_max = float(self.start_sec), float(self.end_sec)
        if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max <= time_min:
            time_min = float(np.nanmin(candidate_times)) if len(candidate_times) else 0.0
            time_max = float(np.nanmax(candidate_times)) if len(candidate_times) else time_min + 1.0
            if not np.isfinite(time_min) or not np.isfinite(time_max):
                time_min, time_max = 0.0, 1.0
        if time_max <= time_min:
            time_max = time_min + 1.0

        def map_x(seconds):
            return left + (seconds - time_min) / (time_max - time_min) * plot_width

        graph.create_line(left, height - bottom, width - right, height - bottom, fill="#aeb6bf")
        graph.create_text(left, height - 17, text=f"{time_min:.0f}s", fill="#ecf0f1", anchor="w")
        graph.create_text(width - right, height - 17, text=f"{time_max:.0f}s", fill="#ecf0f1", anchor="e")
        graph.create_text(
            width - right,
            10,
            text=(
                (
                    "Q1 blue   Q2 green   Q3 orange   Q4 purple   "
                    "red: length outlier / outside requested quartiles   "
                    "white outline: current   grey X: excluded"
                    if "_screening_quartile" in candidates.columns
                    else "light blue: candidate   white outline: current   grey X: excluded"
                )
                + (
                    "   filled green: already accepted   filled grey: already discarded"
                    if self.locked_status_by_position
                    else ""
                )
            ),
            fill="#ecf0f1",
            anchor="e",
        )

        # Strip of frames rejected during screening, coloured by reason. It
        # shows whether a gap in the timeline comes from altitude, tilt, YOLO
        # missing the animal, a mask unlike the mean shape, or no automatic
        # measurement.
        #
        # One tick per FRAME, not per second: with scan_fps > 1 two samples of
        # the same second can fail for different reasons, and both stay visible
        # even if a third sample of that second was accepted.
        #
        # The colour legend is a real widget below the graph (see _build); only
        # the ticks are drawn here. The strip and the ticks are wider than they
        # once were (8px/3px), which was too small to notice or click reliably.
        ribbon_top = height - bottom + 3
        ribbon_bottom = height - bottom + 18

        # Altitude and tilt come from the pre-filter, one entry per SECOND
        # rather than per frame, and usually cover a long unbroken stretch. One
        # thin tick each made that hard to read as a single excluded block, so
        # each run of consecutive seconds with the same reason is filled as one
        # rectangle instead. The other reasons, usually scattered, keep their
        # individual ticks.
        RANGE_REASONS = {"altitude", "tilt"}
        range_entries = sorted(
            (frame_number, self._reason_parts(reason)[0])
            for frame_number, reason in self.rejected_frames.items()
            if self._reason_parts(reason)[0] in RANGE_REASONS
        )
        range_covered_frames = set()
        frame_step = max(1.0, float(self.video_fps))
        run = None  # (first_frame, last_frame, reason) of the current run

        def flush_run():
            if run is None:
                return
            start_frame, end_frame, base = run
            start_time = start_frame / max(float(self.video_fps), 1e-9)
            end_time = (end_frame + frame_step) / max(float(self.video_fps), 1e-9)
            if end_time <= time_min or start_time >= time_max:
                return
            x_start = map_x(max(start_time, time_min))
            x_end = map_x(min(end_time, time_max))
            graph.create_rectangle(
                x_start, ribbon_top, x_end, ribbon_bottom,
                fill=self._REASON_COLORS.get(base, "#bdc3c7"), outline="",
            )

        for frame_number, base in range_entries:
            if run is not None and run[2] == base and frame_number - run[1] <= frame_step * 1.5:
                run = (run[0], frame_number, base)
            else:
                flush_run()
                run = (frame_number, frame_number, base)
            range_covered_frames.add(frame_number)
        flush_run()

        skipped_out_of_range = []
        for frame_number, reason in self.rejected_frames.items():
            frame_time = frame_number / max(float(self.video_fps), 1e-9)
            if not (time_min <= frame_time <= time_max):
                # Should never happen, since rejected_frames is built over the
                # same time_min..time_max range. If it does, the tick would
                # vanish silently, so say so here rather than let it show up as
                # an unexplained second in the graph.
                skipped_out_of_range.append((frame_number, reason, frame_time))
                continue
            color = self._REASON_COLORS.get(self._reason_parts(reason)[0], "#bdc3c7")
            x_coord = map_x(frame_time)
            if frame_number not in range_covered_frames:
                graph.create_line(x_coord, ribbon_top, x_coord, ribbon_bottom, fill=color, width=5)
            if frame_number in self.manual_selected_frames:
                # Mark the candidate-less frames added for manual annotation.
                # Otherwise they blended in with the other frames rejected for
                # the same reason, with no way to remember which were picked.
                graph.create_oval(
                    x_coord - 4, ribbon_top - 7, x_coord + 4, ribbon_top - 1,
                    outline="#2ecc71", width=2,
                )
            # The clickable area is wider than the visible tick, so a thin tick
            # is still easy to hit.
            self.rejected_hits.append(
                (float(x_coord), float((ribbon_top + ribbon_bottom) / 2), frame_number, reason)
            )

        if skipped_out_of_range:
            print(
                f"  > Preview graph: {len(skipped_out_of_range)} excluded-second "
                f"tick(s) fell outside the visible time range and were not drawn "
                f"(video_fps={self.video_fps}, range {time_min:.1f}-{time_max:.1f}s): "
                + ", ".join(
                    f"frame {fn} ({self._reason_parts(r)[0]}, computed {ft:.1f}s)"
                    for fn, r, ft in skipped_out_of_range[:5]
                )
            )

        telemetry = self.telemetry
        altitude_curve_available = (
            telemetry.available
            and len(telemetry.seconds) > 1
            and len(telemetry.altitudes) > 1
        )
        if altitude_curve_available:
            telemetry_mask = (telemetry.seconds >= time_min) & (telemetry.seconds <= time_max)
            curve_times = telemetry.seconds[telemetry_mask]
            curve_altitudes = telemetry.altitudes[telemetry_mask]
            if len(curve_times) < 2:
                curve_times = telemetry.seconds
                curve_altitudes = telemetry.altitudes
            if len(curve_times) > 1000:
                graph_step = math.ceil(len(curve_times) / 1000)
                curve_times = curve_times[::graph_step]
                curve_altitudes = curve_altitudes[::graph_step]
            altitude_min = float(np.nanmin(curve_altitudes))
            altitude_max = float(np.nanmax(curve_altitudes))
            if altitude_max <= altitude_min:
                altitude_max = altitude_min + 1.0

            def map_y(altitude):
                return top + (altitude_max - altitude) / (altitude_max - altitude_min) * plot_height

            # A candidate's altitude (get_nearest_altitude) is a step, not an
            # interpolation: it holds the last known sample until the next one
            # (see telemetry._sample_at_or_before). Drawing the curve as
            # straight lines between samples looked right, but the candidate
            # dots, which use the step, then sat off the line everywhere except
            # exactly on a sample. Drawn as steps, the two always agree.
            #
            # A gap between consecutive samples wider than gap_threshold (the
            # same limit that warns in telemetry.load_telemetry) is left
            # unconnected: there is no laser reading for that stretch, so no
            # line, just a dashed orange box. A smooth line there would suggest
            # an altitude we do not have.
            gap_threshold = 1.0
            gaps = []
            if time_min < curve_times[0] - 1e-9:
                gaps.append((time_min, float(curve_times[0])))
            segment_points = [map_x(curve_times[0]), map_y(curve_altitudes[0])]
            for i in range(1, len(curve_times)):
                if curve_times[i] - curve_times[i - 1] > gap_threshold:
                    if len(segment_points) >= 4:
                        graph.create_line(*segment_points, fill="#5dade2", width=2)
                    gaps.append((float(curve_times[i - 1]), float(curve_times[i])))
                    segment_points = [map_x(curve_times[i]), map_y(curve_altitudes[i])]
                    continue
                # Step: hold the previous altitude up to the new time, then
                # jump to the new value. No diagonal.
                segment_points.extend((map_x(curve_times[i]), segment_points[-1]))
                segment_points.extend((map_x(curve_times[i]), map_y(curve_altitudes[i])))
            if len(segment_points) >= 4:
                graph.create_line(*segment_points, fill="#5dade2", width=2)
            if time_max > curve_times[-1] + 1e-9:
                gaps.append((float(curve_times[-1]), time_max))

            for gap_start, gap_end in gaps:
                x_start, x_end = map_x(max(gap_start, time_min)), map_x(min(gap_end, time_max))
                if x_end - x_start < 1:
                    continue
                graph.create_rectangle(
                    x_start, top, x_end, top + plot_height,
                    outline="#e67e22", dash=(4, 3), width=1,
                )
                if x_end - x_start > 50:
                    graph.create_text(
                        (x_start + x_end) / 2, top + plot_height / 2,
                        text="no laser data", fill="#e67e22", angle=90,
                    )

            graph.create_text(8, top, text=f"{altitude_max:.1f} m", fill="#ecf0f1", anchor="w")
            graph.create_text(8, height - bottom, text=f"{altitude_min:.1f} m", fill="#ecf0f1", anchor="w")
            graph.create_text(left, 10, text="Laser altitude over time", fill="#ecf0f1", anchor="w")
        else:
            map_y = lambda _altitude: top + plot_height / 2
            graph.create_text(
                left,
                10,
                text="Altitude telemetry not available — candidates shown by time position",
                fill="#ecf0f1",
                anchor="w",
            )

        current_candidate_id = self._candidate_id_at(self.current_position.get())
        quartile_colors = {
            1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#9b59b6",
            self.OUTLIER_QUARTILE: "#ff4757",
        }
        skipped_candidates = []
        for candidate_id, seconds in enumerate(candidate_times):
            if not np.isfinite(seconds):
                # frame_pos missing or non-numeric on this row: a candidate
                # accepted in screening that would never be drawn here. Report
                # it rather than let it become an accepted second that appears
                # nowhere.
                skipped_candidates.append(candidate_id)
                continue
            altitude = get_altitude(telemetry, seconds) if altitude_curve_available else np.nan
            # A candidate can have no altitude even when the altitude curve
            # exists, for instance when its second falls before the very first
            # telemetry sample (see telemetry._sample_at_or_before). map_y(nan)
            # would hand the canvas invalid coordinates and silently stop the
            # rest of the drawing, every later candidate included. Such a
            # candidate is placed at the bottom of the graph with its own marker
            # instead, so it stays visible and clickable.
            laser_missing = altitude_curve_available and not np.isfinite(altitude)
            x_coord = map_x(seconds)
            y_coord = (top + plot_height) if laser_missing else map_y(altitude)
            quartile_value = (
                candidates.iloc[candidate_id]["_screening_quartile"]
                if "_screening_quartile" in candidates.columns
                else np.nan
            )
            quartile = int(quartile_value) if pd.notna(quartile_value) else 0
            locked_here = self.locked_status_by_position.get(candidate_id)
            if locked_here is not None:
                color = "#2ecc71" if locked_here == "accepted" else "#7f8c8d"
            elif laser_missing:
                color = "#e67e22"
            else:
                color = quartile_colors.get(quartile, "#5dade2")
            is_current = candidate_id == current_candidate_id
            radius = 6 if is_current else 4
            oval_id = graph.create_oval(
                x_coord - radius,
                y_coord - radius,
                x_coord + radius,
                y_coord + radius,
                fill=color,
                outline="#ffffff" if is_current else ("#e67e22" if laser_missing else ""),
                width=2 if (is_current or laser_missing) else 1,
                dash=(2, 2) if laser_missing else None,
            )
            self._point_items[candidate_id] = oval_id
            self._point_centers[candidate_id] = (x_coord, y_coord)
            if laser_missing:
                graph.create_text(
                    x_coord, y_coord - radius - 8,
                    text="⚠ no laser", fill="#e67e22", font=("", 8, "bold"),
                )
            if locked_here is not None:
                graph.create_text(x_coord, y_coord, text="✓", fill="#ffffff", font=("", 8, "bold"))
            elif candidate_id not in self.selected_positions:
                cross_radius = radius + 2
                cross_line_1 = graph.create_line(
                    x_coord - cross_radius, y_coord - cross_radius,
                    x_coord + cross_radius, y_coord + cross_radius,
                    fill="#566573", width=2,
                )
                cross_line_2 = graph.create_line(
                    x_coord - cross_radius, y_coord + cross_radius,
                    x_coord + cross_radius, y_coord - cross_radius,
                    fill="#566573", width=2,
                )
                self._cross_items[candidate_id] = (cross_line_1, cross_line_2)
            self.graph_candidate_hits.append((float(x_coord), float(y_coord), candidate_id))

        if skipped_candidates:
            frame_values = candidates.iloc[skipped_candidates]["frame_pos"].tolist()
            print(
                f"  > Preview graph: {len(skipped_candidates)} accepted "
                f"candidate(s) had a non-numeric/missing frame_pos and were "
                f"not drawn: {frame_values}"
            )

    def _refresh_current_marker(self, old_position, new_position):
        """Move just the highlight on the current candidate, instead of
        redrawing the whole graph. Keeps navigation quick with many candidates,
        or with an arrow key held down."""
        try:
            old_id = self._candidate_id_at(old_position)
            new_id = self._candidate_id_at(new_position)
            for candidate_id, is_current in ((old_id, False), (new_id, True)):
                item_id = self._point_items.get(candidate_id)
                center = self._point_centers.get(candidate_id)
                if item_id is None or center is None:
                    continue
                x, y = center
                radius = 6 if is_current else 4
                self.graph.coords(item_id, x - radius, y - radius, x + radius, y + radius)
                self.graph.itemconfigure(
                    item_id,
                    outline="#ffffff" if is_current else "",
                    width=2 if is_current else 1,
                )
                if is_current:
                    self.graph.tag_raise(item_id)
                    # The "excluded" X has to stay on top of the dot even after
                    # the dot is raised to the front.
                    for cross_item_id in self._cross_items.get(candidate_id, ()):
                        self.graph.tag_raise(cross_item_id)
        except self._tk.TclError as error:
            print(f"  > Preview graph: highlight skipped ({error}).")

    # ------------------------------------------------------------------
    # Zoom/pan manuali sull'immagine (rotellina + trascinamento)
    # ------------------------------------------------------------------
    def _preview_window(self, source, max_width, max_height):
        """Crop `source` to the current zoom and pan, then scale it to fill
        the panel.

        `source` is already framed on bbox plus margin, or the whole frame for
        rejected frames. Both _show_current_frame and _show_rejected_frame use
        this so they share one implementation.

        Returns (image, scale, origin of the window inside `source`). That
        origin has to be added to the outer crop's own origin (the bbox, say)
        to keep the overlays drawn later — YOLO and SAM boxes, points — in the
        right place.
        """
        source_height, source_width = source.shape[:2]
        # "Fit" means the whole source centred in the panel, capped at 6x so a
        # tiny crop (a distant whale) does not blow up. self._preview_zoom
        # multiplies THAT fit: 1.0 is the default fit, below 1 shrinks it
        # further (empty space around it in the fixed panel), above 1 grows it
        # past the panel, so a smaller window of the source is cropped out and
        # can be dragged around.
        fit_scale = min(max_width / source_width, max_height / source_height, 6.0)
        scale = fit_scale * self._preview_zoom
        if scale <= fit_scale + 1e-9:
            display_width = max(1, round(source_width * scale))
            display_height = max(1, round(source_height * scale))
            resized = cv2.resize(
                source,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            self._preview_last_scale = scale
            self._preview_last_source_size = (source_width, source_height)
            return resized, scale, (0, 0)

        window_width = max(1, max_width / scale)
        window_height = max(1, max_height / scale)
        center_x = self._preview_pan[0] * source_width
        center_y = self._preview_pan[1] * source_height
        x0 = int(np.clip(center_x - window_width / 2, 0, max(0, source_width - window_width)))
        y0 = int(np.clip(center_y - window_height / 2, 0, max(0, source_height - window_height)))
        window = source[y0 : y0 + int(round(window_height)), x0 : x0 + int(round(window_width))]
        if window.size == 0:
            window = source
            x0, y0 = 0, 0
        display_width = max(1, round(window.shape[1] * scale))
        display_height = max(1, round(window.shape[0] * scale))
        resized = cv2.resize(window, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        self._preview_last_scale = scale
        self._preview_last_source_size = (source_width, source_height)
        return resized, scale, (x0, y0)

    def _redraw_current_preview(self):
        if self._viewing_rejected and self._current_rejected_frame is not None:
            self._show_rejected_frame(self._current_rejected_frame, self._current_rejected_reason)
        else:
            self._show_current_frame()

    def _on_preview_wheel(self, event):
        direction = 0
        if getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        elif getattr(event, "delta", 0):
            direction = 1 if event.delta > 0 else -1
        if direction != 0:
            # A floor of 1.0 means the whole frame stays visible: zooming out
            # reveals the rest of the real frame instead of shrinking it inside
            # empty bands. The ceiling is generous so even a tiny bounding box
            # (a small whale in a 4K frame) can reach the default framing and
            # go beyond it for detail.
            self._preview_zoom = float(np.clip(self._preview_zoom * (1.15**direction), 1.0, 60.0))
            self._redraw_current_preview()
        return "break"

    def _on_preview_drag_start(self, event):
        self._preview_drag_origin = (event.x, event.y, tuple(self._preview_pan))

    def _on_preview_drag_move(self, event):
        if self._preview_drag_origin is None or self._preview_zoom <= 1.0001:
            return
        start_x, start_y, start_pan = self._preview_drag_origin
        scale = self._preview_last_scale or 1.0
        source_width, source_height = self._preview_last_source_size
        delta_x_source = (event.x - start_x) / scale
        delta_y_source = (event.y - start_y) / scale
        self._preview_pan = [
            float(np.clip(start_pan[0] - delta_x_source / max(source_width, 1), 0.0, 1.0)),
            float(np.clip(start_pan[1] - delta_y_source / max(source_height, 1), 0.0, 1.0)),
        ]
        self._redraw_current_preview()

    def _on_preview_reset(self, _event=None):
        default_zoom, default_pan = self._preview_default
        self._preview_zoom = default_zoom
        self._preview_pan = list(default_pan)
        self._redraw_current_preview()

    def _on_root_configure(self, event):
        # <Configure> also fires for every child widget that moves or resizes,
        # since the event bubbles up. Only the WINDOW's own resize matters here,
        # not the inner content, which would change with every candidate: the
        # very shuffling the fixed-size panel exists to prevent.
        if event.widget is not self.root:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            150, lambda: self._apply_preview_panel_size(event.width, event.height)
        )

    def _apply_preview_panel_size(self, window_width, window_height):
        self._resize_after_id = None
        # Space set aside for the rest of the interface above and below the
        # image (header, graph, legend, controls, scrollbar). A fixed estimate,
        # not pixel-exact, but it lets the image panel take whatever is left, so
        # the window can grow to nearly full screen and the image grows with it.
        new_width = max(640, window_width - 40)
        new_height = max(360, window_height - 470)
        if (new_width, new_height) == self._PREVIEW_PANEL_SIZE:
            return
        self._PREVIEW_PANEL_SIZE = (new_width, new_height)
        self.image_stage.configure(width=new_width, height=new_height)
        # The default zoom/pan and the cached frames were computed for the old
        # panel size, so drop them rather than leave them wrong until the next
        # candidate.
        self._frame_cache.clear()
        self._preview_key = None
        if self._preview_ready:
            self._redraw_current_preview()

    # ------------------------------------------------------------------
    # Updating the frame on screen
    # ------------------------------------------------------------------
    def _show_current_frame(self, *_args):
        self._viewing_rejected = False
        candidate_count = self.candidate_count
        position = int(np.clip(self.current_position.get(), 0, candidate_count - 1))
        self.current_position.set(position)
        row = self._row_at(position)
        candidate_id = self._candidate_id_at(position)
        frame_number = int(row["frame_pos"])
        preview_key = ("candidate", frame_number)

        max_width = self._PREVIEW_PANEL_SIZE[0] - 12
        max_height = self._PREVIEW_PANEL_SIZE[1] - 12

        # A small store of ready frames makes going back to a candidate just
        # seen instant. Seeking backwards in a video is far slower than forwards
        # (which _read_frame_near already optimises), and here the video does
        # not even have to be touched.
        #
        # The cache holds the WHOLE frame, with the mask tinted only over the
        # bbox area. Manual zoom and pan crop a window out of it on each redraw,
        # which is too cheap to be worth caching per wheel click or drag. It
        # also means zooming out shows the real frame, not a shrunken crop.
        show_mask = self.show_mask.get()
        cache_key = show_mask
        cached = self._frame_cache.get(frame_number)
        if cached is not None and cached[0] == cache_key:
            self._frame_cache.move_to_end(frame_number)
            _, full_source, default_zoom, default_pan = cached
        else:
            success, raw_frame = self._read_frame_near(frame_number)
            if not success:
                self.frame_info.set(
                    f"Candidate {position + 1}/{candidate_count} — "
                    f"frame {frame_number} not readable"
                )
                self.image_label.configure(image="")
                self.preview_photo = None
                return

            full_source = raw_frame
            full_height, full_width = raw_frame.shape[:2]
            # Default zoom and pan frame the subject (bbox plus margin), and
            # the user zooms out from there to see the rest. The margin is the
            # same one used to crop the mask during screening, so they match.
            default_zoom, default_pan = 1.0, [0.5, 0.5]
            if all(column in row.index for column in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
                bbox_xyxy = np.array(
                    [[row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]],
                    dtype=np.float32,
                )
                cropped, crop_coords = expanded_bbox_crop(raw_frame, bbox_xyxy, _PREVIEW_ZOOM_MARGIN)
                if cropped is not None and cropped.size > 0:
                    crop_height, crop_width = cropped.shape[:2]
                    fit_scale_full = min(max_width / full_width, max_height / full_height, 6.0)
                    fit_scale_bbox = min(max_width / crop_width, max_height / crop_height, 6.0)
                    default_zoom = float(
                        np.clip(fit_scale_bbox / max(fit_scale_full, 1e-9), 1.0, 60.0)
                    )
                    default_pan = [
                        (crop_coords[0] + crop_width / 2) / full_width,
                        (crop_coords[1] + crop_height / 2) / full_height,
                    ]

                    mask_crop = row.get("_auto_mask_crop")
                    if (
                        show_mask
                        and isinstance(mask_crop, np.ndarray)
                        and mask_crop.size > 0
                        and mask_crop.shape[:2] == cropped.shape[:2]
                    ):
                        # The mask is tinted at 20% opacity, only over the bbox
                        # it came from: it shows what the automatic points are
                        # based on without hiding the animal.
                        x0c, y0c = crop_coords[0], crop_coords[1]
                        region = full_source[y0c : y0c + crop_height, x0c : x0c + crop_width]
                        tinted = region.copy()
                        tinted[mask_crop > 0] = (0, 0, 255)
                        full_source = full_source.copy()
                        full_source[y0c : y0c + crop_height, x0c : x0c + crop_width] = cv2.addWeighted(
                            tinted, 0.20, region, 0.80, 0
                        )

            self._frame_cache[frame_number] = (cache_key, full_source, default_zoom, default_pan)
            self._frame_cache.move_to_end(frame_number)
            while len(self._frame_cache) > 60:
                self._frame_cache.popitem(last=False)

        if preview_key != self._preview_key:
            # New candidate: go back to the default view (zoomed on the
            # subject) instead of inheriting the previous frame's zoom and pan.
            self._preview_zoom = default_zoom
            self._preview_pan = list(default_pan)
            self._preview_key = preview_key
        self._preview_default = (default_zoom, default_pan)

        # Unlike review, the subject can be magnified past its original size
        # here, not just shrunk to fit the panel. Both the default framing and
        # manual zoom/pan go through the same window on the whole frame, so
        # crop_origin is already in original-frame coordinates and never has to
        # be combined with an outer crop.
        resized_frame, scale, crop_origin = self._preview_window(full_source, max_width, max_height)

        # Contrast is applied after resizing, so the lookup table works on the
        # preview rather than the full-resolution crop, saving time and peak
        # memory. `enhance_review_frame` always returns a new array and never
        # modifies its input, so the cached crop stays intact.
        frame = enhance_review_frame(
            resized_frame,
            contrast=self.preview_contrast.get() / 100.0,
            brightness=0,
            use_clahe=False,
        )
        if self.show_yolo_box.get() and all(
            column in row.index for column in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        ):
            # The tight box YOLO and tracking use, drawn so it is obvious at a
            # glance when it cuts the animal short (the tip of the head, say)
            # before SAM ever sees the frame.
            box_topleft = (
                int((row["bbox_x1"] - crop_origin[0]) * scale),
                int((row["bbox_y1"] - crop_origin[1]) * scale),
            )
            box_bottomright = (
                int((row["bbox_x2"] - crop_origin[0]) * scale),
                int((row["bbox_y2"] - crop_origin[1]) * scale),
            )
            cv2.rectangle(frame, box_topleft, box_bottomright, (0, 165, 255), 1, cv2.LINE_AA)
            draw_readable_text(frame, "YOLO", (box_topleft[0] + 4, box_topleft[1] + 14), 0.5, (0, 165, 255), 1)
        if self.show_sam_box.get() and all(
            column in row.index
            for column in ("sam_bbox_x1", "sam_bbox_y1", "sam_bbox_x2", "sam_bbox_y2")
        ):
            # The box grown by sam_box_margin: what SAM actually gets as a
            # prompt, shown next to YOLO's tight box for comparison.
            sam_topleft = (
                int((row["sam_bbox_x1"] - crop_origin[0]) * scale),
                int((row["sam_bbox_y1"] - crop_origin[1]) * scale),
            )
            sam_bottomright = (
                int((row["sam_bbox_x2"] - crop_origin[0]) * scale),
                int((row["sam_bbox_y2"] - crop_origin[1]) * scale),
            )
            cv2.rectangle(frame, sam_topleft, sam_bottomright, (255, 0, 255), 1, cv2.LINE_AA)
            draw_readable_text(frame, "SAM", (sam_topleft[0] + 4, sam_bottomright[1] - 6), 0.5, (255, 0, 255), 1)
        if self.show_length.get() and all(
            column in row.index and pd.notna(row[column])
            for column in ("_auto_head_x", "_auto_head_y", "_auto_notch_x", "_auto_notch_y")
        ):
            # The head and fluke points measure_length_pca used for the
            # automatic length, so it is clear where the shown value came from,
            # especially when it looks wrong.
            head_point = (
                int((row["_auto_head_x"] - crop_origin[0]) * scale),
                int((row["_auto_head_y"] - crop_origin[1]) * scale),
            )
            notch_point = (
                int((row["_auto_notch_x"] - crop_origin[0]) * scale),
                int((row["_auto_notch_y"] - crop_origin[1]) * scale),
            )
            cv2.line(frame, head_point, notch_point, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, head_point, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, notch_point, 6, (0, 0, 255), -1, cv2.LINE_AA)
            draw_readable_text(frame, "T", (head_point[0] + 10, head_point[1] - 10), 0.7, (0, 255, 0), 1)
            draw_readable_text(frame, "C", (notch_point[0] + 10, notch_point[1] - 10), 0.7, (0, 0, 255), 1)
            unit_short = "m" if self.unit_label == "meters" else "px"
            length_point = (
                (head_point[0] + notch_point[0]) // 2 - 20,
                (head_point[1] + notch_point[1]) // 2 - 14,
            )
            draw_readable_text(
                frame,
                f"{float(row['auto_length']):.2f} {unit_short}",
                length_point,
                0.7,
                (0, 255, 255),
                1,
            )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        # Reuse the same PhotoImage when the size has not changed rather than
        # making a new one per frame. If an update fails anyway (Tk under load
        # from fast navigation), it must not block the rest: the next frame
        # repairs the image.
        try:
            if (
                self.preview_photo is None
                or self.preview_photo.width() != pil_image.width
                or self.preview_photo.height() != pil_image.height
            ):
                self.preview_photo = ImageTk.PhotoImage(image=pil_image, master=self.root)
                self.image_label.configure(image=self.preview_photo)
                self.image_label.image = self.preview_photo
            else:
                self.preview_photo.paste(pil_image)
        except self._tk.TclError:
            self.preview_photo = None
        del rgb_frame, frame

        locked_here = self.locked_status_by_position.get(candidate_id)
        self.include_current.set(locked_here is None and candidate_id in self.selected_positions)
        self.include_checkbox.state(["disabled"] if locked_here is not None else ["!disabled"])
        seconds = frame_number / max(float(self.video_fps), 1e-9)
        altitude = get_nearest_altitude(self.telemetry, seconds)
        laser_missing = self.unit_label == "meters" and not np.isfinite(altitude)
        altitude_text = (
            f" • altitude {altitude:.2f} m"
            if np.isfinite(altitude)
            else (" • NO LASER MEASUREMENT for this second" if laser_missing else "")
        )
        automatic_value = float(row["auto_length"])
        if laser_missing:
            # Different from "mask found but no head/fluke": here the shape may
            # have measured fine in pixels, only the altitude needed to convert
            # to metres is missing.
            length_text = "no laser altitude for this second — length not computed in meters"
        elif np.isfinite(automatic_value):
            length_text = f"automatic length {automatic_value:.1f} {self.unit_label}"
        else:
            length_text = "no automatic measurement — snout/fluke not found, measure manually"
        if "_screening_quartile" in row.index and pd.notna(row["_screening_quartile"]):
            quartile_value = int(row["_screening_quartile"])
            quartile_text = (
                " • length outlier / outside requested quartiles"
                if quartile_value == self.OUTLIER_QUARTILE
                else f" • quartile Q{quartile_value}"
            )
        else:
            quartile_text = ""
        locked_text = f" • ALREADY DECIDED: {locked_here.upper()}" if locked_here is not None else ""
        self.frame_info.set(
            f"Candidate {position + 1}/{candidate_count} • "
            f"frame {frame_number} • second {seconds:.2f}{altitude_text} • "
            f"{length_text}"
            f"{quartile_text}{locked_text} • "
            f"order {self.navigation_mode.get().lower()}"
        )
        self._refresh_current_marker(self._last_shown_position, position)
        self._last_shown_position = position

    # ------------------------------------------------------------------
    # Azioni utente
    # ------------------------------------------------------------------
    def _move(self, step):
        self.current_position.set(
            int(np.clip(self.current_position.get() + step, 0, self.candidate_count - 1))
        )
        self._show_current_frame()

    def _toggle_current(self):
        if self._viewing_rejected:
            frame_number = self._current_rejected_frame
            if self.include_current.get():
                self.manual_selected_frames[frame_number] = self._current_rejected_frame_size
            else:
                self.manual_selected_frames.pop(frame_number, None)
            self._update_selected_count_text()
            self._draw_graph_full()
            return
        candidate_id = self._candidate_id_at(self.current_position.get())
        if candidate_id in self.locked_status_by_position:
            return
        if self.include_current.get():
            self.selected_positions.add(candidate_id)
        else:
            self.selected_positions.discard(candidate_id)
        self._update_selected_count_text()
        self._draw_graph_full()

    def _handle_space(self, _event=None):
        """The C key: same as clicking the include checkbox, so frames can be
        selected without leaving the keyboard while arrowing through them.

        It also works on a rejected frame with no candidate, where it adds or
        removes that frame from the fully manual annotation list.
        """
        if not self._viewing_rejected:
            candidate_id = self._candidate_id_at(self.current_position.get())
            if candidate_id in self.locked_status_by_position:
                return "break"
        self.include_current.set(not self.include_current.get())
        self._toggle_current()
        return "break"

    def _select_everything(self, selected):
        self.selected_positions.clear()
        if selected:
            self.selected_positions.update(
                position
                for position in range(self.candidate_count)
                if position not in self.locked_status_by_position
            )
        current_candidate_id = self._candidate_id_at(self.current_position.get())
        self.include_current.set(
            current_candidate_id not in self.locked_status_by_position
            and current_candidate_id in self.selected_positions
        )
        self._update_selected_count_text()
        self._draw_graph_full()

    def _change_navigation_mode(self, _event=None):
        current_candidate = self._candidate_id_at(self.current_position.get())
        self.navigation_order = candidate_navigation_order(self.candidates, self.navigation_mode.get())
        self.current_position.set(self.navigation_order.index(current_candidate))
        self._show_current_frame()
        self.root.focus_set()

    def _update_contrast_label(self, value):
        self.contrast_text.set(f"Contrast: {float(value) / 100.0:.2f}×")

    def _apply_preview_contrast(self, _event=None):
        self._show_current_frame()
        self.root.focus_set()

    def _hover_graph(self, event):
        """Raise the dot under the cursor without a click, which would instead
        jump to that candidate. See _jump_from_graph for the same
        nearest-point search."""
        if not self.graph_candidate_hits:
            return
        nearest_x, nearest_y, nearest_candidate = min(
            self.graph_candidate_hits,
            key=lambda marker: (marker[0] - event.x) ** 2 + (marker[1] - event.y) ** 2,
        )
        if math.hypot(nearest_x - event.x, nearest_y - event.y) > 18:
            return
        item_id = self._point_items.get(nearest_candidate)
        if item_id is None:
            return
        try:
            self.graph.tag_raise(item_id)
            for cross_item_id in self._cross_items.get(nearest_candidate, ()):
                self.graph.tag_raise(cross_item_id)
        except self._tk.TclError:
            pass

    def _jump_from_graph(self, event):
        best_candidate = None
        if self.graph_candidate_hits:
            nearest_x, nearest_y, nearest_candidate = min(
                self.graph_candidate_hits,
                key=lambda marker: (marker[0] - event.x) ** 2 + (marker[1] - event.y) ** 2,
            )
            candidate_distance = math.hypot(nearest_x - event.x, nearest_y - event.y)
            if candidate_distance <= 18:
                best_candidate = (candidate_distance, nearest_candidate)

        best_rejected = None
        if self.rejected_hits:
            nearest_x, nearest_y, nearest_frame_number, nearest_reason = min(
                self.rejected_hits,
                key=lambda marker: (marker[0] - event.x) ** 2 + (marker[1] - event.y) ** 2,
            )
            rejected_distance = math.hypot(nearest_x - event.x, nearest_y - event.y)
            if rejected_distance <= 20:
                best_rejected = (rejected_distance, nearest_frame_number, nearest_reason)

        if best_rejected is not None and (
            best_candidate is None or best_rejected[0] <= best_candidate[0]
        ):
            self._show_rejected_frame(best_rejected[1], best_rejected[2])
            self.root.focus_set()
            return

        if best_candidate is not None:
            self.current_position.set(self.navigation_order.index(best_candidate[1]))
            self._show_current_frame()
            self.root.focus_set()

    def _show_rejected_frame(self, frame_number, reason):
        """Show a frame that screening rejected, with that sample's own reason,
        after the user clicked it on the graph.

        No candidate was saved for this frame, so there is no bbox and no mask:
        the whole frame is shown with no zoom and no overlays.
        """
        success, raw_frame = self._read_frame_near(frame_number)
        label = self._reason_label(reason)
        second = frame_number / max(float(self.video_fps), 1e-9)
        self._viewing_rejected = True
        self._current_rejected_frame = frame_number
        self._current_rejected_reason = reason
        preview_key = ("rejected", frame_number)
        if preview_key != self._preview_key:
            self._preview_zoom = 1.0
            self._preview_pan = [0.5, 0.5]
            self._preview_key = preview_key
        self._preview_default = (1.0, [0.5, 0.5])
        if not success:
            self.frame_info.set(
                f"Frame {frame_number} (second {second:.2f}) — EXCLUDED ({label}) "
                "— frame not readable"
            )
            self.image_label.configure(image="")
            self.preview_photo = None
            self.include_current.set(False)
            self.include_checkbox.state(["disabled"])
            return

        max_width = self._PREVIEW_PANEL_SIZE[0] - 12
        max_height = self._PREVIEW_PANEL_SIZE[1] - 12
        resized, _scale, _window_origin = self._preview_window(raw_frame, max_width, max_height)
        frame = enhance_review_frame(
            resized,
            contrast=self.preview_contrast.get() / 100.0,
            brightness=0,
            use_clahe=False,
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        try:
            if (
                self.preview_photo is None
                or self.preview_photo.width() != pil_image.width
                or self.preview_photo.height() != pil_image.height
            ):
                self.preview_photo = ImageTk.PhotoImage(image=pil_image, master=self.root)
                self.image_label.configure(image=self.preview_photo)
                self.image_label.image = self.preview_photo
            else:
                self.preview_photo.paste(pil_image)
        except self._tk.TclError:
            self.preview_photo = None

        self._current_rejected_frame_size = (raw_frame.shape[1], raw_frame.shape[0])
        self.include_current.set(frame_number in self.manual_selected_frames)
        self.include_checkbox.state(["!disabled"])
        self.frame_info.set(
            f"Frame {frame_number} (second {second:.2f}) — EXCLUDED: {label} "
            "— no automatic candidate; tick the box (or press C) to add it "
            "for fully manual annotation, or use the arrow keys / click a "
            "dot to go back to the candidates"
        )

    def _handle_arrow(self, _event, step):
        self._move(step)
        return "break"

    def _close_window(self):
        self.image_label.configure(image="")
        self.image_label.image = None
        self.preview_photo = None
        self.root.withdraw()
        self.root.quit()

    def _finish_selection(self):
        if not self.selected_positions and not self.manual_selected_frames:
            self._messagebox.showwarning(
                "No frame selected",
                "Select at least one frame to analyze.",
                parent=self.root,
            )
            return
        self.result_positions.extend(
            candidate_id
            for candidate_id in self.navigation_order
            if candidate_id in self.selected_positions
        )
        self._close_window()

    def _request_settings(self):
        if ask_confirm(
            self.root,
            "Back to settings?",
            "Going back to settings will discard the frame selection made "
            "so far in this window.",
            yes_label="Yes, discard selection and go to settings",
            no_label="No, keep selecting frames",
        ):
            self.return_to_settings = True
            self._close_window()


def _format_second_ranges(seconds):
    """Comprime un insieme di secondi in intervalli leggibili, es. {5,6,7,9} -> '5-7s, 9s'."""
    ordered = sorted(seconds)
    if not ordered:
        return ""
    ranges = []
    range_start = range_end = ordered[0]
    for value in ordered[1:]:
        if value == range_end + 1:
            range_end = value
        else:
            ranges.append((range_start, range_end))
            range_start = range_end = value
    ranges.append((range_start, range_end))
    return ", ".join(
        f"{start}s" if start == end else f"{start}-{end}s"
        for start, end in ranges
    )


def _box_matches_any(box_xyxy, other_boxes_xyxy, iou_threshold=0.3):
    """True if `box_xyxy` overlaps (by IoU) any box in `other_boxes_xyxy`
    enough to be the same animal seen at a different confidence threshold,
    rather than an extra detection.

    A pure function. run_screening uses it to tell whether a box found during
    the low-confidence recheck is an animal already handled at the main
    confidence, or evidence of a second untracked animal.
    """
    other_boxes_xyxy = np.asarray(other_boxes_xyxy, dtype=np.float64)
    if len(other_boxes_xyxy) == 0:
        return False
    x1 = np.maximum(box_xyxy[0], other_boxes_xyxy[:, 0])
    y1 = np.maximum(box_xyxy[1], other_boxes_xyxy[:, 1])
    x2 = np.minimum(box_xyxy[2], other_boxes_xyxy[:, 2])
    y2 = np.minimum(box_xyxy[3], other_boxes_xyxy[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    box_area = max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])
    other_areas = (
        np.clip(other_boxes_xyxy[:, 2] - other_boxes_xyxy[:, 0], 0, None)
        * np.clip(other_boxes_xyxy[:, 3] - other_boxes_xyxy[:, 1], 0, None)
    )
    union = box_area + other_areas - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return bool(np.any(iou >= iou_threshold))


def classify_length_quartiles(auto_length, target_quartiles):
    """Sort every `auto_length` value (a pd.Series) into a size quartile.

    0 means no automatic length; 1-4 are the quartiles, with boundaries
    computed only over the values inside 1.5x IQR so extreme outliers cannot
    shift them; CandidatePreviewWindow.OUTLIER_QUARTILE marks an IQR outlier or
    anything outside `target_quartiles` (empty or None means all four are
    wanted, so no quartile filtering).

    A pure function: no models or video, just statistics on a Series. Used by
    run_screening (phase 2) and tested directly in smoke_test.py, since
    run_screening itself needs a real video and the real models.

    Returns (quartile, is_iqr_outlier, outside_requested_quartile, q0, q4).
    """
    no_length = auto_length.isna()
    q1 = auto_length.quantile(0.25)
    q2 = auto_length.quantile(0.50)
    q3 = auto_length.quantile(0.75)
    iqr = q3 - q1
    within_iqr = no_length | (
        (auto_length >= q1 - 1.5 * iqr) & (auto_length <= q3 + 1.5 * iqr)
    )
    q0 = auto_length[within_iqr].min()
    q4 = auto_length[within_iqr].max()
    quartile = pd.Series(
        np.select(
            [no_length, auto_length <= q1, auto_length <= q2, auto_length <= q3],
            [0, 1, 2, 3],
            default=4,
        ),
        index=auto_length.index,
    ).astype(int)
    is_iqr_outlier = ~within_iqr
    outside_requested_quartile = ~no_length & ~quartile.isin(target_quartiles or [1, 2, 3, 4])
    outlier_mask = is_iqr_outlier | outside_requested_quartile
    quartile = quartile.where(~outlier_mask, CandidatePreviewWindow.OUTLIER_QUARTILE)
    return quartile, is_iqr_outlier, outside_requested_quartile, q0, q4


def run_screening(config, telemetry, video_path, output_path, screening_checkpoint_path):
    """FASE 1 (scansione + PCA) e FASE 2 (selezione statistica + anteprima)."""
    # Delete any mean_shape_masks*.png left by an earlier analysis, the
    # per-whale_id variants of a past multi-whale run included. Without this, a
    # scan that never computes a new one (mean-shape filter off, or fewer
    # animals than last time) would leave old images in the folder looking like
    # they belong to the current analysis.
    for stale_mean_shape_path in output_path.glob("mean_shape_masks*.png"):
        stale_mean_shape_path.unlink()

    measure_in_meters = config.measurement_unit == "Meters"
    pixel_size = config.sensor_width_mm / config.image_width_px
    focal_length = config.focal_length_mm
    start_sec, end_sec = config.start_sec, config.end_sec
    target_quartiles = (
        sorted(
            {
                int(q.strip())
                for q in config.target_quartiles.split(",")
                if q.strip()
            }
        )
        if measure_in_meters
        else []
    )

    unit_label = "meters" if measure_in_meters else "pixels"
    print(
        "PHASE 1: Video screening and automatic length estimation "
        f"in {unit_label} (in-line PCA)..."
    )

    # --- PHASE 1A: pre-filter on altitude stability ---
    # Why each second was excluded before any frame was even read (unstable
    # altitude or tilt). A second -> reason dict is fine at this stage, since no
    # particular frame has been tried yet. It is converted to frame keys further
    # down, as soon as fps is known, and merged into rejected_frames, which is
    # keyed per frame rather than per second so one rejected sample neither
    # hides nor is hidden by another sample of the same second.
    rejected_seconds_prefilter = {}
    valid_seconds = set()
    rejected_seconds_by_tilt = set()
    first_second = math.floor(start_sec)
    last_second = math.ceil(end_sec)
    if measure_in_meters:
        print(f"  > Telemetry analysis: maximum tolerance {config.max_alt_diff}m/s")
        # Second `sec` covers the frames from `sec` to `sec + 1`
        # (int(frame_time) == sec). That interval is unstable only if the
        # altitude difference between `sec` and `sec + 1` exceeds the limit: a
        # jump in the pair (sec, sec+1) affects [sec, sec+1) only, not the next
        # interval, which depends on its own pair (sec+1, sec+2).
        unstable_seconds = set()
        for sec in range(first_second, last_second):
            alt_curr = get_altitude(telemetry, sec)
            alt_next = get_altitude(telemetry, sec + 1)
            if abs(alt_next - alt_curr) > abs(config.max_alt_diff):
                unstable_seconds.add(sec)

        for sec in range(first_second, last_second + 1):
            altitude_is_stable = sec not in unstable_seconds
            tilt_is_valid = True
            if config.use_tilt_filter and telemetry.tilt_available:
                tilt_is_valid = get_tilt(telemetry, sec) <= config.max_tilt_deg
                if not tilt_is_valid:
                    rejected_seconds_by_tilt.add(sec)

            if altitude_is_stable and tilt_is_valid:
                valid_seconds.add(sec)

        print(
            "  > Stable flight seconds identified: "
            f"{len(valid_seconds)} of {last_second - first_second + 1}"
        )
        if unstable_seconds:
            print(
                f"  > Seconds excluded for altitude instability "
                f"({len(unstable_seconds)}): "
                f"{_format_second_ranges(unstable_seconds)}"
            )
        if config.use_tilt_filter and telemetry.tilt_available and rejected_seconds_by_tilt:
            print(
                f"  > tilt_deg ≤ {config.max_tilt_deg:g}° filter: "
                f"{len(rejected_seconds_by_tilt)} seconds excluded: "
                f"{_format_second_ranges(rejected_seconds_by_tilt)}"
            )
        # If a second fails both altitude and tilt, record it as "altitude",
        # the filter that conceptually comes first.
        for sec in unstable_seconds:
            rejected_seconds_prefilter[sec] = "altitude"
        for sec in rejected_seconds_by_tilt:
            rejected_seconds_prefilter.setdefault(sec, "tilt")
    else:
        valid_seconds.update(range(first_second, last_second + 1))
        print(
            "  > Pixel mode: telemetry, altitude filter, tilt and quartiles "
            "are not used to select frames."
        )

    # --- FASE 1B: Estrazione visiva e PCA ---
    yolo_model = device.get_yolo_model()
    sequential_indexing = use_sequential_indexing(config)
    cap = open_capture(video_path, sequential=sequential_indexing)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    kernel = np.ones((5, 5), np.uint8)
    frame_folder_seconds = getattr(cap, "available_seconds", None)
    if frame_folder_seconds is not None:
        if sequential_indexing:
            # Pixel mode without telemetry: there is no second or interval to
            # respect, so every image in the folder is analyzed, whatever it is
            # called.
            valid_seconds = set(frame_folder_seconds)
            print(
                f"  > Frame folder: analyzing all {len(valid_seconds)} images "
                "(pixel mode, no telemetry — filenames don't need to look "
                "like seconds)."
            )
        else:
            # One sample per second (the one that exists), not `scan_fps` of
            # them, and only for seconds that really have a file. The rest are
            # simply ignored rather than reported as having no sample.
            valid_seconds &= frame_folder_seconds
            print(
                f"  > Frame folder: {len(frame_folder_seconds)} frames available, "
                f"{len(valid_seconds)} within the valid range."
            )

    # Why each individual FRAME (not second) was excluded, shown later in the
    # preview to explain why one sample never became a candidate. Altitude and
    # tilt exclusions, decided per second before any frame is read, are filed
    # under the first sample that second would have produced, just so they can
    # be drawn.
    #
    # round(), not int(): int() truncates (sec=1 at fps=23.976 gives 23, not
    # 24), and dividing that frame number back by fps in the graph lands below
    # the first valid second (0.9593s against a 1.0s lower bound). The tick then
    # fell outside the visible range and vanished, leaving a second excluded for
    # altitude or tilt with no visible explanation. round() is also what
    # planned_frames below uses, so the two second<->frame conversions agree.
    #
    # Each value is a dict {whale_id or None: reason}, not one reason per frame.
    # With several animals expected, two whales in the same frame can end
    # differently (one accepted, one rejected), and the second one's reason used
    # to disappear because the frame counted as successful anyway. The None key
    # holds reasons that apply to the whole frame before we know which animals
    # are in it (altitude, tilt, nothing detected at all). Those are always
    # mutually exclusive with the whale_id keys, because the frame is either
    # never read (altitude, tilt) or no animal was ever detected in it
    # (yolo_none / yolo_low_conf when len(yolo_res.boxes) == 0, see below).
    rejected_frames = {
        round(sec * fps): {None: reason}
        for sec, reason in rejected_seconds_prefilter.items()
    }
    # Seconds whose outcome (candidate or rejection) is already settled,
    # tracked here rather than derived later from frame_idx via
    # int(frame_idx / fps). That round trip does not always invert with
    # non-integer fps (59.94 NTSC, say), so deriving it sometimes filed the
    # outcome under the wrong second, leaving the right one with no reason at
    # all and falling back on the generic "no sample" catch-all below even
    # though a precise reason had been found.
    resolved_seconds = set(rejected_seconds_prefilter.keys())

    area_m2, frame_pos, h_list, gsd_list, auto_len_list = [], [], [], [], []
    whale_id_list = []
    tilt_list = []
    shape_score_list = []
    auto_head_list = []
    auto_notch_list = []
    auto_mask_crop_list = []
    auto_mask_crop_origin_list = []
    auto_width_lists = {pct: [] for pct in range(5, 100, 5)}
    bbox_list = []
    sam_bbox_list = []
    missing_yolo_masks = False
    last_checkpoint_count = 0
    last_checkpoint_time = time.monotonic()
    # One mean shape per whale_id rather than a single shared one. Different
    # whales hold slightly different poses, and a mean mixed from two animals
    # ends up resembling neither, rejecting good candidates from both. So all
    # of this state is keyed by whale_id.
    mean_shape_sum = {}
    mean_shape_count = {}
    rejected_by_shape = 0
    # Candidates seen while the mean shape is warming up. They are accepted
    # only provisionally, since a few are needed to build the mean at all, and
    # rechecked against the finished mean as soon as it is ready. Otherwise an
    # odd shape among the first samples would pass unchecked forever, and would
    # pollute the reference mean too.
    pending_warmup_candidates = {}
    warmup_finalized = {}

    def visible_candidate_count():
        """Candidates to show in the live counter.

        Confirmed ones plus those still waiting on the mean-shape warmup. The
        waiting ones will be rechecked (see flush_pending_warmup_candidates),
        but they are candidates found all the same. Leaving them out kept the
        counter at 0 for the whole warmup, making it look like the analysis was
        finding nothing.
        """
        return len(auto_len_list) + sum(
            len(pending) for pending in pending_warmup_candidates.values()
        )

    def commit_candidate(record, shape_score):
        """Add a candidate to the final lists.

        Used both for normal candidates and for warmup ones rechecked later
        (see flush_pending_warmup_candidates).

        `pca_info` may be None: the animal was found (centred mask, valid
        shape) but the PCA could not locate head and fluke, because of posture
        or partial occlusion. The frame still becomes a candidate, just without
        an automatic length, and is measured by hand during review rather than
        disappearing unseen.
        """
        pca_info = record["pca_info"]
        mask_cleaned = record["mask_cleaned"]
        gsd_t = record["gsd_t"]
        x1, y1, x2, y2 = record["x1"], record["y1"], record["x2"], record["y2"]
        auto_len_list.append(pca_info[0] if pca_info is not None else np.nan)
        whale_id_list.append(record["whale_id"])
        area_m2.append(np.sum(mask_cleaned > 0) * (gsd_t ** 2))
        frame_pos.append(record["frame_idx"])
        resolved_seconds.add(record["current_sec"])
        h_list.append(record["h_t"])
        gsd_list.append(gsd_t)
        tilt_list.append(get_tilt(telemetry, record["current_sec"]))
        shape_score_list.append(shape_score)
        bbox_list.append([x1, y1, x2, y2])
        sam_bbox_list.append(record["sam_bbox"][0].tolist())
        auto_head_list.append(pca_info[1] if pca_info is not None else (np.nan, np.nan))
        auto_notch_list.append(pca_info[2] if pca_info is not None else (np.nan, np.nan))
        # The mask is cropped around the subject (with a margin) rather than
        # kept full-frame: light enough to hold in memory for hundreds of
        # candidates, and used in the preview to show which points were taken.
        mask_crop_for_preview, mask_crop_origin = expanded_bbox_crop(
            mask_cleaned,
            np.array([[x1, y1, x2, y2]], dtype=np.float32),
            _PREVIEW_ZOOM_MARGIN,
        )
        auto_mask_crop_list.append(mask_crop_for_preview)
        auto_mask_crop_origin_list.append(
            mask_crop_origin if mask_crop_origin is not None else (0, 0, 0, 0)
        )
        automatic_widths = (
            {width["pct"]: width["width_m"] for width in pca_info[3]}
            if pca_info is not None
            else {}
        )
        for pct in range(5, 100, 5):
            auto_width_lists[pct].append(automatic_widths.get(pct, np.nan))

    def flush_pending_warmup_candidates(whale_id):
        """Recheck one animal's buffered warmup candidates against its mean.

        The mean is the final one, or whatever was available at the end of the
        scan if the warmup never finished. Candidates that pass are committed;
        those that fail are rejected with a reason, and their contribution is
        subtracted from the mean so it does not stay polluted for the rest of
        the video.
        """
        nonlocal rejected_by_shape, last_checkpoint_count, last_checkpoint_time
        pending_list = pending_warmup_candidates.get(whale_id, [])
        shape_sum = mean_shape_sum.get(whale_id)
        if not pending_list or shape_sum is None:
            pending_warmup_candidates[whale_id] = []
            return
        final_reference_shape = shape_sum / mean_shape_count[whale_id] >= 0.5
        for pending in pending_list:
            pending_shape = pending["normalized_shape"].astype(bool)
            union = np.logical_or(final_reference_shape, pending_shape).sum()
            intersection = np.logical_and(final_reference_shape, pending_shape).sum()
            pending_shape_score = intersection / union if union else 0.0
            if pending_shape_score >= config.mean_mask_min_iou:
                commit_candidate(pending, pending_shape_score)
                print(
                    f"  > second {pending['current_sec']:.2f} "
                    f"(frame {pending['frame_idx']}): accepted "
                    "(re-checked after mean-shape warmup)"
                )
            else:
                rejected_by_shape += 1
                rejected_frames.setdefault(pending["frame_idx"], {})[whale_id] = "unusual_shape"
                resolved_seconds.add(pending["current_sec"])
                print(
                    f"  > second {pending['current_sec']:.2f} "
                    f"(frame {pending['frame_idx']}): excluded — "
                    f"{CandidatePreviewWindow._reason_label('unusual_shape')} "
                    "(re-checked after mean-shape warmup)"
                )
                mean_shape_sum[whale_id] -= pending["normalized_shape"]
                mean_shape_count[whale_id] -= 1
        pending_warmup_candidates[whale_id] = []
        if len(auto_len_list) != last_checkpoint_count:
            save_screening_checkpoint()
            last_checkpoint_count = len(auto_len_list)
            last_checkpoint_time = time.monotonic()

    def save_screening_checkpoint():
        """Save the raw candidates so progress can be watched and recovered if
        the scan is stopped."""
        checkpoint_data = {
            "frame_pos": frame_pos,
            "whale_id": whale_id_list,
            "mean_shape_iou": shape_score_list,
            "bbox_x1": [bbox[0] for bbox in bbox_list],
            "bbox_y1": [bbox[1] for bbox in bbox_list],
            "bbox_x2": [bbox[2] for bbox in bbox_list],
            "bbox_y2": [bbox[3] for bbox in bbox_list],
        }
        if measure_in_meters:
            checkpoint_data.update(
                {
                    "area_m2": area_m2,
                    "h": h_list,
                    "gsd": gsd_list,
                    "auto_length_m": auto_len_list,
                }
            )
        else:
            checkpoint_data.update(
                {
                    "area_px2": area_m2,
                    "auto_length_px": auto_len_list,
                }
            )
        checkpoint = pd.DataFrame(checkpoint_data)
        for pct in range(5, 100, 5):
            unit_suffix = "m" if measure_in_meters else "px"
            checkpoint[f"auto_width_{pct}pct_{unit_suffix}"] = (
                auto_width_lists[pct]
            )
        if measure_in_meters and telemetry.tilt_available:
            checkpoint["tilt_deg"] = tilt_list
        checkpoint.to_csv(screening_checkpoint_path, index=False)

    if fps <= 0:
        raise RuntimeError("Could not determine the video FPS.")
    # Sample second by second rather than with a fixed frame step. A fixed step
    # (round(fps / scan_fps)) almost never divides fps exactly, so the rounding
    # error drifts and on a long enough video whole seconds get skipped: a gap
    # in the preview timeline with no recorded reason, because no frame was ever
    # read for that second. Generating the samples one second at a time keeps
    # every valid second covered however long the video is.
    #
    # For a folder of frames there is already one image per second, so
    # `scan_fps` does not apply: there is nothing to choose between.
    samples_per_second = (
        1 if frame_folder_seconds is not None else max(1, round(config.scan_fps))
    )
    if sequential_indexing and frame_folder_seconds:
        # The indices are positions (0, 1, 2...), not real seconds, so the
        # configured start_sec/end_sec mean nothing here: the valid range is
        # simply every image.
        first_frame_bound = min(frame_folder_seconds)
        last_frame_bound = max(frame_folder_seconds) + 1
    else:
        first_frame_bound = int(start_sec * fps)
        last_frame_bound = int(end_sec * fps)
    # (frame_number, second): the second that produced the sample travels with
    # it instead of being recomputed below as int(frame_idx / fps). The round
    # trip out (second -> frame_number via round()) and back (frame_number ->
    # second via int()) does not always invert: with non-integer fps (59.94
    # NTSC) the first sample of a second can map back to the previous second.
    # That frame's outcome was then filed under the wrong second, leaving the
    # right one with no reason and falling back on the generic "no sample"
    # catch-all below even when a precise reason had been found.
    planned_frames = []
    for second in sorted(valid_seconds):
        for sample_index in range(samples_per_second):
            frame_number = int(round((second + sample_index / samples_per_second) * fps))
            if first_frame_bound <= frame_number < last_frame_bound:
                planned_frames.append((frame_number, second))
    planned_frames.sort()
    progress = AnalysisProgress(len(planned_frames))

    def read_frame_at(video_capture, target_frame):
        """Read a frame, skipping a costly seek when the target is close."""
        current = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        gap = target_frame - current
        if 0 <= gap <= int(fps * 2):
            for _ in range(gap):
                if not video_capture.grab():
                    return False, None
        else:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return video_capture.read()

    screening_interrupted = False
    for scan_number, (frame_idx, current_sec) in enumerate(planned_frames):
        console_interval = max(1, len(planned_frames) // 20)
        if scan_number % console_interval == 0:
            percentage = (scan_number / len(planned_frames) * 100) if planned_frames else 100
            print(
                f"  > Screening {percentage:.0f}% — "
                f"frame {scan_number}/{len(planned_frames)}, "
                f"candidates {visible_candidate_count()}"
            )
        if not progress.update(
            scan_number,
            len(planned_frames),
            frame_idx,
            current_sec,
            visible_candidate_count(),
            force=True,
        ):
            screening_interrupted = True
            break

        success, frame = read_frame_at(cap, frame_idx)
        if not success:
            if frame_folder_seconds is not None:
                # In a folder of frames an unreadable file (corrupt, or a
                # system leftover with an image extension) concerns only THAT
                # file. Unlike a video, it does not mean the stream ended, so it
                # must not stop the scan of the rest.
                continue
            break

        img_h, img_w = frame.shape[:2]

        # retina_masks only when YOLO's mask is the real measurement ("YOLO
        # segmentation only" mode). In "YOLO + SAM" the measurement always comes
        # from SAM at full resolution below, and YOLO's mask is just a reference
        # for the mean-shape filter, which shrinks it to 256x128 in
        # normalize_mask_shape anyway. Computing it at full resolution would be
        # wasted work.
        yolo_res = device.predict_adaptive(
            yolo_model,
            frame,
            config.analysis_imgsz,
            "YOLO screening",
            conf=config.confidence,
            max_det=5,
            retina_masks=(config.pipeline_mode != "YOLO + SAM"),
            # ByteTrack (shipped with ultralytics; persist=True is handled in
            # predict_adaptive) gives each animal a steady ID from one sampled
            # frame to the next. That is what keeps several whales in one video
            # apart instead of swapping them or always measuring the same one.
            #
            # With a single declared animal it is not needed, and it hurts: if
            # tracking is lost between two samples (an under-sampled video) the
            # ID changes anyway, splitting the per-whale_id mean shape below for
            # one animal that never actually left. So the tracker is skipped
            # entirely and a fixed ID is used.
            tracker=(None if config.expected_whale_count == 1 else "bytetrack.yaml"),
        )
        # One reason per REJECTED ANIMAL in this frame, not one for the frame
        # as a whole. With several animals expected, one accepted and one
        # rejected in the same frame are independent outcomes, and the second
        # must not vanish just because the first succeeded.
        # frame_has_accepted_detection stays frame-level only to decide the
        # summary line printed below.
        frame_reasons_by_whale = {}
        frame_has_accepted_detection = False
        if len(yolo_res.boxes) >= 1:
            track_ids = yolo_res.boxes.id
            if config.expected_whale_count == 1:
                # The user said there is one animal, so only the best box (by
                # area, as before multi-whale tracking) is used and the rest are
                # ignored. That removes any chance of duplicate masks for the
                # same animal at the source: there is no second box to process.
                areas = (
                    (yolo_res.boxes.xywh[:, 2] * yolo_res.boxes.xywh[:, 3])
                    .cpu()
                    .numpy()
                )
                det_indices = [int(np.argmax(areas))]
            else:
                # Several animals expected. YOLO's NMS works on boxes, and for a
                # long body two boxes covering different parts of the same whale
                # can overlap little while still being one animal. Duplicates
                # are dropped here by comparing the real masks instead of the
                # boxes (see deduplicate_detections_by_mask_iou).
                det_indices = deduplicate_detections_by_mask_iou(yolo_res, frame.shape)
            for det_idx in det_indices:
                rejection_reason = None
                if config.expected_whale_count == 1:
                    # Fixed, not from the tracker, which was not even called
                    # above: one mean-shape accumulator for the whole video,
                    # exactly as before multi-whale tracking.
                    whale_id = 1
                else:
                    whale_id = (
                        int(track_ids[det_idx].item())
                        if track_ids is not None
                        # Rare fallback when the tracker gave no ID for this
                        # frame: use the box's position rather than an ID unique
                        # per frame, so a single animal per frame (by far the
                        # common case) always lands in the same whale_id=0 group
                        # even if the tracker fails completely.
                        else det_idx
                    )

                bbox_xyxy = yolo_res.boxes.xyxy[det_idx].cpu().numpy()
                x1, y1, x2, y2 = bbox_xyxy

                bbox = yolo_res.boxes.xyxy[det_idx].unsqueeze(0).cpu().numpy()
                # A slightly larger box, used only as SAM's prompt: YOLO's box
                # sometimes cuts thin parts short, such as the tip of the head.
                # The tight box is still the one recorded for the candidate.
                # Computed right away, not only when SAM actually runs, so the
                # preview can always show it.
                sam_bbox = expand_bbox(bbox, frame.shape, config.sam_box_margin / 100.0)
                yolo_mask = extract_detection_mask(yolo_res, det_idx, frame.shape)

                if config.pipeline_mode == "YOLO + SAM":
                    # The measurement always comes from SAM, as it does in
                    # review.py. YOLO's mask is not measured here, only used as
                    # the mean-shape reference below.
                    #
                    # One prompt (bboxes=sam_bbox) per animal: SAM segments only
                    # that box, so the mask stays one whale's even when others
                    # share the frame.
                    mask_raw = None
                    active_sam_model = device.get_sam_model()
                    sam_res = device.predict_adaptive(
                        active_sam_model,
                        frame,
                        config.analysis_imgsz,
                        "SAM screening",
                        bboxes=sam_bbox,
                    )
                    if sam_res.masks is not None:
                        mask_raw = (
                            (sam_res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                        )
                        if mask_raw.shape != frame.shape[:2]:
                            mask_raw = cv2.resize(
                                mask_raw,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                    del sam_res
                    # clear_cache=False, as for YOLO just below. On MPS,
                    # torch.mps.empty_cache() synchronises with the GPU, so
                    # calling it every frame instead of every 20 (see below)
                    # halved throughput in "YOLO + SAM" mode on an M1.
                    device.release_prediction(active_sam_model, clear_cache=False)
                else:
                    mask_raw = yolo_mask
                    if mask_raw is None:
                        missing_yolo_masks = True

                if mask_raw is None:
                    # There is a box, so something really was detected; only the
                    # mask failed to come out. Kept separate from "nothing
                    # detected" (yolo_none) below, or the user would read "YOLO
                    # found nothing" when YOLO clearly found something.
                    rejection_reason = "yolo_mask"

                if mask_raw is not None:
                    # The shape's centre of mass describes where the animal
                    # really is better than the centre of its bounding box.
                    moments = cv2.moments(mask_raw, binaryImage=True)
                    if moments["m00"] > 0:
                        center_x = moments["m10"] / moments["m00"]
                        center_y = moments["m01"] / moments["m00"]
                    else:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    margin_x = img_w * (1.0 - config.central_fraction) / 2.0
                    margin_y = img_h * (1.0 - config.central_fraction) / 2.0
                    is_central = (
                        margin_x <= center_x <= img_w - margin_x
                        and margin_y <= center_y <= img_h - margin_y
                    )
                    if not is_central:
                        mask_raw = None
                        rejection_reason = "off_center"

                if mask_raw is not None:
                    mask_cleaned = cv2.morphologyEx(
                        mask_raw,
                        cv2.MORPH_OPEN, kernel
                    )
                    shape_score = 1.0
                    shape_is_valid = True

                    # Reference for the mean-shape filter: YOLO's mask whenever
                    # there is one, since it is steady and does not depend on the
                    # enlarged box given to SAM. mask_raw is the fallback when
                    # YOLO produced none for this frame.
                    shape_source = yolo_mask if yolo_mask is not None else mask_raw
                    shape_source_cleaned = cv2.morphologyEx(shape_source, cv2.MORPH_OPEN, kernel)
                    normalized_shape = normalize_mask_shape(shape_source_cleaned)

                    was_in_warmup = False
                    if config.use_mean_mask_filter and normalized_shape is not None:
                        # The mean and its warmup belong to THIS whale_id; see
                        # the comment where mean_shape_sum is set up for why
                        # they are not shared between animals.
                        whale_shape_count = mean_shape_count.get(whale_id, 0)
                        was_in_warmup = (
                            not warmup_finalized.get(whale_id, False)
                            and whale_shape_count < config.mean_mask_warmup
                        )
                        if not was_in_warmup:
                            reference_shape = (
                                mean_shape_sum[whale_id] / whale_shape_count >= 0.5
                            )
                            candidate_shape = normalized_shape.astype(bool)
                            union = np.logical_or(reference_shape, candidate_shape).sum()
                            intersection = np.logical_and(
                                reference_shape, candidate_shape
                            ).sum()
                            shape_score = intersection / union if union else 0.0
                            shape_is_valid = shape_score >= config.mean_mask_min_iou
                            if not shape_is_valid:
                                rejected_by_shape += 1
                                rejection_reason = "unusual_shape"
                            else:
                                mean_shape_sum[whale_id] = mean_shape_sum[whale_id] + normalized_shape
                                mean_shape_count[whale_id] = whale_shape_count + 1
                        else:
                            # Still warming up: accepted only provisionally and
                            # rechecked once the final mean is ready (see
                            # flush_pending_warmup_candidates). Otherwise an odd
                            # shape among the first samples would never be
                            # checked at all.
                            if whale_id not in mean_shape_sum:
                                mean_shape_sum[whale_id] = normalized_shape.astype(np.float32)
                            else:
                                mean_shape_sum[whale_id] = mean_shape_sum[whale_id] + normalized_shape
                            mean_shape_count[whale_id] = whale_shape_count + 1

                    if shape_is_valid:
                        frame_has_accepted_detection = True
                        # GSD, and so every physical measurement, comes from the
                        # nearest CSV altitude sample, not an interpolation (see
                        # get_nearest_altitude).
                        h_t = get_nearest_altitude(telemetry, current_sec)
                        gsd_t = (
                            (float(h_t) * float(pixel_size)) / float(focal_length)
                            if measure_in_meters
                            else 1.0
                        )

                        # pca_info may stay None when head and fluke were not
                        # found; the frame still becomes a candidate, see
                        # commit_candidate.
                        pca_info = measure_length_pca(mask_cleaned, gsd_t)

                        record = {
                            "frame_idx": frame_idx,
                            "current_sec": current_sec,
                            "whale_id": whale_id,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "h_t": h_t,
                            "gsd_t": gsd_t,
                            "sam_bbox": sam_bbox,
                            "pca_info": pca_info,
                            "mask_cleaned": mask_cleaned,
                            "normalized_shape": normalized_shape,
                        }
                        if was_in_warmup:
                            pending_warmup_candidates.setdefault(whale_id, []).append(record)
                        else:
                            commit_candidate(record, shape_score)
                            # ponytail: this used to rewrite the whole checkpoint
                            # every 10 candidates, which is O(n^2) over a scan,
                            # since each rewrite copies every row saved so far.
                            # On a long video with hundreds of candidates the
                            # pauses got steadily worse. Rewrites are now paced
                            # by elapsed time, not by candidate count.
                            if (
                                len(auto_len_list) != last_checkpoint_count
                                and time.monotonic() - last_checkpoint_time >= 15.0
                            ):
                                save_screening_checkpoint()
                                last_checkpoint_count = len(auto_len_list)
                                last_checkpoint_time = time.monotonic()

                    if (
                        not warmup_finalized.get(whale_id, False)
                        and config.use_mean_mask_filter
                        and mean_shape_count.get(whale_id, 0) >= config.mean_mask_warmup
                    ):
                        warmup_finalized[whale_id] = True
                        flush_pending_warmup_candidates(whale_id)

                if rejection_reason is not None:
                    frame_reasons_by_whale[whale_id] = rejection_reason
                    # Mark the second settled here, not only in
                    # commit_candidate. Otherwise an animal rejected in a frame
                    # with no other outcome would fall through to the
                    # "no_sample" catch-all below, hiding the real reason
                    # just found.
                    resolved_seconds.add(current_sec)

            # With several animals expected, a second animal below the main
            # confidence threshold never produces a box in yolo_res: the loop
            # above cannot see it, so it used to vanish without trace even when
            # another animal in the same frame was handled normally.
            #
            # The same frame is rechecked here at low confidence (as it already
            # is below when there is NO box at all), and any box that matches
            # none of those handled above is flagged. Matching is by position,
            # not identity: a below-threshold detection never has a reliable
            # track_id, so it cannot be tied to a whale_id and stays a general
            # note about the frame rather than about one animal.
            if config.expected_whale_count > 1:
                low_conf_floor = min(0.05, config.confidence * 0.5)
                extra_low_conf_res = device.predict_adaptive(
                    yolo_model,
                    frame,
                    config.analysis_imgsz,
                    "YOLO screening (extra low-confidence check)",
                    conf=low_conf_floor,
                    max_det=5,
                    retina_masks=False,
                    tracker=None,
                )
                if len(extra_low_conf_res.boxes) >= 1:
                    known_boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
                    extra_boxes_xyxy = extra_low_conf_res.boxes.xyxy.cpu().numpy()
                    extra_confs = extra_low_conf_res.boxes.conf.cpu().numpy()
                    best_unmatched_conf = None
                    for extra_box, extra_conf in zip(extra_boxes_xyxy, extra_confs):
                        if _box_matches_any(extra_box, known_boxes_xyxy):
                            continue
                        if best_unmatched_conf is None or extra_conf > best_unmatched_conf:
                            best_unmatched_conf = float(extra_conf)
                    if best_unmatched_conf is not None:
                        frame_reasons_by_whale[None] = (
                            f"yolo_extra_low_conf:{best_unmatched_conf:.0%} seen, "
                            f"{config.confidence:.0%} required"
                        )
                del extra_low_conf_res
        else:
            # No box at the configured confidence. Before recording "nothing
            # detected", look once at low confidence, only for THIS frame: doing
            # it on every frame would slow the whole scan for a check that only
            # matters when the main confidence finds nothing. It separates
            # "really nothing there" from "something below the threshold", which
            # mean very different things to a user who has to trust the
            # explanation shown in the preview.
            low_conf_floor = min(0.05, config.confidence * 0.5)
            low_conf_res = device.predict_adaptive(
                yolo_model,
                frame,
                config.analysis_imgsz,
                "YOLO screening (low confidence check)",
                conf=low_conf_floor,
                max_det=5,
                retina_masks=False,
                tracker=None,
            )
            if len(low_conf_res.boxes) >= 1:
                best_conf = float(low_conf_res.boxes.conf.max().item())
                frame_reasons_by_whale[None] = (
                    f"yolo_low_conf:{best_conf:.0%} seen, "
                    f"{config.confidence:.0%} required"
                )
            else:
                frame_reasons_by_whale[None] = "yolo_none"
            del low_conf_res
            resolved_seconds.add(current_sec)

        if frame_reasons_by_whale:
            # Keyed on the exact frame, not the second, so a rejected sample
            # (an odd shape, say) stays visible even if another sample of that
            # second later became a candidate. The value is a dict per frame
            # (whale_id or None -> reason) rather than one reason: see the
            # comment on frame_reasons_by_whale above. An animal accepted in
            # this frame no longer blocks recording why another was rejected.
            rejected_frames.setdefault(frame_idx, {}).update(frame_reasons_by_whale)

        # One line per sample, not just the final per-reason summary. With
        # scan_fps > 1 several samples of the same second can end differently,
        # and this is the only way to watch them all while the scan runs instead
        # of waiting for the graph at the end.
        #
        # "accepted (candidate)" here can still change: during the mean-shape
        # warmup a candidate passes provisionally (see above) and is rechecked
        # by flush_pending_warmup_candidates, which prints its final outcome
        # separately if it differs.
        if frame_has_accepted_detection:
            print(f"  > second {current_sec:.2f} (frame {frame_idx}): accepted (candidate)")
        for reason_whale_id, reason in frame_reasons_by_whale.items():
            whale_label = f"whale {reason_whale_id}" if reason_whale_id is not None else "frame"
            print(
                f"  > second {current_sec:.2f} (frame {frame_idx}, {whale_label}): excluded — "
                f"{CandidatePreviewWindow._reason_label(reason)}"
            )

        progress.update(
            scan_number + 1,
            len(planned_frames),
            frame_idx,
            current_sec,
            visible_candidate_count(),
            force=True,
        )
        del yolo_res
        del frame
        device.release_prediction(yolo_model, clear_cache=False)
        if (scan_number + 1) % 20 == 0:
            device.clear_accelerator_memory()

    cap.release()
    # If the mean-shape warmup never finished (short video, or a very strict
    # filter), the buffered candidates would otherwise be lost entirely. They
    # are rechecked here against whatever partial mean exists, one whale_id at a
    # time, each with its own mean.
    for pending_whale_id in list(pending_warmup_candidates):
        if pending_warmup_candidates[pending_whale_id]:
            flush_pending_warmup_candidates(pending_whale_id)
    # resolved_seconds is updated as each outcome happens (see
    # commit_candidate, flush_pending_warmup_candidates and the rejection branch
    # above) instead of being derived from frame_idx. That round trip through
    # fps does not always invert (see the comment above planned_frames), so
    # deriving it here could call a second unaccounted for when it already had a
    # reason, filed under a frame_idx that fell in the neighbouring second.
    #
    # Safety net: a valid second that really ended with no frame accepted or
    # rejected (rare, for instance when the mean-shape warmup never finished)
    # still gets a generic reason, so no gap in the timeline is left without a
    # tick explaining it.
    for second in valid_seconds:
        if second not in resolved_seconds:
            rejected_frames.setdefault(int(second * fps), {})[None] = "no_sample"
    save_screening_checkpoint()
    # One debug image per whale_id. With a single animal the file keeps its
    # usual name; with more, each gets its own so the means can be compared by
    # eye.
    multi_whale_debug = len(mean_shape_sum) > 1
    for shape_whale_id, whale_shape_sum in mean_shape_sum.items():
        whale_shape_count = mean_shape_count.get(shape_whale_id, 0)
        if whale_shape_sum is None or not whale_shape_count:
            continue
        mean_shape_image = (
            whale_shape_sum / whale_shape_count * 255
        ).astype(np.uint8)
        debug_name = (
            f"mean_shape_masks_W{shape_whale_id}.png"
            if multi_whale_debug
            else "mean_shape_masks.png"
        )
        cv2.imwrite(str(output_path / debug_name), mean_shape_image)
    progress.close()
    if screening_interrupted:
        print("Screening interrupted: using the candidates collected so far.")
    print(f"Scan completed. Found {len(auto_len_list)} valid frames computed.")
    if config.use_mean_mask_filter:
        print(
            f"  > Mean shape filter: {rejected_by_shape} anomalous masks excluded."
        )
    if rejected_frames:
        # Always print the per-reason summary. The tick strip on the graph
        # alone does not show whether "no_sample" (the catch-all above) is
        # firing rarely, as intended, or often. The exact count is right here,
        # with no white ticks to count by eye.
        reason_breakdown = collections.Counter(
            reason.split(":", 1)[0]
            for reasons_by_whale in rejected_frames.values()
            for reason in reasons_by_whale.values()
        )
        print(
            "  > Excluded seconds by reason: "
            + ", ".join(
                f"{count} {reason}"
                for reason, count in sorted(reason_breakdown.items(), key=lambda kv: -kv[1])
            )
        )
        unexplained = reason_breakdown.get("no_sample", 0)
        if unexplained:
            print(
                f"  > {unexplained} second(s) excluded with no specific reason "
                "(rare edge case, e.g. the mean-shape warmup never finished for "
                "that animal) — look for the white ticks in the preview graph."
            )

    if not auto_len_list:
        if config.pipeline_mode == "YOLO segmentation only" and missing_yolo_masks:
            raise RuntimeError(
                "The selected YOLO model does not produce masks. "
                "Choose a segmentation YOLO model or use YOLO + SAM."
            )
        print("No valid frames.")
        offer_new_analysis()
        sys.exit(0)

    # Build the DataFrame that phase 2 works on
    valid_data = pd.DataFrame({
        "area_m2": area_m2,
        "frame_pos": frame_pos,
        "whale_id": whale_id_list,
        "h": h_list,
        "gsd": gsd_list,
        "auto_length": auto_len_list,
        "mean_shape_iou": shape_score_list,
        "bbox_x1": [bbox[0] for bbox in bbox_list],
        "bbox_y1": [bbox[1] for bbox in bbox_list],
        "bbox_x2": [bbox[2] for bbox in bbox_list],
        "bbox_y2": [bbox[3] for bbox in bbox_list],
        "sam_bbox_x1": [bbox[0] for bbox in sam_bbox_list],
        "sam_bbox_y1": [bbox[1] for bbox in sam_bbox_list],
        "sam_bbox_x2": [bbox[2] for bbox in sam_bbox_list],
        "sam_bbox_y2": [bbox[3] for bbox in sam_bbox_list],
    })
    for pct in range(5, 100, 5):
        valid_data[f"_auto_width_{pct}pct"] = auto_width_lists[pct]
    valid_data["_auto_head_x"] = [point[0] for point in auto_head_list]
    valid_data["_auto_head_y"] = [point[1] for point in auto_head_list]
    valid_data["_auto_notch_x"] = [point[0] for point in auto_notch_list]
    valid_data["_auto_notch_y"] = [point[1] for point in auto_notch_list]
    valid_data["_auto_mask_crop"] = pd.Series(auto_mask_crop_list, dtype=object)
    valid_data["_auto_mask_crop_x1"] = [origin[0] for origin in auto_mask_crop_origin_list]
    valid_data["_auto_mask_crop_y1"] = [origin[1] for origin in auto_mask_crop_origin_list]
    if telemetry.tilt_available:
        valid_data["tilt_deg"] = tilt_list

    # ─── FASE 2: Selezione statistica ─────────────────────────────────────
    if measure_in_meters:
        print(f"PHASE 2: Quartile selection {target_quartiles} and shuffle...")
        # IQR outliers and candidates outside the requested quartiles used to
        # be dropped from the final pool, so the user lost them without ever
        # seeing them. They all stay selectable now, just marked
        # OUTLIER_QUARTILE so the graph draws them in a different colour (see
        # CandidatePreviewWindow). Whether to review them is the reviewer's
        # call, not the statistical filter's.
        #
        # Computed one whale_id at a time, not over the whole pool: a large
        # adult and a small calf have completely different length
        # distributions, and quartiles over the mixed pool could mark the calf
        # an outlier merely for being smaller than the adult, rather than odd
        # for ITS own size.
        candidate_df = valid_data.copy()
        quartile_column = pd.Series(0, index=candidate_df.index, dtype="int64")
        is_iqr_outlier = pd.Series(False, index=candidate_df.index)
        outside_requested_quartile = pd.Series(False, index=candidate_df.index)
        for whale_id, group in candidate_df.groupby("whale_id"):
            (
                group_quartile,
                group_is_iqr_outlier,
                group_outside_requested,
                q0_stat,
                q4_stat,
            ) = classify_length_quartiles(group["auto_length"], target_quartiles)
            quartile_column.loc[group.index] = group_quartile
            is_iqr_outlier.loc[group.index] = group_is_iqr_outlier
            outside_requested_quartile.loc[group.index] = group_outside_requested
            print(f"  > Whale {whale_id}: length range {q0_stat:.2f}-{q4_stat:.2f} m")
        candidate_df["_screening_quartile"] = quartile_column
        outlier_mask = is_iqr_outlier | outside_requested_quartile

        for _, row in candidate_df.loc[is_iqr_outlier].iterrows():
            print(
                f"  > second {row['frame_pos'] / fps:.2f} (frame {int(row['frame_pos'])}, "
                f"whale {row['whale_id']}): "
                f"kept as outlier — length outside 1.5x IQR of this "
                f"individual's length distribution"
            )
        for _, row in candidate_df.loc[outside_requested_quartile & ~is_iqr_outlier].iterrows():
            print(
                f"  > second {row['frame_pos'] / fps:.2f} (frame {int(row['frame_pos'])}, "
                f"whale {row['whale_id']}): "
                f"kept as outlier — outside the requested size quartiles "
                f"{target_quartiles} for this individual (auto length {row['auto_length']:.2f} m)"
            )
        print(
            f"  Candidates in the requested quartiles: "
            f"{int((~outlier_mask).sum())} frames, "
            f"{int(outlier_mask.sum())} kept as outliers"
        )
    else:
        print(
            "PHASE 2: pixel mode — all valid candidates are available "
            "in the visual selection, without quartiles."
        )
        candidate_df = valid_data.copy()

    final_df = candidate_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df["video_second"] = (final_df["frame_pos"] / fps).astype(int)
    # A stable id for each candidate, so it survives later sorting and
    # filtering (the preview, going back to the timeline for more).
    final_df["_candidate_id"] = final_df.index

    if measure_in_meters:
        print(f"  Total candidates available for review (outliers included): {len(final_df)} frames")
    else:
        print(f"  Valid candidates in pixels: {len(final_df)} frames")
    print(f"  Proceeding with random extraction up to the limit of {config.max_accepted}...")

    # Most of screening's buffers are no longer needed, so the models are
    # unloaded here. Review, and any later timeline round, reloads them on
    # demand.
    device.unload_inference_models()
    device.clear_accelerator_memory()

    return final_df, fps, rejected_frames
