"""Anatomical landmarks: the list of points, the CSV, and the picker panel.

`LandmarkControlPanel` shares its interaction state (`active_landmark_code`,
`pending_landmark_point`, `landmark_mode`, `has_mouse_position`) with
`mouse_callback` in `review.py`. Both read and write the same `ReviewState`
object, handed over once per frame by `bind_state()`.
"""

import csv

LANDMARK_DEFINITIONS = (
    ("BH", "Blowhole"),
    ("E_R", "Eye right"),
    ("E_L", "Eye left"),
    ("SC_E", "Spermaceti end"),
    ("PF_R", "Pectoral fin right"),
    ("PF_L", "Pectoral fin left"),
    ("DF_S", "Dorsal fin start"),
    ("DF_E", "Dorsal fin end"),
    ("PD", "Peduncle"),
    ("F_R", "Fluke right"),
    ("F_L", "Fluke left"),
)
# Head and fluke are normally fixed: they come from the length measurement the
# user already confirmed, so they can be trusted. They become editable like any
# other landmark when LandmarkControlPanel.start is called with
# editable_head_fluke=True. That happens for an animal with no hand-measured
# length, whose head and fluke come only from the automatic PCA done during
# screening (see other_whale_landmark_mode in review.py).
_HEAD_FLUKE_DEFINITIONS = (("S", "Head/snout"), ("F", "Fluke"))
LANDMARK_LABELS = dict(_HEAD_FLUKE_DEFINITIONS + LANDMARK_DEFINITIONS)
LANDMARK_EXPORT_ORDER = (
    "S",
    "F",
    "BH",
    "DF_S",
    "DF_E",
    "PD",
    "PF_R",
    "PF_L",
    "F_R",
    "F_L",
    "E_L",
    "E_R",
    "SC_E",
)


def load_existing_landmark_records(landmark_csv_path):
    records = {}
    if not landmark_csv_path.is_file():
        return records
    try:
        with landmark_csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        if len(rows) < 2:
            return records

        old_bodyparts = rows[0]
        old_columns = {}
        for column_index in range(3, len(old_bodyparts), 2):
            bodypart = old_bodyparts[column_index].strip()
            if bodypart:
                old_columns[bodypart] = (column_index, column_index + 1)

        for old_row in rows[2:]:
            if len(old_row) < 3 or not old_row[2]:
                continue
            normalized_row = old_row[:3]
            for bodypart in LANDMARK_EXPORT_ORDER:
                coordinate_columns = old_columns.get(bodypart)
                if coordinate_columns is None:
                    normalized_row.extend(["", ""])
                    continue
                x_column, y_column = coordinate_columns
                x_value = old_row[x_column] if x_column < len(old_row) else ""
                y_value = old_row[y_column] if y_column < len(old_row) else ""
                normalized_row.extend([x_value, y_value])
            records[normalized_row[2]] = normalized_row
    except (OSError, csv.Error):
        pass
    return records


def write_landmark_csv(records_by_frame, landmark_csv_path):
    bodyparts_row = ["bodyparts", "", ""]
    coords_row = ["coords", "", ""]
    for bodypart in LANDMARK_EXPORT_ORDER:
        bodyparts_row.extend([bodypart, bodypart])
        coords_row.extend(["x", "y"])

    with landmark_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(bodyparts_row)
        writer.writerow(coords_row)
        writer.writerows(records_by_frame.values())


def make_landmark_record(frame_filename, landmark_points, video_dataset_label):
    row = ["labeled-data", video_dataset_label, frame_filename]
    for bodypart in LANDMARK_EXPORT_ORDER:
        point = landmark_points.get(bodypart)
        if point is None:
            row.extend(["", ""])
        else:
            # Float, not int(): a placed or dragged landmark is accurate to
            # below a pixel (see review.py), and this CSV feeds the training of
            # a pose estimation network (DeepLabCut format). Rounding here would
            # throw that accuracy away in the training file itself, for nothing.
            row.extend([float(point[0]), float(point[1])])
    return row


class LandmarkControlPanel:
    """The landmark section of the single review panel."""

    def __init__(self, master=None, review_action_panel=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self._panel = review_action_panel
        self._state = None
        self.root = ttk.LabelFrame(
            master,
            text="Anatomical points",
            padding=10,
        )
        self.points = {}
        self.visited_codes = set()
        self.finish_requested = False
        self.buttons = {}

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text=(
                "Press a body part, click the point on the frame,\n"
                "then press A. Without clicking, A leaves it empty."
            ),
        ).pack(anchor="w", pady=(0, 8))

        self.fixed_points_text = tk.StringVar(master=self.root, value="")
        self.fixed_points_label = ttk.Label(
            container,
            textvariable=self.fixed_points_text,
            foreground="#555555",
        )
        self.fixed_points_label.pack(anchor="w", pady=(0, 10))

        # Head (S) and fluke (F) are usually just a fixed label at the top
        # (see fixed_points_text). They turn into two editable buttons like the
        # rest when start() is called with editable_head_fluke=True; see the
        # comment on _HEAD_FLUKE_DEFINITIONS.
        self.head_fluke_frame = ttk.Frame(container)
        for button_index, (code, label) in enumerate(_HEAD_FLUKE_DEFINITIONS):
            button = ttk.Button(
                self.head_fluke_frame,
                text=f"{label} ({code})",
                command=lambda selected_code=code: self.select(selected_code),
            )
            button.grid(row=0, column=button_index, sticky="ew", padx=3, pady=2)
            self.head_fluke_frame.columnconfigure(button_index, weight=1)
            button.bind(
                "<ButtonPress-1>",
                lambda _event, selected_code=code: (
                    self.select(selected_code),
                    "break",
                )[1],
            )
            self.buttons[code] = (button, label)
        # Not packed here: start() only shows it when needed, see
        # editable_head_fluke.

        button_frame = ttk.Frame(container)
        self.button_frame = button_frame
        button_frame.pack(fill="x")
        for button_column in range(3):
            button_frame.columnconfigure(button_column, weight=1)
        for button_index, (code, label) in enumerate(LANDMARK_DEFINITIONS):
            button = ttk.Button(
                button_frame,
                text=f"{label} ({code})",
                command=lambda selected_code=code: self.select(selected_code),
            )
            button.grid(
                row=button_index // 3,
                column=button_index % 3,
                sticky="ew",
                padx=3,
                pady=2,
            )
            button.bind(
                "<ButtonPress-1>",
                lambda _event, selected_code=code: (
                    self.select(selected_code),
                    "break",
                )[1],
            )
            self.buttons[code] = (button, label)

        self.status = tk.StringVar(
            master=self.root,
            value="Select a body part.",
        )
        ttk.Label(
            container,
            textvariable=self.status,
            wraplength=400,
            font=("", 11, "bold"),
        ).pack(fill="x", pady=(14, 8))

        actions = ttk.Frame(container)
        actions.pack(fill="x")
        confirm_button = ttk.Button(
            actions,
            text="Confirm / leave empty (A)",
            command=self.confirm_and_advance,
        )
        confirm_button.pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        self.bind_immediate(confirm_button)
        redo_button = ttk.Button(actions, text="Redo", command=self.redo)
        redo_button.pack(
            side="left", expand=True, fill="x", padx=4
        )
        self.bind_immediate(redo_button)
        discard_button = ttk.Button(
            actions,
            text="Discard frame",
            command=self.discard_frame,
        )
        discard_button.pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        self.bind_immediate(discard_button)
        finish_button = ttk.Button(
            container,
            text="Confirm all and continue",
            command=self.finish,
        )
        finish_button.pack(fill="x", pady=(14, 0))
        self.bind_immediate(finish_button)
        self.root.pack_forget()

    @staticmethod
    def bind_immediate(button):
        """Fire the command on press, so a click is never lost on release."""
        def invoke_on_press(_event):
            button.invoke()
            return "break"

        button.bind("<ButtonPress-1>", invoke_on_press)

    def bind_state(self, state):
        """Attach the current frame's ReviewState (called once per frame)."""
        self._state = state

    def start(self, head_point, fluke_point, editable_head_fluke=False):
        """Open the landmark panel for one frame.

        `editable_head_fluke` is normally False. When True, head and fluke
        become two editable buttons at the top like every other landmark,
        instead of a read-only label. That is for an animal whose head and
        fluke come from the automatic PCA during screening and were never
        checked by hand, unlike the main animal's, which come from a length
        measurement the user confirmed.

        `head_point` and `fluke_point` may be None ONLY when
        `editable_head_fluke` is True, meaning no PCA result exists for that
        animal (for example an outline drawn from scratch with no clear
        shape). The matching button then starts as "still to place", like any
        other point, instead of being prefilled and marked as visited.
        """
        state = self._state
        self._editable_head_fluke = editable_head_fluke
        # Float, not int(): S and F start from the ends of the measurement just
        # confirmed, already accurate to below a pixel, and end up in the same
        # training CSV as the other landmarks (see make_landmark_record).
        self.points = {
            "S": (float(head_point[0]), float(head_point[1])) if head_point is not None else None,
            "F": (float(fluke_point[0]), float(fluke_point[1])) if fluke_point is not None else None,
        }
        self.points.update({code: None for code, _label in LANDMARK_DEFINITIONS})
        if editable_head_fluke:
            self.visited_codes = {
                code for code in ("S", "F") if self.points[code] is not None
            }
        else:
            self.visited_codes = set()
        self.finish_requested = False
        state.active_landmark_code = None
        state.pending_landmark_point = None
        if editable_head_fluke:
            self.fixed_points_label.pack_forget()
            self.head_fluke_frame.pack(fill="x", pady=(0, 10), before=self.button_frame)
        else:
            self.head_fluke_frame.pack_forget()
            self.fixed_points_label.pack(anchor="w", pady=(0, 10), before=self.button_frame)
            self.fixed_points_text.set(
                f"Head (S): {self.points['S']}    Fluke (F): {self.points['F']}"
            )
        for code, (button, label) in self.buttons.items():
            marker = "✓ " if code in self.visited_codes else ""
            button.configure(text=f"{marker}{label} ({code})")
        self.status.set("")
        self.root.pack(fill="both", expand=True)
        if self._panel is not None:
            self._panel.show_landmarks()
        # If head or fluke are still missing (editable_head_fluke, and the PCA
        # did not find them), start there rather than at the first anatomical
        # point, so the user places them first, as feels natural.
        initial_code = None
        if editable_head_fluke and self.points["S"] is None:
            initial_code = "S"
        elif editable_head_fluke and self.points["F"] is None:
            initial_code = "F"
        elif LANDMARK_DEFINITIONS:
            initial_code = LANDMARK_DEFINITIONS[0][0]
        if initial_code is not None:
            self.select(initial_code)

    def select(self, code):
        if self._panel is not None:
            self._panel.mark_dirty()
        state = self._state
        previous_code = state.active_landmark_code
        if previous_code is not None and previous_code != code:
            previous_button, previous_label = self.buttons[previous_code]
            if previous_code in self.visited_codes:
                previous_marker = (
                    "✓" if self.points.get(previous_code) is not None else "—"
                )
                previous_button.configure(
                    text=f"{previous_marker} {previous_label} ({previous_code})"
                )
            else:
                previous_button.configure(
                    text=f"{previous_label} ({previous_code})"
                )
        state.active_landmark_code = code
        state.pending_landmark_point = self.points.get(code)
        label = LANDMARK_LABELS[code]
        button, _button_label = self.buttons[code]
        button.configure(text=f"→ {label} ({code})")
        if state.pending_landmark_point is None:
            self.status.set(
                f"{label}: click the point on the frame, then press A. "
                "Press A without clicking to leave it empty."
            )
        else:
            self.status.set(
                f"{label}: current point {state.pending_landmark_point}. "
                "Click to change it or press A to confirm it."
            )

    def set_pending_point(self, point):
        state = self._state
        # Float, not int(): the click already arrives accurate to below a pixel
        # from mouse_callback, and this value goes into the training CSV once
        # confirmed (see make_landmark_record).
        state.pending_landmark_point = (float(point[0]), float(point[1]))
        if state.active_landmark_code:
            label = LANDMARK_LABELS[state.active_landmark_code]
            self.status.set(
                f"{label}: point {state.pending_landmark_point}. "
                "Press A to confirm or Redo."
            )

    def confirm_and_advance(self):
        if self._panel is not None:
            self._panel.mark_dirty()
        state = self._state
        if state.active_landmark_code is None:
            self.status.set(
                "Sequence completed. To redo a point, press its "
                "button; otherwise confirm all and continue."
            )
            return
        completed_code = state.active_landmark_code
        self.points[completed_code] = state.pending_landmark_point
        self.visited_codes.add(completed_code)
        button, label = self.buttons[completed_code]
        marker = "✓" if state.pending_landmark_point is not None else "—"
        button.configure(text=f"{marker} {label} ({completed_code})")
        state.active_landmark_code = None
        state.pending_landmark_point = None
        self.advance_after(completed_code)

    def advance_after(self, completed_code):
        # S and F go at the front only when they are real buttons (see start,
        # editable_head_fluke). Otherwise completed_code could be "S" or "F"
        # without appearing in ordered_codes, and the .index() call below would
        # raise ValueError instead of moving on to the next landmark.
        ordered_codes = (
            (["S", "F"] if getattr(self, "_editable_head_fluke", False) else [])
            + [code for code, _label in LANDMARK_DEFINITIONS]
        )
        if not ordered_codes:
            return
        completed_index = ordered_codes.index(completed_code)
        search_order = (
            ordered_codes[completed_index + 1 :]
            + ordered_codes[:completed_index]
        )
        next_code = next(
            (
                code
                for code in search_order
                if code not in self.visited_codes
            ),
            None,
        )
        if next_code is None:
            self.status.set(
                "All landmarks have been reviewed. Press a point's button "
                "to redo it, or Confirm all and continue."
            )
            return
        self.select(next_code)

    def redo(self):
        if self._panel is not None:
            self._panel.mark_dirty()
        state = self._state
        if state.active_landmark_code is None:
            self.status.set("Select the body part to redo.")
            return
        state.pending_landmark_point = None
        self.points[state.active_landmark_code] = None
        button, label = self.buttons[state.active_landmark_code]
        button.configure(text=f"→ {label} ({state.active_landmark_code})")
        self.status.set(f"{label}: click the point on the frame again.")

    def discard_frame(self):
        """Drop the whole current frame from the landmark step.

        Landmarks are the last review step, so there is no next step to skip
        to: the equivalent here is jumping straight to the next frame (see the
        X/K key in the main handler).
        """
        if self._panel is not None:
            self._panel.queue_key(ord("x"))

    def finish(self):
        state = self._state
        if state.active_landmark_code is not None and state.pending_landmark_point is not None:
            self.status.set(
                "The selected point is not confirmed yet: press "
                "Confirm point or Redo (Confirm without clicking "
                "leaves it empty)."
            )
            return
        self.finish_requested = True

    def hide(self):
        self.root.pack_forget()
        if self._panel is not None:
            self._panel.hide_landmarks()

    def return_to_settings(self):
        if self._panel is not None:
            self._panel.request_settings()

    def update(self):
        return
