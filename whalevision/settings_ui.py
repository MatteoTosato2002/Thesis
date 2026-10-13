"""The settings window shown at startup, plus restarting the app.

Nothing here imports cv2, numpy, pandas, torch or ultralytics. The settings
window has to appear straight away, without waiting for those heavy libraries
to load (see `pipeline.main`).
"""

import csv
import os
import sys
from pathlib import Path

from .config import AppConfig, SCRIPT_DIR, save_last_configuration

_shared_tk_root = None


def get_shared_tk_root():
    """The one Tk root for the whole process.

    Every window must be opened as `tk.Toplevel(get_shared_tk_root())`, never
    with its own `tk.Tk()`. Each `tk.Tk()` starts a separate Tcl interpreter
    whose events cannot reach the others, which is what used to freeze the
    interface right after a measurement and leave black leftover windows.
    """
    global _shared_tk_root
    import tkinter as tk

    if _shared_tk_root is None:
        _shared_tk_root = tk.Tk()
        _shared_tk_root.withdraw()
    return _shared_tk_root


def bind_vertical_scroll(root, canvas) -> None:
    """Let the mouse wheel and two-finger trackpad scrolling move a Canvas."""
    def scroll_units(units):
        if units:
            canvas.yview_scroll(units, "units")
        return "break"

    def on_mousewheel(event):
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return None
        if abs(delta) >= 120:
            units = -int(delta / 120)
        else:
            # macOS sends a stream of small deltas for two-finger scrolling.
            units = -1 if delta > 0 else 1
        return scroll_units(units)

    root.bind("<MouseWheel>", on_mousewheel, add="+")
    root.bind("<Button-4>", lambda _event: scroll_units(-1), add="+")
    root.bind("<Button-5>", lambda _event: scroll_units(1), add="+")


def ask_confirm(parent, title, message, yes_label, no_label, default_yes=False):
    """A yes/no dialog whose buttons say what they will do.

    messagebox.askyesno only offers a plain Yes/No, which never tells the user
    what either choice leads to.
    """
    import tkinter as tk
    from tkinter import ttk

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.transient(parent)

    result = {"value": False}

    def choose(value):
        result["value"] = value
        dialog.destroy()

    container = ttk.Frame(dialog, padding=20)
    container.pack(fill="both", expand=True)
    ttk.Label(container, text=message, wraplength=420, justify="left").pack(anchor="w")

    button_row = ttk.Frame(container)
    button_row.pack(fill="x", pady=(18, 0))
    yes_button = ttk.Button(button_row, text=yes_label, command=lambda: choose(True))
    no_button = ttk.Button(button_row, text=no_label, command=lambda: choose(False))
    no_button.pack(side="right")
    yes_button.pack(side="right", padx=(0, 8))

    dialog.protocol("WM_DELETE_WINDOW", lambda: choose(False))
    dialog.bind("<Escape>", lambda _event: choose(False))
    (yes_button if default_yes else no_button).focus_set()
    dialog.update_idletasks()
    # The parent is usually the hidden shared root (get_shared_tk_root). Unless
    # this dialog is explicitly raised, macOS can leave it behind the other
    # windows, or invisible until the user clicks the app icon.
    dialog.deiconify()
    dialog.lift()
    dialog.focus_force()
    dialog.grab_set()
    dialog.wait_window()
    return result["value"]


def restart_application():
    """Restart the program, with the settings window prefilled as it was."""
    cv2_module = sys.modules.get("cv2")
    if cv2_module is not None:
        cv2_module.destroyAllWindows()
    os.execv(
        sys.executable,
        [sys.executable, str(SCRIPT_DIR / "whale_vision.py")],
    )


def offer_new_analysis() -> None:
    # ask_confirm makes its own Toplevel, so no extra window is needed here.
    run_again = ask_confirm(
        get_shared_tk_root(),
        "Whale Vision",
        "Review finished.\n\nDo you want to go back to settings and start "
        "a new analysis?",
        yes_label="Yes, new analysis",
        no_label="No, quit the program",
        default_yes=True,
    )
    if run_again:
        restart_application()


def select_configuration(initial_config: "AppConfig | None" = None) -> "AppConfig | None":
    """Show the settings window, before any heavy library is imported."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Toplevel(get_shared_tk_root())
    root.title("Whale Vision — Configuration")
    root.geometry("820x700")
    root.minsize(760, 620)

    cfg = initial_config or AppConfig()
    values: dict[str, tk.Variable] = {}
    input_widgets = {}
    input_labels = {}
    browse_buttons = {}
    result: list[AppConfig] = []

    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    body = ttk.Frame(canvas, padding=18)
    body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    body_window = canvas.create_window((0, 0), window=body, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(body_window, width=event.width),
    )
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    bind_vertical_scroll(root, canvas)

    ttk.Label(body, text="Whale Vision", font=("", 22, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 3)
    )
    ttk.Label(
        body,
        text="Select the data and adjust the parameters before analysis.",
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 18))
    body.columnconfigure(1, weight=1)

    row = 2

    def section(title: str) -> None:
        nonlocal row
        ttk.Separator(body).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        row += 1
        ttk.Label(body, text=title, font=("", 13, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        row += 1

    def field(
        key: str,
        label: str,
        default,
        browse: "str | None" = None,
        width: int = 18,
    ) -> None:
        nonlocal row
        var_type = tk.BooleanVar if isinstance(default, bool) else tk.StringVar
        var = var_type(value=default)
        values[key] = var
        label_widget = ttk.Label(body, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        input_labels[key] = label_widget
        if isinstance(default, bool):
            input_widget = ttk.Checkbutton(body, variable=var)
            input_widget.grid(row=row, column=1, sticky="w", pady=4)
        else:
            input_widget = ttk.Entry(body, textvariable=var, width=width)
            input_widget.grid(
                row=row, column=1, sticky="ew", pady=4
            )
        input_widgets[key] = input_widget
        if browse:
            def choose(kind=browse, target=var):
                if kind == "video":
                    if values["input_mode"].get() == "Frames folder":
                        path = filedialog.askdirectory(
                            title="Choose the frames folder"
                        )
                    else:
                        path = filedialog.askopenfilename(
                            title="Choose the video",
                            filetypes=[
                                ("Video", "*.mp4 *.mov *.avi *.mkv"),
                                ("All files", "*.*"),
                            ],
                        )
                elif kind == "csv":
                    path = filedialog.askopenfilename(
                        title="Choose the telemetry", filetypes=[("CSV", "*.csv")]
                    )
                elif kind == "model":
                    path = filedialog.askopenfilename(
                        title="Choose the model", filetypes=[("PyTorch model", "*.pt")]
                    )
                elif kind == "dataset_folder":
                    path = filedialog.askdirectory(
                        title="Choose the YOLO-Seg dataset folder"
                    )
                else:
                    path = filedialog.askdirectory(title="Choose the output folder")
                if path:
                    target.set(path)
                    if kind == "video":
                        video = Path(path)
                        values["video_id"].set(video.stem)
                        if not values["output_folder"].get():
                            values["output_folder"].set(
                                str(video.parent / f"whale_results_{video.stem}")
                            )
            browse_button = ttk.Button(body, text="Browse…", command=choose)
            browse_button.grid(
                row=row, column=2, padx=(10, 0), pady=4
            )
            browse_buttons[key] = browse_button
        row += 1

    def choice_field(key: str, label: str, default: str, choices: "tuple[str, ...]") -> None:
        nonlocal row
        var = tk.StringVar(value=default)
        values[key] = var
        label_widget = ttk.Label(body, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        input_labels[key] = label_widget
        input_widget = ttk.Combobox(
            body,
            textvariable=var,
            values=choices,
            state="readonly",
        )
        input_widget.grid(row=row, column=1, sticky="ew", pady=4)
        input_widgets[key] = input_widget
        row += 1

    section("Measurement unit")
    choice_field(
        "measurement_unit",
        "Analyze and save measurements in",
        cfg.measurement_unit,
        ("Meters", "Pixels"),
    )
    measurement_mode_help = ttk.Label(
        body,
        wraplength=700,
        foreground="#555555",
    )
    measurement_mode_help.grid(
        row=row,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(0, 8),
    )
    row += 1

    section("Files")
    choice_field(
        "input_mode",
        "Input source",
        cfg.input_mode,
        ("Video", "Frames folder"),
    )
    field("video_path", "Video", cfg.video_path, "video")
    field("csv_path", "Telemetry CSV", cfg.csv_path, "csv")
    field(
        "video_time_column",
        "Video seconds column",
        cfg.video_time_column,
    )
    field(
        "altitude_column",
        "Altitude column",
        cfg.altitude_column,
    )
    field(
        "flight_date_column",
        "Flight date column (empty = automatic)",
        cfg.flight_date_column,
    )
    field("output_folder", "Results folder", cfg.output_folder, "folder")
    field("video_id", "Video identifier", cfg.video_id)
    field("yolo_model_path", "YOLO model", cfg.yolo_model_path, "model")
    field("sam_model_path", "SAM model (YOLO + SAM only)", cfg.sam_model_path, "model")

    section("Acquisition and scale")
    field("start_sec", "Analysis start (s)", cfg.start_sec)
    field("end_sec", "Analysis end (s)", cfg.end_sec)
    field("sensor_width_mm", "Sensor width (mm)", cfg.sensor_width_mm)
    field("image_width_px", "Image width (px)", cfg.image_width_px)
    field("focal_length_mm", "Focal length (mm)", cfg.focal_length_mm)

    section("Screening")
    quality_presets = ("320", "480", "640", "800", "960", "1280", "1536")
    choice_field(
        "pipeline_mode",
        "Segmentation mode",
        cfg.pipeline_mode,
        ("YOLO + SAM", "YOLO segmentation only"),
    )
    choice_field(
        "analysis_imgsz",
        "Screening quality (px)",
        str(cfg.analysis_imgsz),
        quality_presets,
    )
    choice_field(
        "review_imgsz",
        "Review quality (px)",
        str(cfg.review_imgsz),
        quality_presets,
    )
    field(
        "sam_box_margin",
        "SAM box margin (%)",
        cfg.sam_box_margin,
    )
    choice_field(
        "compute_device",
        "Accelerator",
        cfg.compute_device,
        ("Automatic", "CPU (lower memory risk)"),
    )
    field("confidence", "YOLO confidence (0–1)", cfg.confidence)
    field("scan_fps", "Frames analyzed per second (2 recommended)", cfg.scan_fps)
    field("max_alt_diff", "Max altitude change (m/s)", cfg.max_alt_diff)
    field(
        "use_tilt_filter",
        "Filter using tilt_deg (if present)",
        cfg.use_tilt_filter,
    )
    field(
        "max_tilt_deg",
        "Maximum allowed tilt (degrees)",
        cfg.max_tilt_deg,
    )
    field(
        "central_fraction",
        "Allowed central fraction (1 = whole frame)",
        cfg.central_fraction,
    )
    field(
        "use_mean_mask_filter",
        "Blow filter based on the mean shape",
        cfg.use_mean_mask_filter,
    )
    field(
        "mean_mask_min_iou",
        "Minimum similarity to the mean shape",
        cfg.mean_mask_min_iou,
    )
    field(
        "mean_mask_warmup",
        "Initial masks used to build the mean",
        cfg.mean_mask_warmup,
    )

    section("Selection and review")
    field("target_quartiles", "Quartiles (e.g. 3,4)", cfg.target_quartiles)
    field(
        "use_candidate_preview",
        "Show the visual preselection before measuring",
        cfg.use_candidate_preview,
    )
    field(
        "expected_whale_count",
        "Sperm whales in the video (1 forces a single animal, ignores the tracker)",
        cfg.expected_whale_count,
    )
    field("max_accepted", "Maximum accepted count", cfg.max_accepted)
    field("max_frames_per_sec", "Maximum accepted per second", cfg.max_frames_per_sec)
    field("skip_review", "Automatic analysis only", cfg.skip_review)
    ttk.Label(
        body,
        text=(
            "Skips the visual preselection and the interactive review: "
            "exports the automatic-measurement CSV directly."
        ),
        wraplength=700,
        foreground="#555555",
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
    row += 1

    section("YOLO-Seg dataset (optional)")
    ttk.Label(
        body,
        text=(
            "If set, frames marked with the \"add to database\" checkbox "
            "during review (P key) are saved here instead of inside the "
            "results folder, so a single dataset accumulates across "
            "multiple videos and sessions."
        ),
        wraplength=700,
        foreground="#555555",
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
    row += 1
    field(
        "yolo_dataset_folder",
        "Database folder (empty = inside the results)",
        cfg.yolo_dataset_folder,
        "dataset_folder",
    )

    section("Extra segments (optional)")
    ttk.Label(
        body,
        text=(
            "Names of specific measurements to draw on every accepted frame "
            "(e.g. pectoral_fin, girth), comma-separated. If present, "
            "review will ask you to trace each segment in order; leave "
            "empty to skip this."
        ),
        wraplength=700,
        foreground="#555555",
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
    row += 1
    field(
        "custom_segment_names",
        "Segments (e.g. pectoral_fin, girth)",
        cfg.custom_segment_names,
    )

    def update_measurement_mode_controls(*_args) -> None:
        pixel_mode = values["measurement_unit"].get() == "Pixels"
        disabled_keys = (
            "sensor_width_mm",
            "image_width_px",
            "focal_length_mm",
            "max_alt_diff",
            "use_tilt_filter",
            "max_tilt_deg",
            "target_quartiles",
        )
        for key in disabled_keys:
            widget = input_widgets[key]
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="disabled" if pixel_mode else "readonly")
            else:
                widget.configure(state="disabled" if pixel_mode else "normal")
        if pixel_mode:
            measurement_mode_help.configure(
                text=(
                    "Pixel mode: telemetry is optional, quartiles and flight "
                    "filters are disabled. CSVs will only contain measurements in px."
                )
            )
            input_labels["csv_path"].configure(text="Telemetry CSV (optional)")
            input_labels["target_quartiles"].configure(
                text="Quartiles (not available in pixel mode)"
            )
        else:
            if not values["target_quartiles"].get().strip():
                values["target_quartiles"].set("3,4")
            measurement_mode_help.configure(
                text=(
                    "Meters mode: the telemetry CSV is required and must "
                    "contain the configured time and altitude columns."
                )
            )
            input_labels["csv_path"].configure(text="Telemetry CSV (required)")
            input_labels["target_quartiles"].configure(
                text="Quartiles (e.g. 3,4)"
            )

    values["measurement_unit"].trace_add(
        "write",
        update_measurement_mode_controls,
    )
    update_measurement_mode_controls()

    def update_input_mode_controls(*_args) -> None:
        frames_folder_mode = values["input_mode"].get() == "Frames folder"
        input_labels["video_path"].configure(
            text="Frames folder" if frames_folder_mode else "Video"
        )

    values["input_mode"].trace_add("write", update_input_mode_controls)
    update_input_mode_controls()

    def close_dialog():
        # This window is a Toplevel of the shared Tk root, not a tk.Tk() of
        # its own, so destroy() alone does not end mainloop(): the hidden
        # shared root stays alive. quit() has to stop the loop first.
        root.quit()
        root.destroy()

    def submit() -> None:
        try:
            pixel_mode = values["measurement_unit"].get() == "Pixels"

            def numeric_or_default(key, converter, default):
                raw_value = values[key].get()
                if pixel_mode and str(raw_value).strip() == "":
                    return default
                return converter(raw_value)

            parsed = AppConfig(
                measurement_unit=values["measurement_unit"].get(),
                input_mode=values["input_mode"].get(),
                video_path=values["video_path"].get().strip(),
                csv_path=values["csv_path"].get().strip(),
                video_time_column=values["video_time_column"].get().strip(),
                altitude_column=values["altitude_column"].get().strip(),
                flight_date_column=values["flight_date_column"].get().strip(),
                output_folder=values["output_folder"].get().strip(),
                video_id=values["video_id"].get().strip(),
                yolo_model_path=values["yolo_model_path"].get().strip(),
                sam_model_path=values["sam_model_path"].get().strip(),
                start_sec=float(values["start_sec"].get()),
                end_sec=float(values["end_sec"].get()),
                sensor_width_mm=numeric_or_default(
                    "sensor_width_mm",
                    float,
                    AppConfig.sensor_width_mm,
                ),
                image_width_px=numeric_or_default(
                    "image_width_px",
                    int,
                    AppConfig.image_width_px,
                ),
                focal_length_mm=numeric_or_default(
                    "focal_length_mm",
                    float,
                    AppConfig.focal_length_mm,
                ),
                confidence=float(values["confidence"].get()),
                analysis_imgsz=int(values["analysis_imgsz"].get()),
                review_imgsz=int(values["review_imgsz"].get()),
                sam_box_margin=int(values["sam_box_margin"].get()),
                compute_device=values["compute_device"].get(),
                pipeline_mode=values["pipeline_mode"].get(),
                scan_fps=float(values["scan_fps"].get()),
                max_alt_diff=numeric_or_default(
                    "max_alt_diff",
                    float,
                    AppConfig.max_alt_diff,
                ),
                use_tilt_filter=bool(values["use_tilt_filter"].get()),
                max_tilt_deg=numeric_or_default(
                    "max_tilt_deg",
                    float,
                    AppConfig.max_tilt_deg,
                ),
                central_fraction=float(values["central_fraction"].get()),
                use_mean_mask_filter=bool(values["use_mean_mask_filter"].get()),
                mean_mask_min_iou=float(values["mean_mask_min_iou"].get()),
                mean_mask_warmup=int(values["mean_mask_warmup"].get()),
                use_candidate_preview=bool(
                    values["use_candidate_preview"].get()
                ),
                expected_whale_count=int(values["expected_whale_count"].get()),
                max_accepted=int(values["max_accepted"].get()),
                max_frames_per_sec=int(values["max_frames_per_sec"].get()),
                skip_review=bool(values["skip_review"].get()),
                target_quartiles=values["target_quartiles"].get().strip(),
                yolo_dataset_folder=values["yolo_dataset_folder"].get().strip(),
                custom_segment_names=values["custom_segment_names"].get().strip(),
            )
            if parsed.measurement_unit not in {"Meters", "Pixels"}:
                raise ValueError("Choose whether to measure in meters or in pixels.")
            if parsed.input_mode not in {"Video", "Frames folder"}:
                raise ValueError("Choose whether the input is a video or a frames folder.")
            if parsed.expected_whale_count < 1:
                raise ValueError("The number of sperm whales in the video must be at least 1.")
            if parsed.measurement_unit == "Meters" and (
                not parsed.video_time_column
                or not parsed.altitude_column
            ):
                raise ValueError(
                    "Enter the name of the video seconds column and of the "
                    "altitude column."
                )
            if (
                (parsed.measurement_unit == "Meters" or parsed.csv_path)
                and parsed.video_time_column
                and parsed.altitude_column
                and parsed.video_time_column == parsed.altitude_column
            ):
                raise ValueError(
                    "The video seconds column and the altitude column "
                    "must be different."
                )
            video_field_label = (
                "Frames folder" if parsed.input_mode == "Frames folder" else "Video"
            )
            required = {
                video_field_label: parsed.video_path,
                "Results folder": parsed.output_folder,
                "Video identifier": parsed.video_id,
                "YOLO model": parsed.yolo_model_path,
            }
            if parsed.measurement_unit == "Meters":
                required["Telemetry CSV"] = parsed.csv_path
            if parsed.pipeline_mode == "YOLO + SAM":
                required["SAM model"] = parsed.sam_model_path
            missing = [name for name, value in required.items() if not value]
            missing_files = [
                name for name, value in required.items()
                if name != "Results folder" and value and name != "Video identifier"
                and not (
                    Path(value).expanduser().is_dir()
                    if name == video_field_label and parsed.input_mode == "Frames folder"
                    else Path(value).expanduser().is_file()
                )
            ]
            if (
                parsed.csv_path
                and not Path(parsed.csv_path).expanduser().is_file()
                and "Telemetry CSV" not in missing_files
            ):
                missing_files.append("Telemetry CSV")
            if missing:
                raise ValueError("Missing fields: " + ", ".join(missing))
            if missing_files:
                raise ValueError("Files not found: " + ", ".join(missing_files))
            if parsed.measurement_unit == "Meters":
                try:
                    with Path(parsed.csv_path).expanduser().open(
                        newline="",
                        encoding="utf-8-sig",
                    ) as telemetry_file:
                        telemetry_header = next(csv.reader(telemetry_file), [])
                except OSError as exc:
                    raise ValueError(
                        f"Could not read the telemetry: {exc}"
                    ) from exc
                missing_telemetry_columns = {
                    parsed.video_time_column,
                    parsed.altitude_column,
                }.difference(telemetry_header)
                if missing_telemetry_columns:
                    raise ValueError(
                        "The CSV does not contain the configured columns: "
                        + ", ".join(sorted(missing_telemetry_columns))
                    )
            if parsed.end_sec <= parsed.start_sec:
                raise ValueError("The end must be after the start.")
            if not 0 < parsed.confidence <= 1 or not 0 < parsed.central_fraction <= 1:
                raise ValueError("Confidence and central fraction must be between 0 and 1.")
            if not 0 < parsed.mean_mask_min_iou <= 1:
                raise ValueError("Shape similarity must be between 0 and 1.")
            if parsed.mean_mask_warmup < 3:
                raise ValueError("At least 3 masks are needed to build the mean shape.")
            if parsed.scan_fps <= 0 or parsed.focal_length_mm <= 0 or parsed.image_width_px <= 0:
                raise ValueError("FPS, focal length and image width must be positive.")
            if parsed.analysis_imgsz < 256 or parsed.review_imgsz < 256:
                raise ValueError("Model quality settings must be at least 256 px.")
            if not 0 <= parsed.sam_box_margin <= 100:
                raise ValueError("The SAM box margin must be between 0 and 100%.")
            if parsed.measurement_unit == "Meters":
                quartiles = {
                    int(q.strip())
                    for q in parsed.target_quartiles.split(",")
                }
                if not quartiles or not quartiles <= {1, 2, 3, 4}:
                    raise ValueError("Allowed quartiles are 1, 2, 3 and 4.")
            else:
                parsed.target_quartiles = ""
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        parsed.video_path = str(Path(parsed.video_path).expanduser().resolve())
        if parsed.csv_path:
            parsed.csv_path = str(
                Path(parsed.csv_path).expanduser().resolve()
            )
        parsed.output_folder = str(Path(parsed.output_folder).expanduser().resolve())
        parsed.yolo_model_path = str(Path(parsed.yolo_model_path).expanduser().resolve())
        if parsed.sam_model_path:
            parsed.sam_model_path = str(
                Path(parsed.sam_model_path).expanduser().resolve()
            )
        if parsed.yolo_dataset_folder:
            parsed.yolo_dataset_folder = str(
                Path(parsed.yolo_dataset_folder).expanduser().resolve()
            )
        save_last_configuration(parsed)
        result.append(parsed)
        close_dialog()

    buttons = ttk.Frame(body)
    buttons.grid(row=row, column=0, columnspan=3, sticky="e", pady=(20, 10))
    ttk.Button(buttons, text="Cancel", command=close_dialog).pack(side="left", padx=6)
    ttk.Button(buttons, text="Start analysis", command=submit).pack(side="left")
    root.protocol("WM_DELETE_WINDOW", close_dialog)
    root.mainloop()
    return result[0] if result else None
