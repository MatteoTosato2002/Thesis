"""The main entry point: settings, then screening, review and export.

Import order matters here. `select_configuration` must run before anything
imports cv2, numpy, pandas, torch or ultralytics, otherwise the settings
window stops appearing instantly (see the "Delayed imports" comment below).
That is why `screening`, `review`, `export` and `device` are imported inside
`main()` and never at the top of the file.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import load_last_configuration
from .settings_ui import offer_new_analysis, restart_application, select_configuration


def collapse_fragmented_track_ids(candidate_pool, expected_whale_count=None):
    """Merge tracking fragments of one animal back into a single whale_id.

    The tracker (ByteTrack) only sees the sampled frames (scan_fps), not every
    frame of the video. If the whale moves far enough between two samples, the
    tracker can fail to match them and hands out a new ID for what is still the
    same animal. With a single whale in the video that leaves several "ghost"
    whale_ids for one animal that never actually left the frame.

    If `expected_whale_count == 1` (the user said there is one animal), all
    fragments are always merged. That is safer than any guess: YOLO sometimes
    returns two nearly identical boxes for the same animal in the same frame
    (NMS noise, not a second whale), and that alone is enough for ByteTrack to
    open two IDs — so the "evidence" of several animals used below can be a
    false alarm.

    Otherwise, the evidence for REAL multiple animals is two different
    whale_ids present in the same frame. If that never happens anywhere in the
    scan, they are all fragments of one animal.
    """
    if not len(candidate_pool):
        return candidate_pool
    if expected_whale_count != 1:
        max_concurrent_per_frame = candidate_pool.groupby("frame_pos")["whale_id"].nunique().max()
        if max_concurrent_per_frame > 1:
            return candidate_pool
    candidate_pool = candidate_pool.copy()
    candidate_pool["whale_id"] = candidate_pool["whale_id"].iloc[0]
    return candidate_pool


def rejected_frames_for_whale(rejected_frames, whale_id):
    """Keep only the rejection reasons that concern one animal.

    `rejected_frames` maps frame -> {whale_id or None: reason} (see
    run_screening). For each frame this picks that whale's own reason if there
    is one, otherwise the reason filed under None, which applies to the whole
    frame (altitude, tilt, or nothing detected at all).

    Without this filter, reviewing whale 1 also showed the frames rejected for
    whale 2, with no way to tell them apart.
    """
    flattened = {}
    for frame_idx, reasons_by_whale in rejected_frames.items():
        if whale_id in reasons_by_whale:
            flattened[frame_idx] = reasons_by_whale[whale_id]
        elif None in reasons_by_whale:
            flattened[frame_idx] = reasons_by_whale[None]
    return flattened


@dataclass(frozen=True)
class SessionPaths:
    output_path: Path
    frame_output_folder: Path
    mask_database_folder: Path
    landmark_csv_path: Path
    screening_checkpoint_path: Path
    video_dataset_label: str
    flight_date_label: str


def main():
    config = select_configuration(load_last_configuration())
    if config is None:
        print("Operation cancelled.")
        return

    # Delayed imports: the settings window can appear without waiting for PyTorch.
    import json
    import os
    from dataclasses import asdict

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from . import device, export, review, screening
    from .measurement import sanitize_name
    from .telemetry import extract_flight_date, load_telemetry

    measure_in_meters = config.measurement_unit == "Meters"

    output_path = Path(config.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "configuration.json").open("w", encoding="utf-8") as config_file:
        json.dump(asdict(config), config_file, indent=2, ensure_ascii=False)

    accelerator = device.init(
        config.compute_device,
        config.yolo_model_path,
        config.sam_model_path,
    )
    print(f"Selected accelerator: {accelerator}")

    telemetry = load_telemetry(
        config.csv_path,
        config.video_time_column,
        config.altitude_column,
        config.use_tilt_filter,
        measure_in_meters,
    )

    flight_date_label = extract_flight_date(
        telemetry.dataframe,
        config.flight_date_column,
    )
    video_dataset_label = sanitize_name(
        f"{Path(config.video_path).stem}_{flight_date_label}"
    )
    frame_output_folder = output_path / "frame"
    frame_output_folder.mkdir(parents=True, exist_ok=True)
    paths = SessionPaths(
        output_path=output_path,
        frame_output_folder=frame_output_folder,
        mask_database_folder=(
            Path(config.yolo_dataset_folder).expanduser()
            if config.yolo_dataset_folder
            else output_path / "mask_database"
        ),
        landmark_csv_path=output_path / f"landmarks_{video_dataset_label}.csv",
        screening_checkpoint_path=(
            output_path / f"screening_partial_{video_dataset_label}.csv"
        ),
        video_dataset_label=video_dataset_label,
        flight_date_label=flight_date_label,
    )

    import pandas as pd

    candidate_pool, video_fps, rejected_frames = screening.run_screening(
        config,
        telemetry,
        config.video_path,
        paths.output_path,
        paths.screening_checkpoint_path,
    )

    candidate_pool = collapse_fragmented_track_ids(
        candidate_pool, config.expected_whale_count
    )

    if config.skip_review:
        # No human review wanted: export the candidates straight out of
        # screening, skipping the per-animal preview and review loop below.
        export.write_automatic_only_csv(candidate_pool, paths.output_path, config)
        offer_new_analysis()
        return

    # One ID per whale, assigned by the tracker in screening.run_screening.
    # With several animals in the video the whole preview + review loop runs
    # once per animal, each with its own quota (max_accepted) and its own
    # rounds back to the timeline, so measurements never get mixed up.
    whale_ids = list(pd.unique(candidate_pool["whale_id"])) if len(candidate_pool) else []
    if config.expected_whale_count > 1 and len(whale_ids) != config.expected_whale_count:
        print(
            f"Warning: you indicated {config.expected_whale_count} sperm whales, "
            f"but tracking distinguished {len(whale_ids)}. Likely tracking "
            "fragmentation: check the per-animal results in the preview "
            "before proceeding."
        )
    multi_whale = len(whale_ids) > 1
    if multi_whale:
        print(f"Detected {len(whale_ids)} distinct sperm whales in the video.")

    restart_requested = False
    preview_window = screening.CandidatePreviewWindow()

    for whale_index, whale_id in enumerate(whale_ids or [None], start=1):
        whale_pool = (
            candidate_pool[candidate_pool["whale_id"] == whale_id]
            if whale_id is not None
            else candidate_pool
        )
        whale_label = f"Sperm whale {whale_index}/{len(whale_ids)}" if multi_whale else None
        if multi_whale:
            print(f"--- {whale_label}: {len(whale_pool)} candidates ---")

        # Each round reviews a subset of THIS animal's candidates.
        # decided_status marks the ones already reviewed (accepted or
        # discarded), so going back to the timeline only offers new ones.
        whale_reviewed_rounds = []
        decided_status = {}
        accepted_per_second = {}
        round_number = 1

        while True:
            if config.use_candidate_preview or round_number > 1:
                to_review = preview_window.run(
                    whale_pool,
                    config.video_path,
                    video_fps,
                    telemetry,
                    config.start_sec,
                    config.end_sec,
                    measure_in_meters,
                    config,
                    decided_status=decided_status,
                    rejected_frames=rejected_frames_for_whale(rejected_frames, whale_id),
                    subject_label=whale_label,
                    current_whale_id=whale_id,
                )
                if to_review is None:
                    print("Preview stopped: returning to settings.")
                    restart_application()
            else:
                to_review = whale_pool
            print(f"  Frames chosen for detailed review: {len(to_review)}.")

            already_accepted = sum(
                len(round_df.dropna(subset=["measured_length"]))
                for round_df in whale_reviewed_rounds
            )
            reviewed_df, restart_requested, round_decided, accepted_per_second = review.run_review(
                to_review,
                config,
                config.video_path,
                paths,
                previously_accepted=already_accepted,
                previously_accepted_per_second=accepted_per_second,
                subject_label=whale_label,
                include_whale_suffix=multi_whale,
                full_candidate_pool=candidate_pool if multi_whale else None,
            )
            whale_reviewed_rounds.append(reviewed_df)
            decided_status.update(round_decided)

            if restart_requested:
                break

            total_accepted = sum(
                len(round_df.dropna(subset=["measured_length"]))
                for round_df in whale_reviewed_rounds
            )
            if total_accepted >= config.max_accepted:
                break

            remaining_candidates = whale_pool[
                ~whale_pool["_candidate_id"].isin(set(decided_status))
            ]
            if remaining_candidates.empty:
                print(
                    "Quota not reached "
                    f"({total_accepted}/{config.max_accepted}), but there are "
                    "no other candidates to propose."
                )
                break

            if not _prompt_return_to_timeline(total_accepted, config.max_accepted):
                break
            round_number += 1

        # One CSV per animal rather than one combined file. With several
        # whales the file name carries the ID (risultati_finali_..._W{id}),
        # and each is written as soon as that animal's review ends. So if the
        # user restarts before reaching the next animal, the work already done
        # is not lost.
        whale_video_id = f"{config.video_id}_W{whale_id}" if multi_whale else config.video_id
        export.write_final_csvs(
            pd.concat(whale_reviewed_rounds, ignore_index=True),
            paths.output_path,
            whale_video_id,
            measure_in_meters,
        )

        if restart_requested:
            break

    if restart_requested:
        restart_application()
    else:
        offer_new_analysis()


def _prompt_return_to_timeline(accepted_count, max_accepted):
    """Ask whether to go back to the timeline for more candidates, or stop."""
    from .settings_ui import ask_confirm, get_shared_tk_root

    # ask_confirm makes its own Toplevel. The extra hidden wrapper window that
    # used to be here served no purpose and on macOS could stay invisible,
    # making it look like the dialog never opened.
    return ask_confirm(
        get_shared_tk_root(),
        "Quota not reached",
        (
            f"You accepted {accepted_count} of {max_accepted} requested frames.\n\n"
            "Do you want to go back to the timeline to look for more candidates "
            "to analyze, or close the analysis with the frames collected so far?"
        ),
        yes_label="Back to frame selection",
        no_label="Close and save the collected frames",
        default_yes=True,
    )
