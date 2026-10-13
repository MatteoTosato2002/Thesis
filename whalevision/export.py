"""Frame annotati, dataset YOLO-Seg, confronto automatico/umano e CSV finali."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .frame_source import FrameFolderCapture, use_sequential_indexing
from .landmarks import LANDMARK_EXPORT_ORDER
from .measurement import draw_readable_text, mask_to_yolo_segmentation_lines


def build_annotated_result_frame(
    clean_frame,
    mask,
    spine,
    widths,
    landmark_points,
):
    """Draw the shapes and their names, without any numbers."""
    annotated = clean_frame.copy()
    short_side = min(annotated.shape[:2])
    main_thickness = max(2, int(round(short_side / 900)))
    thin_thickness = max(1, main_thickness - 1)

    if mask is not None:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            annotated,
            contours,
            -1,
            (0, 255, 255),
            main_thickness,
        )

    spine_points = [tuple(map(int, point)) for point in spine]
    for point_a, point_b in zip(spine_points, spine_points[1:]):
        cv2.line(
            annotated,
            point_a,
            point_b,
            (0, 165, 255),
            main_thickness,
            cv2.LINE_AA,
        )

    for width_info in widths:
        cv2.line(
            annotated,
            tuple(map(int, width_info["pt_top"])),
            tuple(map(int, width_info["pt_bot"])),
            (255, 0, 255),
            thin_thickness,
            cv2.LINE_AA,
        )

    for bodypart in LANDMARK_EXPORT_ORDER:
        point = landmark_points.get(bodypart)
        if point is None:
            continue
        point = tuple(map(int, point))
        cv2.circle(
            annotated,
            point,
            max(3, main_thickness + 1),
            (80, 255, 80),
            thickness=cv2.FILLED,
            lineType=cv2.LINE_AA,
        )
        draw_readable_text(
            annotated,
            bodypart,
            (point[0] + 8, point[1] - 8),
            0.65,
            (80, 255, 80),
            max(1, thin_thickness),
        )
    return annotated


def reconstruct_mask_from_crop(crop, x0, y0, frame_h, frame_w):
    """Paste a cropped mask back onto a full-frame canvas.

    The crop's top-left corner in the full frame is (x0, y0); see
    _PREVIEW_ZOOM_MARGIN in screening.py. Used both to combine several animals
    in other_whale_instance_lines and to offer another animal's outline for
    hand-correction in review.py.
    """
    canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if not isinstance(crop, np.ndarray) or crop.size == 0 or pd.isna(x0) or pd.isna(y0):
        return canvas
    x0i, y0i = int(x0), int(y0)
    crop_h, crop_w = crop.shape[:2]
    y1, x1 = min(y0i + crop_h, frame_h), min(x0i + crop_w, frame_w)
    if y1 <= y0i or x1 <= x0i:
        return canvas
    canvas[y0i:y1, x0i:x1] = crop[: y1 - y0i, : x1 - x0i]
    return canvas


def _polygon_to_yolo_line(polygon, frame_w, frame_h):
    """One YOLO-Seg label line (class 0) from a full-frame pixel polygon.

    Coordinates stay at full precision (float, not rounded): dragging a vertex
    can place it inside a pixel, see review.py.
    """
    normalized = []
    for point_x, point_y in polygon:
        normalized.extend(
            (
                np.clip(float(point_x) / frame_w, 0.0, 1.0),
                np.clip(float(point_y) / frame_h, 0.0, 1.0),
            )
        )
    return "0 " + " ".join(f"{coordinate:.6f}" for coordinate in normalized)


def other_whale_instance_lines(
    candidate_pool,
    frame_pos,
    exclude_whale_id,
    frame_shape,
    epsilon_fraction=0.0005,
    confirmed_polygons=None,
):
    """Extra YOLO-Seg label lines for the OTHER animals in the SAME frame.

    `exclude_whale_id` is the animal being reviewed, which already has its own
    line elsewhere.

    When several whales share a frame, the training image saved for one of them
    still shows the others. Without a label for those too, the model sees a
    perfectly good whale shape marked as background: a contradictory training
    signal that makes training worse rather than better.

    `confirmed_polygons` (optional) maps whale_id -> polygon for animals whose
    outline was corrected by hand during THIS review (see the "other whale"
    step in review.py). Those are used instead of the raw automatic mask, being
    more accurate. The map also covers animals added by hand during review,
    which have no row in `candidate_pool` at all because screening never found
    or accepted them (see start_manual_other_whale in review.py); for those
    there is no automatic mask to fall back on, only the outline drawn from
    scratch. Any other animal, with no correction available (for instance
    because the step was skipped), falls back to its automatic mask.

    Known and accepted limit: an animal that SCREENING rejected in this
    particular frame (off-centre, odd shape) and that was never added by hand
    has no automatic mask here to use or correct, so it stays unlabelled. Those
    masks were rejected for being low quality in the first place.
    """
    confirmed_polygons = confirmed_polygons or {}
    same_frame = None
    if (
        candidate_pool is not None
        and len(candidate_pool)
        and "whale_id" in candidate_pool.columns
        and "_auto_mask_crop" in candidate_pool.columns
    ):
        same_frame = candidate_pool[
            (candidate_pool["frame_pos"] == frame_pos)
            & (candidate_pool["whale_id"] != exclude_whale_id)
        ].drop_duplicates(subset="whale_id")
    rows_by_whale_id = (
        {row.get("whale_id"): row for _, row in same_frame.iterrows()}
        if same_frame is not None and not same_frame.empty
        else {}
    )
    # Take the union, not just same_frame: an animal added by hand (see
    # start_manual_other_whale) exists only in confirmed_polygons and never had
    # a row in candidate_pool.
    all_whale_ids = set(rows_by_whale_id) | {
        whale_id for whale_id in confirmed_polygons if whale_id != exclude_whale_id
    }
    if not all_whale_ids:
        return []
    frame_h, frame_w = frame_shape[:2]
    lines = []
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
    has_combined = False
    for other_whale_id in all_whale_ids:
        confirmed_polygon = confirmed_polygons.get(other_whale_id)
        if confirmed_polygon and len(confirmed_polygon) >= 3:
            lines.append(_polygon_to_yolo_line(confirmed_polygon, frame_w, frame_h))
            continue
        other_row = rows_by_whale_id.get(other_whale_id)
        if other_row is None:
            # Added by hand but with no confirmed outline (the user answered No
            # to "draw it from scratch?"): nothing to label, not an error.
            continue
        combined = np.maximum(
            combined,
            reconstruct_mask_from_crop(
                other_row.get("_auto_mask_crop"),
                other_row.get("_auto_mask_crop_x1"),
                other_row.get("_auto_mask_crop_y1"),
                frame_h,
                frame_w,
            ),
        )
        has_combined = True
    if has_combined and combined.any():
        lines.extend(mask_to_yolo_segmentation_lines(combined, epsilon_fraction))
    return lines


def save_yolo_seg_training_example(
    clean_frame,
    mask,
    frame_filename,
    mask_database_folder,
    epsilon_fraction=0.0005,
    override_polygon=None,
    extra_instance_lines=None,
):
    """Save the clean image, the exact mask and the YOLO-Seg label.

    If `override_polygon` is given (pixel vertices the user adjusted after the
    automatic simplification), the saved mask and label are rebuilt from that
    polygon instead of from `mask`: it is exactly what the user corrected, so
    it must not be simplified again.

    `extra_instance_lines` (optional, see other_whale_instance_lines) are label
    lines for OTHER animals visible in the same image but not under review.
    They are appended to the label file and never touch `mask`/`mask_to_save`,
    which stay those of the main animal.
    """
    if mask is None:
        return False
    frame_h, frame_w = mask.shape[:2]
    if override_polygon and len(override_polygon) >= 3:
        # int32 only to rasterize the saved mask, since pixels are whole
        # numbers. The YOLO label below uses the original polygon coordinates at
        # full precision: the user can drag a vertex to inside a pixel (see
        # review.py), and rounding here would lose that in the training file.
        mask_to_save = np.zeros((frame_h, frame_w), dtype=np.uint8)
        fill_points = np.array(override_polygon, dtype=np.int32)
        cv2.fillPoly(mask_to_save, [fill_points], 255)
        label_lines = [_polygon_to_yolo_line(override_polygon, frame_w, frame_h)]
    else:
        mask_to_save = (mask > 0).astype(np.uint8) * 255
        label_lines = mask_to_yolo_segmentation_lines(mask, epsilon_fraction)
        if not label_lines:
            return False
    label_lines = label_lines + list(extra_instance_lines or [])

    images_folder = mask_database_folder / "images"
    labels_folder = mask_database_folder / "labels"
    masks_folder = mask_database_folder / "masks"
    for database_folder in (
        images_folder,
        labels_folder,
        masks_folder,
    ):
        database_folder.mkdir(parents=True, exist_ok=True)

    frame_path = images_folder / frame_filename
    label_path = labels_folder / f"{Path(frame_filename).stem}.txt"
    mask_path = masks_folder / frame_filename
    if not cv2.imwrite(str(frame_path), clean_frame):
        return False
    if not cv2.imwrite(str(mask_path), mask_to_save):
        return False
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    yaml_path = mask_database_folder / "dataset.yaml"
    yaml_root = mask_database_folder.resolve().as_posix().replace("'", "''")
    yaml_path.write_text(
        "path: '" + yaml_root + "'\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: whale\n",
        encoding="utf-8",
    )
    return True


def build_model_human_comparison(source_df, measure_in_meters=True):
    """Affianca geometria automatica e revisione umana con relativi errori."""

    def numeric(column_name):
        if column_name not in source_df.columns:
            return pd.Series(np.nan, index=source_df.index, dtype=float)
        return pd.to_numeric(source_df[column_name], errors="coerce")

    def point_distance(x_a, y_a, x_b, y_b):
        return np.hypot(x_a - x_b, y_a - y_b)

    def percentage_error(automatic, human):
        denominator = human.abs().where(human.abs() > 1e-12)
        return (automatic - human).abs() / denominator * 100.0

    measurement_columns = {
        "measured_length",
        "measured_length_px",
    }
    measurement_columns.update(
        {
            f"width_{pct}pct"
            for pct in range(5, 100, 5)
        }
    )
    measurement_columns.update(
        {
            f"width_{pct}pct_px"
            for pct in range(5, 100, 5)
        }
    )

    comparison_data = {}
    for column_name in source_df.columns:
        if column_name.startswith("_") or column_name in measurement_columns:
            continue
        if not measure_in_meters and column_name in {"h", "gsd", "tilt_deg"}:
            continue
        if column_name == "auto_length":
            output_name = (
                "screening_auto_length_m"
                if measure_in_meters
                else "screening_auto_length_px"
            )
        elif column_name == "area_m2" and not measure_in_meters:
            output_name = "area_px2"
        else:
            output_name = column_name
        comparison_data[output_name] = source_df[column_name]

    gsd = numeric("gsd")
    auto_length = numeric("_model_auto_length_m")
    human_length = numeric("measured_length")
    if measure_in_meters:
        comparison_data["automatic_length_m"] = auto_length
        comparison_data["human_length_m"] = human_length
        comparison_data["automatic_length_px"] = auto_length / gsd
        comparison_data["human_length_px"] = numeric("measured_length_px")
        comparison_data["length_error_m"] = auto_length - human_length
        comparison_data["length_abs_error_m"] = (
            auto_length - human_length
        ).abs()
    else:
        comparison_data["automatic_length_px"] = auto_length
        comparison_data["human_length_px"] = numeric("measured_length_px")
        comparison_data["length_error_px"] = auto_length - human_length
        comparison_data["length_abs_error_px"] = (
            auto_length - human_length
        ).abs()
    comparison_data["length_error_pct"] = percentage_error(
        auto_length,
        human_length,
    )
    comparison_data["human_measurement_mode"] = source_df[
        "_human_measurement_mode"
    ]

    for point_name in ("head", "fluke"):
        auto_x = numeric(f"_model_auto_{point_name}_x")
        auto_y = numeric(f"_model_auto_{point_name}_y")
        human_x = numeric(f"_human_{point_name}_x")
        human_y = numeric(f"_human_{point_name}_y")
        error_px = point_distance(auto_x, auto_y, human_x, human_y)

        comparison_data[f"automatic_{point_name}_x"] = auto_x
        comparison_data[f"automatic_{point_name}_y"] = auto_y
        comparison_data[f"human_{point_name}_x"] = human_x
        comparison_data[f"human_{point_name}_y"] = human_y
        comparison_data[f"{point_name}_delta_x_px"] = auto_x - human_x
        comparison_data[f"{point_name}_delta_y_px"] = auto_y - human_y
        comparison_data[f"{point_name}_error_px"] = error_px
        if measure_in_meters:
            comparison_data[f"{point_name}_error_m"] = error_px * gsd

    for pct in range(5, 100, 5):
        prefix = f"width_{pct}pct"
        auto_width = numeric(f"_model_auto_{prefix}_m")
        human_width = numeric(f"_human_{prefix}_m")
        if measure_in_meters:
            comparison_data[f"automatic_{prefix}_m"] = auto_width
            comparison_data[f"human_{prefix}_m"] = human_width
            comparison_data[f"automatic_{prefix}_px"] = auto_width / gsd
            comparison_data[f"human_{prefix}_px"] = numeric(f"{prefix}_px")
            comparison_data[f"{prefix}_error_m"] = auto_width - human_width
            comparison_data[f"{prefix}_abs_error_m"] = (
                auto_width - human_width
            ).abs()
        else:
            comparison_data[f"automatic_{prefix}_px"] = auto_width
            comparison_data[f"human_{prefix}_px"] = numeric(f"{prefix}_px")
            comparison_data[f"{prefix}_error_px"] = auto_width - human_width
            comparison_data[f"{prefix}_abs_error_px"] = (
                auto_width - human_width
            ).abs()
        comparison_data[f"{prefix}_error_pct"] = percentage_error(
            auto_width,
            human_width,
        )

        auto_top_x = numeric(f"_model_auto_{prefix}_top_x")
        auto_top_y = numeric(f"_model_auto_{prefix}_top_y")
        auto_bottom_x = numeric(f"_model_auto_{prefix}_bottom_x")
        auto_bottom_y = numeric(f"_model_auto_{prefix}_bottom_y")
        human_top_x = numeric(f"_human_{prefix}_top_x")
        human_top_y = numeric(f"_human_{prefix}_top_y")
        human_bottom_x = numeric(f"_human_{prefix}_bottom_x")
        human_bottom_y = numeric(f"_human_{prefix}_bottom_y")

        comparison_data[f"automatic_{prefix}_top_x"] = auto_top_x
        comparison_data[f"automatic_{prefix}_top_y"] = auto_top_y
        comparison_data[f"automatic_{prefix}_bottom_x"] = auto_bottom_x
        comparison_data[f"automatic_{prefix}_bottom_y"] = auto_bottom_y
        comparison_data[f"human_{prefix}_top_x"] = human_top_x
        comparison_data[f"human_{prefix}_top_y"] = human_top_y
        comparison_data[f"human_{prefix}_bottom_x"] = human_bottom_x
        comparison_data[f"human_{prefix}_bottom_y"] = human_bottom_y

        direct_error = (
            point_distance(
                auto_top_x,
                auto_top_y,
                human_top_x,
                human_top_y,
            )
            + point_distance(
                auto_bottom_x,
                auto_bottom_y,
                human_bottom_x,
                human_bottom_y,
            )
        )
        swapped_error = (
            point_distance(
                auto_top_x,
                auto_top_y,
                human_bottom_x,
                human_bottom_y,
            )
            + point_distance(
                auto_bottom_x,
                auto_bottom_y,
                human_top_x,
                human_top_y,
            )
        )
        both_valid = direct_error.notna() & swapped_error.notna()
        use_swapped = both_valid & (swapped_error < direct_error)
        endpoint_error_px = direct_error.where(
            ~use_swapped,
            swapped_error,
        ) / 2.0
        endpoint_error_px = endpoint_error_px.where(both_valid)
        pairing = pd.Series("", index=source_df.index, dtype=object)
        pairing.loc[both_valid & ~use_swapped] = "direct"
        pairing.loc[use_swapped] = "swapped"
        comparison_data[f"{prefix}_endpoint_pairing"] = pairing
        comparison_data[f"{prefix}_mean_endpoint_error_px"] = endpoint_error_px
        if measure_in_meters:
            comparison_data[f"{prefix}_mean_endpoint_error_m"] = (
                endpoint_error_px * gsd
            )

    return pd.DataFrame(comparison_data, index=source_df.index)


def _comparison_error_detail_columns():
    """Detail columns dropped from the exported CSV.

    build_model_human_comparison computes errors, deltas, top/bottom
    coordinates and pairing info because print_model_accuracy_estimate needs
    them for its summary. They only bloat the final CSV, which already carries
    the automatic and human measurements in px and m, so they are removed from
    the export but not from the calculation.
    """
    columns = {
        "length_error_m", "length_abs_error_m", "length_error_pct",
        "length_error_px", "length_abs_error_px",
        "head_error_m", "head_error_px", "head_delta_x_px", "head_delta_y_px",
        "fluke_error_m", "fluke_error_px", "fluke_delta_x_px", "fluke_delta_y_px",
    }
    for pct in range(5, 100, 5):
        prefix = f"width_{pct}pct"
        columns.update(
            {
                f"{prefix}_error_m",
                f"{prefix}_abs_error_m",
                f"{prefix}_error_px",
                f"{prefix}_abs_error_px",
                f"{prefix}_error_pct",
                f"automatic_{prefix}_top_x",
                f"automatic_{prefix}_top_y",
                f"automatic_{prefix}_bottom_x",
                f"automatic_{prefix}_bottom_y",
                f"human_{prefix}_top_x",
                f"human_{prefix}_top_y",
                f"human_{prefix}_bottom_x",
                f"human_{prefix}_bottom_y",
                f"{prefix}_endpoint_pairing",
                f"{prefix}_mean_endpoint_error_px",
                f"{prefix}_mean_endpoint_error_m",
            }
        )
    return columns


def print_model_accuracy_estimate(comparison_df, measure_in_meters=True):
    """Print how close the model's measurements were to the reviewed ones."""

    def paired_values(automatic_column, human_column):
        automatic = pd.to_numeric(
            comparison_df[automatic_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        human = pd.to_numeric(
            comparison_df[human_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(automatic) & np.isfinite(human)
        return automatic[valid], human[valid]

    def error_statistics(automatic, human):
        if len(automatic) == 0:
            return None
        errors = automatic - human
        nonzero = np.abs(human) > 1e-12
        return {
            "count": len(errors),
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mape": (
                float(np.mean(np.abs(errors[nonzero] / human[nonzero])) * 100)
                if np.any(nonzero)
                else np.nan
            ),
        }

    print("\nMODEL ESTIMATE VS HUMAN CORRECTION")
    unit_suffix = "m" if measure_in_meters else "px"
    auto_length, human_length = paired_values(
        f"automatic_length_{unit_suffix}",
        f"human_length_{unit_suffix}",
    )
    length_stats = error_statistics(auto_length, human_length)
    if length_stats is None:
        print("  Length: no comparable frames.")
    else:
        print(
            "  Length "
            f"({length_stats['count']} frames): "
            f"bias {length_stats['bias']:+.3f} {unit_suffix}, "
            f"MAE {length_stats['mae']:.3f} {unit_suffix}, "
            f"RMSE {length_stats['rmse']:.3f} {unit_suffix}, "
            f"MAPE {length_stats['mape']:.2f}%"
        )

    for point_name, label in (("head", "head"), ("fluke", "fluke")):
        errors_px = pd.to_numeric(
            comparison_df[f"{point_name}_error_px"],
            errors="coerce",
        )
        valid = errors_px.notna()
        if valid.any():
            point_message = (
                f"  {label.capitalize()} point ({int(valid.sum())} frames): "
                f"mean error {errors_px[valid].mean():.2f} px"
            )
            if measure_in_meters:
                errors_m = pd.to_numeric(
                    comparison_df[f"{point_name}_error_m"],
                    errors="coerce",
                )
                metric_valid = valid & errors_m.notna()
                if metric_valid.any():
                    point_message += (
                        f" / {errors_m[metric_valid].mean():.3f} m"
                    )
            print(point_message)

    width_automatic = []
    width_human = []
    endpoint_errors_px = []
    endpoint_errors_m = []
    for pct in range(5, 100, 5):
        automatic, human = paired_values(
            f"automatic_width_{pct}pct_{unit_suffix}",
            f"human_width_{pct}pct_{unit_suffix}",
        )
        if len(automatic):
            width_automatic.append(automatic)
            width_human.append(human)
        errors_px = pd.to_numeric(
            comparison_df[f"width_{pct}pct_mean_endpoint_error_px"],
            errors="coerce",
        ).dropna()
        if not errors_px.empty:
            endpoint_errors_px.append(errors_px.to_numpy(dtype=float))
        if measure_in_meters:
            errors_m = pd.to_numeric(
                comparison_df[f"width_{pct}pct_mean_endpoint_error_m"],
                errors="coerce",
            ).dropna()
            if not errors_m.empty:
                endpoint_errors_m.append(errors_m.to_numpy(dtype=float))

    if width_automatic:
        width_stats = error_statistics(
            np.concatenate(width_automatic),
            np.concatenate(width_human),
        )
        print(
            "  Widths "
            f"({width_stats['count']} segments): "
            f"bias {width_stats['bias']:+.3f} {unit_suffix}, "
            f"MAE {width_stats['mae']:.3f} {unit_suffix}, "
            f"RMSE {width_stats['rmse']:.3f} {unit_suffix}, "
            f"MAPE {width_stats['mape']:.2f}%"
        )
    else:
        print("  Widths: no comparable segments.")

    if endpoint_errors_px:
        all_endpoint_errors_px = np.concatenate(endpoint_errors_px)
        endpoint_message = (
            "  Width endpoints: mean error "
            f"{np.mean(all_endpoint_errors_px):.2f} px"
        )
        if measure_in_meters and endpoint_errors_m:
            all_endpoint_errors_m = np.concatenate(endpoint_errors_m)
            endpoint_message += (
                f" / {np.mean(all_endpoint_errors_m):.3f} m"
            )
        print(endpoint_message)


def prepare_reviewed_results(source_df, measure_in_meters=True):
    """Drop the internal columns and lay out the CSV for the chosen unit."""
    internal_columns = [
        column_name
        for column_name in source_df.columns
        if column_name.startswith("_")
    ]
    reviewed = source_df.drop(columns=internal_columns)
    if measure_in_meters:
        return reviewed

    non_pixel_measurements = [
        "measured_length",
        "h",
        "gsd",
        "tilt_deg",
    ]
    non_pixel_measurements.extend(
        f"width_{pct}pct" for pct in range(5, 100, 5)
    )
    reviewed = reviewed.drop(
        columns=[
            column_name
            for column_name in non_pixel_measurements
            if column_name in reviewed.columns
        ]
    )
    return reviewed.rename(
        columns={
            "area_m2": "area_px2",
            "auto_length": "screening_auto_length_px",
        }
    )


def write_automatic_only_csv(candidate_pool, output_path, config):
    """Export the candidates exactly as screening left them, skipping review.

    Used when `config.skip_review` is set. The CSV holds one automatic
    measurement per frame. If the source is a folder of frames, the original
    image file name is added too; otherwise the only identifier is
    `frame_pos`/`video_second`.
    """
    if candidate_pool.empty:
        print("\nNo candidates found: no CSV generated.")
        return None

    internal_columns = [
        column
        for column in candidate_pool.columns
        if column.startswith("_auto_mask_crop")
    ] + ["_candidate_id"]
    export_df = candidate_pool.drop(
        columns=[column for column in internal_columns if column in candidate_pool.columns]
    ).copy()

    if Path(config.video_path).is_dir():
        filenames = FrameFolderCapture(
            config.video_path, sequential=use_sequential_indexing(config)
        ).all_filenames()
        export_df.insert(0, "frame_name", export_df["frame_pos"].map(filenames))

    csv_path = output_path / f"automatic_analysis_{config.video_id}.csv"
    export_df.to_csv(csv_path, index=False)
    print(f"\nAutomatic analysis (no review): {len(export_df)} frames -> {csv_path}")
    return csv_path


def write_final_csvs(final_df, output_path, video_id, measure_in_meters):
    """Drop outliers by IQR (in metres), write the two final CSVs, print the
    summary."""
    res_df = final_df.dropna(subset=["measured_length"]).copy()
    if measure_in_meters and len(res_df) > 2:
        q1, q3 = res_df["measured_length"].quantile(0.25), res_df["measured_length"].quantile(0.75)
        iqr = q3 - q1
        f_res = res_df[
            (res_df["measured_length"] >= q1 - 1.5 * iqr) &
            (res_df["measured_length"] <= q3 + 1.5 * iqr)
        ]
        export_df = f_res.copy()
    elif not res_df.empty:
        export_df = res_df.copy()
        if measure_in_meters:
            print(
                "\nInsufficient sample for the IQR filter; "
                f"saved {len(res_df)} measurements anyway."
            )
    else:
        export_df = None
        print("\nNo frames accepted: no CSV generated.")

    if export_df is None:
        return None

    reviewed_df = prepare_reviewed_results(
        export_df,
        measure_in_meters=measure_in_meters,
    )
    comparison_df = build_model_human_comparison(
        export_df,
        measure_in_meters=measure_in_meters,
    )

    reviewed_path = output_path / f"final_results_{video_id}.csv"
    automatic_path = output_path / f"final_results_automatic_{video_id}.csv"
    reviewed_df.to_csv(reviewed_path, index=False)
    # The summary below (print_model_accuracy_estimate) still needs the detail
    # columns, so they are dropped only from the copy written to disk, not from
    # comparison_df itself.
    comparison_export_df = comparison_df.drop(
        columns=[
            column_name
            for column_name in _comparison_error_detail_columns()
            if column_name in comparison_df.columns
        ]
    )
    comparison_export_df.to_csv(automatic_path, index=False)

    result_length_column = (
        "measured_length"
        if measure_in_meters
        else "measured_length_px"
    )
    print(
        f"\nFINAL AVERAGE LENGTH: "
        f"{reviewed_df[result_length_column].mean():.3f} "
        f"{'m' if measure_in_meters else 'px'}"
    )
    print(f"Reviewed results: {reviewed_path}")
    print(f"Automatic/human comparison: {automatic_path}")
    print_model_accuracy_estimate(
        comparison_df,
        measure_in_meters=measure_in_meters,
    )
    return reviewed_path, automatic_path


def _self_check():
    """Check that saving to the YOLO-Seg dataset writes every expected file,
    both with the automatic polygon and with a hand-edited one
    (override_polygon)."""
    import tempfile

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 60), 40, 255, -1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        database_folder = Path(tmp_dir) / "mask_database"

        assert save_yolo_seg_training_example(
            frame, mask, "auto.png", database_folder
        )
        assert (database_folder / "images" / "auto.png").is_file()
        assert (database_folder / "masks" / "auto.png").is_file()
        assert (database_folder / "labels" / "auto.txt").is_file()
        assert (database_folder / "dataset.yaml").is_file()
        auto_label = (database_folder / "labels" / "auto.txt").read_text().split()
        assert auto_label[0] == "0"
        auto_coords = [float(value) for value in auto_label[1:]]
        assert len(auto_coords) % 2 == 0 and len(auto_coords) >= 6
        assert all(0.0 <= value <= 1.0 for value in auto_coords)

        edited_polygon = [[40, 20], [120, 20], [120, 100], [40, 100]]
        assert save_yolo_seg_training_example(
            frame,
            mask,
            "edited.png",
            database_folder,
            override_polygon=edited_polygon,
        )
        edited_label = (database_folder / "labels" / "edited.txt").read_text().split()
        edited_coords = [float(value) for value in edited_label[1:]]
        assert len(edited_coords) == len(edited_polygon) * 2
        assert abs(edited_coords[0] - 40 / 160) < 1e-5
        assert abs(edited_coords[1] - 20 / 120) < 1e-5
        edited_mask = cv2.imread(
            str(database_folder / "masks" / "edited.png"), cv2.IMREAD_GRAYSCALE
        )
        assert edited_mask[60, 80] == 255  # dentro al rettangolo modificato
        assert edited_mask[5, 5] == 0  # outside the rectangle

        assert not save_yolo_seg_training_example(frame, None, "none.png", database_folder)

    print("export.py self-check OK")


if __name__ == "__main__":
    _self_check()
