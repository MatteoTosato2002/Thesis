"""Quick regression check for the whalevision package.

It only exercises pure functions on synthetic data: no video, no GUI. Run it
with `python3 -m whalevision.smoke_test`. It does not replace starting the app
for real, which should still be done by hand after changing whalevision/.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .config import AppConfig, load_last_configuration, save_last_configuration
from .device import choose_device, is_memory_error, predict_adaptive
from .export import (
    build_annotated_result_frame,
    other_whale_instance_lines,
    prepare_reviewed_results,
    reconstruct_mask_from_crop,
)
from .pipeline import collapse_fragmented_track_ids, rejected_frames_for_whale
from .landmarks import (
    LANDMARK_DEFINITIONS,
    load_existing_landmark_records,
    make_landmark_record,
    write_landmark_csv,
)
from .measurement import (
    measure_length_pca_along_axis,
    bezier_curve_points,
    measure_length_pca_tail_concavity,
    measure_length_pca,
    measure_length_pca_notch_curve,
    measure_length_pca_robust_notch,
    measure_widths_along_curve,
    clean_review_mask,
    deduplicate_detections_by_mask_iou,
    draw_cross_marker,
    expand_bbox,
    mask_to_polygons,
    polygon_to_mask,
    sanitize_name,
)
from .review import ReviewState, measuring_line_points
from .screening import CandidatePreviewWindow, _box_matches_any, classify_length_quartiles
from .telemetry import extract_flight_date, get_altitude, load_telemetry


def point_at_percent(points, percent):
    """The point at this percentage along a line, walking it segment by
    segment: the same walk measure_widths_along_curve does to place a width."""
    points = np.asarray(points, dtype=float)
    step_lengths = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    walked = np.concatenate([[0.0], np.cumsum(step_lengths)])
    target = walked[-1] * percent / 100.0
    i = int(np.searchsorted(walked, target)) - 1
    i = max(0, min(i, len(points) - 2))
    fraction = (target - walked[i]) / step_lengths[i]
    return points[i] + fraction * (points[i + 1] - points[i])


def check(label, condition):
    status = "OK" if condition else "FALLITO"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(f"Smoke test fallito: {label}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        cfg = AppConfig(video_id="smoke")
        import whalevision.config as config_module
        config_module.LAST_CONFIG_PATH = tmp_path / "cfg.json"
        save_last_configuration(cfg)
        check("AppConfig round-trip", load_last_configuration().video_id == "smoke")

        check("choose_device('CPU...') == 'cpu'", choose_device("CPU (lower memory risk)") == "cpu")
        check("is_memory_error spots a Metal out-of-memory error", is_memory_error(RuntimeError("MPS backend out of memory")))

        # predict_adaptive must call model.track(persist=True, ...) when given
        # tracker=..., and model.predict(...) otherwise. That switch is what
        # gives each whale a lasting ID (see screening.py), so a fake model is
        # enough to notice if it ever stops working.
        class _FakeResult:
            boxes = None

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def predict(self, *args, **kwargs):
                self.calls.append(("predict", kwargs))
                return [_FakeResult()]

            def track(self, *args, **kwargs):
                self.calls.append(("track", kwargs))
                return [_FakeResult()]

        fake = _FakeModel()
        predict_adaptive(fake, np.zeros((10, 10, 3), dtype=np.uint8), 640, "fake")
        check(
            "predict_adaptive without a tracker calls model.predict",
            fake.calls[-1][0] == "predict",
        )
        predict_adaptive(
            fake, np.zeros((10, 10, 3), dtype=np.uint8), 640, "fake",
            tracker="bytetrack.yaml",
        )
        check(
            "predict_adaptive with a tracker calls model.track(persist=True, ...)",
            fake.calls[-1][0] == "track" and fake.calls[-1][1].get("persist") is True
            and fake.calls[-1][1].get("tracker") == "bytetrack.yaml",
        )

        # Real regression: one whale in the video, but the tracker loses and
        # regains the ID between sampled frames, leaving several ghost whale_ids
        # for the same animal, never seen together in one frame. They must
        # collapse back to a single ID.
        fragmented_single_whale = pd.DataFrame({
            "frame_pos": [0, 10, 20, 30, 40, 50, 60, 70],
            "whale_id": [0, 1, 2, 3, 4, 5, 6, 7],
        })
        collapsed = collapse_fragmented_track_ids(fragmented_single_whale)
        check(
            "collapse_fragmented_track_ids merges fragments of the same animal",
            collapsed["whale_id"].nunique() == 1,
        )
        # Two REAL animals must never be merged: appearing together in even one
        # frame is direct proof there is more than one whale.
        real_two_whales = pd.DataFrame({
            "frame_pos": [0, 0, 10, 20, 20],
            "whale_id": [1, 2, 1, 1, 2],
        })
        not_collapsed = collapse_fragmented_track_ids(real_two_whales)
        check(
            "collapse_fragmented_track_ids leaves two whales seen together alone",
            not_collapsed["whale_id"].nunique() == 2,
        )
        # With expected_whale_count=1 the user said there is one animal, so
        # merge even when two whale_ids show up in the same frame (NMS letting
        # two nearly identical boxes through for one whale). The automatic
        # "proof" can be a false alarm and is ignored here.
        forced = collapse_fragmented_track_ids(real_two_whales, expected_whale_count=1)
        check(
            "expected_whale_count=1 forces one group even with false co-occurrences",
            forced["whale_id"].nunique() == 1,
        )

        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 20:180] = 255
        widths = measure_widths_along_curve(mask, [(20, 50), (100, 50), (180, 50)], gsd=0.01)
        check("measure_widths_along_curve returns 19 points", len(widths) == 19)
        # Needs a mask with a real centred notch: with the centring check in
        # place, a plain rectangle (no notch at all) correctly makes the active
        # method give up instead of inventing a measurement.
        notched_mask = np.zeros((100, 220), dtype=np.uint8)
        notched_mask[40:60, 10:170] = 255
        notched_mask[25:75, 170:210] = 255
        for x in range(180, 208):
            depth = int((x - 180) / 27 * 20)
            notched_mask[50 - depth:50 + depth, x] = 0
        pca = measure_length_pca(notched_mask, gsd=0.01)
        check("measure_length_pca (the method actually in use)", pca is not None and pca[0] > 0)
        pca_axis = measure_length_pca_along_axis(mask, gsd=0.01)
        check("measure_length_pca_along_axis (experimental, not in use)", pca_axis is not None and pca_axis[0] > 0)
        pca_concav = measure_length_pca_tail_concavity(mask, gsd=0.01)
        check(
            "measure_length_pca_tail_concavity (experimental, not in use)",
            pca_concav is not None and pca_concav[0] > 0,
        )
        pca_robust = measure_length_pca_robust_notch(mask, gsd=0.01)
        check("measure_length_pca_robust_notch (experimental, not in use)", pca_robust is not None and pca_robust[0] > 0)
        pca_curve = measure_length_pca_notch_curve(notched_mask, gsd=0.01)
        check("measure_length_pca_notch_curve (experimental, not in use)", pca_curve is not None and pca_curve[0] > 0)
        # Real regression: with the Bezier box ticked (the default), the manual
        # length was measured along the curve while the widths were still placed
        # along the straight polyline between clicks. "35% of the body" then
        # meant two different places, and the width segments did not even touch
        # the drawn curve.
        traced_points = [(100.0, 300.0), (400.0, 220.0), (700.0, 240.0), (1000.0, 330.0)]
        curve = bezier_curve_points(traced_points)

        state = ReviewState()
        state.manual_measure_points = list(traced_points)
        state.bezier_length_mode = True
        check(
            "with the Bezier box on, widths follow the curve, not the polyline",
            np.allclose(np.asarray(measuring_line_points(state)), curve),
        )
        state.bezier_length_mode = False
        check(
            "with the Bezier box off, widths follow the clicked points",
            measuring_line_points(state) == traced_points,
        )
        # Head and fluke only: there is no curve to follow, so nothing changes.
        state.bezier_length_mode = True
        state.manual_measure_points = [(100.0, 300.0), (1000.0, 330.0)]
        check(
            "with only head and fluke the line stays straight",
            measuring_line_points(state) == [(100.0, 300.0), (1000.0, 330.0)],
        )

        # Widths now really start from the same curve the length was measured
        # along.
        curved_mask = np.zeros((700, 1200), dtype=np.uint8)
        for cx, cy in bezier_curve_points(traced_points, 400):
            cv2.circle(curved_mask, (int(cx), int(cy)), 50, 255, -1)
        state.manual_measure_points = list(traced_points)
        widths_on_curve = measure_widths_along_curve(
            curved_mask, measuring_line_points(state), gsd=0.01
        )
        check("widths along the curve are still 19", len(widths_on_curve) == 19)

        # Compare against a very densely sampled copy of the curve, so "on the
        # curve" really means on the curve and not just near a sample.
        fine_curve = bezier_curve_points(traced_points, 4000)

        def distance_from_curve(point):
            return float(
                np.min(np.hypot(fine_curve[:, 0] - point[0], fine_curve[:, 1] - point[1]))
            )

        straight_line_widths = measure_widths_along_curve(
            curved_mask, traced_points, gsd=0.01
        )
        worst_on_curve = max(
            distance_from_curve(point_at_percent(curve, w["pct"]))
            for w in widths_on_curve
        )
        worst_off_curve = max(
            distance_from_curve(point_at_percent(np.asarray(traced_points), w["pct"]))
            for w in straight_line_widths
        )
        check(
            "every width starts from a point on the curve; before the fix they "
            f"were up to {worst_off_curve:.0f} px off it",
            worst_on_curve < 1.0 and worst_off_curve > 10.0,
        )

        expanded = expand_bbox(np.array([[10, 10, 30, 20]], dtype=np.float32), (100, 200, 3), 0.10)
        check("expand_bbox grows the box without leaving the frame", expanded[0][0] < 10 and expanded[0][2] > 30)
        check("clean_review_mask keeps the shape", clean_review_mask(mask, (100, 200, 3)).shape == (100, 200))
        cross_canvas = np.zeros((50, 50, 3), dtype=np.uint8)
        draw_cross_marker(cross_canvas, (25, 25), 7, (0, 255, 255))
        check("draw_cross_marker draws something visible", int((cross_canvas > 0).sum()) > 0)
        # Regression: the same draw_cross_marker draws both the fixed points at
        # image resolution (landmarks, polygon, SAM) and, in review.py, the
        # draggable ones at screen resolution after the zoom resize
        # (INTER_NEAREST). It must stay crisp and snapped to whole pixels in
        # both cases, or anti-aliasing would survive the enlargement as a blurry
        # block instead of a sharp mark (see its docstring). If it went back to
        # sub-pixel, two centres inside the same whole pixel would draw
        # differently and this check would fail.
        canvas_a = np.zeros((50, 50, 3), dtype=np.uint8)
        canvas_b = np.zeros((50, 50, 3), dtype=np.uint8)
        draw_cross_marker(canvas_a, (25.1, 25.1), 7, (0, 255, 255))
        draw_cross_marker(canvas_b, (25.4, 25.4), 7, (0, 255, 255))
        check(
            "draw_cross_marker stays crisp (not sub-pixel): same whole pixel, same output",
            np.array_equal(canvas_a, canvas_b),
        )
        # Thickness comes from the UI slider ("Cross marker thickness"): a
        # thicker stroke must light up more pixels, not be ignored.
        canvas_thin = np.zeros((50, 50, 3), dtype=np.uint8)
        canvas_thick = np.zeros((50, 50, 3), dtype=np.uint8)
        draw_cross_marker(canvas_thin, (25, 25), 7, (0, 255, 255), thickness=1)
        draw_cross_marker(canvas_thick, (25, 25), 7, (0, 255, 255), thickness=5)
        check(
            "draw_cross_marker: a higher thickness lights up more pixels",
            int((canvas_thick > 0).sum()) > int((canvas_thin > 0).sum()),
        )

        # yolo_none / yolo_low_conf / yolo_mask: whatever follows the ":" (for
        # example the confidence value) must be left out of the lookup in the
        # fixed dictionaries, or every distinct value would fall through to the
        # default colour and label. It still has to appear in the label shown.
        base, detail = CandidatePreviewWindow._reason_parts("yolo_low_conf:42% seen, 70% required")
        check("_reason_parts splits the base code from the detail", base == "yolo_low_conf" and "42%" in detail)
        check(
            "_reason_label uses the fixed label and appends the detail",
            CandidatePreviewWindow._reason_label("yolo_low_conf:42% seen, 70% required")
            == "detected below the confidence threshold (42% seen, 70% required)",
        )
        check(
            "_reason_label with no detail is just the plain label",
            CandidatePreviewWindow._reason_label("yolo_none") == "no whale detected",
        )
        check(
            "yolo_mask has its own label, separate from yolo_none",
            CandidatePreviewWindow._REASON_LABELS["yolo_mask"] != CandidatePreviewWindow._REASON_LABELS["yolo_none"],
        )
        # Real regression, found on real data with NTSC fps (23.976...). In
        # run_screening, rejected_frames turned seconds into frames with
        # int(sec * fps), which truncates. For sec=1 (the first valid second of
        # an analysis starting at 1.0) that gave frame 23, which divided back by
        # fps in the graph (screening.py, "Preview graph") came out as 0.959s:
        # below the 1.0s lower bound, so the tick fell outside the visible range
        # and vanished. round() picks the nearest frame (24) instead, and the
        # rebuilt time (1.001s) stays in range. This checks the property for ANY
        # start second with a non-integer fps, not just this one fps.
        for fps_value in (23.976023976023978, 29.97002997002997, 59.94005994005994):
            frame_number = round(1 * fps_value)
            reconstructed_time = frame_number / fps_value
            check(
                f"round(sec*fps) keeps the first second from falling below itself (fps={fps_value:.3f})",
                reconstructed_time >= 1.0,
            )

        # classify_length_quartiles: IQR outliers and candidates outside the
        # requested quartiles used to be dropped from the final pool with no way
        # for the user to see them. They now stay, marked OUTLIER_QUARTILE. The
        # test data is 9 values clustered around 9-10.6 (normal quartiles 1-4)
        # plus one extreme outlier (50.0) and one NaN.
        lengths = pd.Series([9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 50.0, np.nan])
        quartile, is_iqr_outlier, outside_requested, q0, q4 = classify_length_quartiles(lengths, [3, 4])
        check(
            "classify_length_quartiles keeps every row (none dropped)",
            len(quartile) == len(lengths),
        )
        check("classify_length_quartiles: NaN -> quartile 0", quartile.iloc[10] == 0)
        check(
            "classify_length_quartiles: the extreme IQR outlier is marked OUTLIER_QUARTILE",
            is_iqr_outlier.iloc[9] and quartile.iloc[9] == CandidatePreviewWindow.OUTLIER_QUARTILE,
        )
        check(
            "classify_length_quartiles: a normal value outside target_quartiles=[3,4] "
            "becomes OUTLIER_QUARTILE instead of being dropped",
            quartile.iloc[0] == CandidatePreviewWindow.OUTLIER_QUARTILE and not is_iqr_outlier.iloc[0],
        )
        check(
            "classify_length_quartiles: a value inside target_quartiles keeps its own quartile",
            quartile.iloc[7] in (3, 4),
        )
        quartile_no_filter, _, outside_requested_no_filter, _, _ = classify_length_quartiles(lengths, [])
        check(
            "classify_length_quartiles with empty target_quartiles marks no quartile outliers (IQR only)",
            not outside_requested_no_filter.iloc[0],
        )

        # Real regression (2026-09): with several animals expected, a rejected
        # frame needs a reason PER ANIMAL, not one for the whole frame.
        # Otherwise an animal accepted in a frame hid the reason a second animal
        # was rejected in that same frame. rejected_frames_for_whale picks, for
        # one whale_id, between that animal's own reason and the general one
        # (stored under the None key).
        multi_animal_rejected_frames = {
            10: {1: "unusual_shape"},
            20: {2: "off_center"},
            30: {None: "altitude"},
            40: {1: "unusual_shape", None: "yolo_extra_low_conf:12% seen, 70% required"},
        }
        whale1_view = rejected_frames_for_whale(multi_animal_rejected_frames, 1)
        whale2_view = rejected_frames_for_whale(multi_animal_rejected_frames, 2)
        check(
            "rejected_frames_for_whale: animal 1's own reason stays visible and "
            "animal 2's in the same frame does not overwrite it",
            whale1_view.get(10) == "unusual_shape" and 20 not in whale1_view,
        )
        check(
            "rejected_frames_for_whale: animal 2's own reason stays visible "
            "whatever animal 1's is",
            whale2_view.get(20) == "off_center" and 10 not in whale2_view,
        )
        check(
            "rejected_frames_for_whale: a general reason (None key, e.g. altitude) "
            "is visible to BOTH animals",
            whale1_view.get(30) == "altitude" and whale2_view.get(30) == "altitude",
        )
        check(
            "rejected_frames_for_whale: an animal's own reason wins over the general "
            "one when both exist for the same frame",
            whale1_view.get(40) == "unusual_shape",
        )

        # _box_matches_any tells whether a low-confidence detection is the same
        # animal already handled at the main confidence (same box, nothing new)
        # or evidence of a second untracked animal (box far from every known
        # one).
        known_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        check(
            "_box_matches_any: a box almost identical to a known one counts as the same animal",
            _box_matches_any(np.array([11.0, 11.0, 49.0, 49.0]), known_boxes),
        )
        check(
            "_box_matches_any: a box far from every known one is not the same animal",
            not _box_matches_any(np.array([200.0, 200.0, 240.0, 240.0]), known_boxes),
        )
        check(
            "_box_matches_any: no known boxes means never a match",
            not _box_matches_any(np.array([11.0, 11.0, 49.0, 49.0]), np.empty((0, 4))),
        )

        # Real regression: NMS compares boxes, not masks. For a long body, two
        # boxes covering different parts of the same animal can overlap little
        # while their masks are all but identical. This checks that comparing
        # the real masks drops the duplicate but keeps a genuinely separate
        # animal.
        class _FakeMasks:
            def __init__(self, arrays):
                self.data = [torch.tensor(arr.astype("float32")) for arr in arrays]

        class _FakeBoxes:
            def __init__(self, confidences):
                self.conf = torch.tensor(confidences)

            def __len__(self):
                return len(self.conf)

        class _FakeDetectionResult:
            def __init__(self, confidences, mask_arrays):
                self.boxes = _FakeBoxes(confidences)
                self.masks = _FakeMasks(mask_arrays)

        dedup_frame_shape = (100, 100, 3)
        mask_whale_full = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_whale_full, (30, 30), 20, 1, -1)
        mask_whale_subset = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_whale_subset, (30, 30), 15, 1, -1)
        mask_other_whale = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_other_whale, (80, 80), 15, 1, -1)
        dedup_result = _FakeDetectionResult(
            confidences=[0.9, 0.6, 0.8],
            mask_arrays=[mask_whale_full, mask_whale_subset, mask_other_whale],
        )
        kept = deduplicate_detections_by_mask_iou(dedup_result, dedup_frame_shape)
        check(
            "deduplicate_detections_by_mask_iou drops the duplicate box "
            "(a subset of the same mask) and keeps the separate animal",
            kept == [0, 2],
        )
        check("sanitize_name", sanitize_name(" héllo/world ") == "h_llo_world")

        # The dataset polygon now becomes the mask the measurement is taken
        # from (finalize_measurement_and_start_quality in review.py), so the
        # round trip mask -> polygon -> mask has to keep a comparable area or
        # the measurement would break.
        polygons = mask_to_polygons(mask, epsilon_fraction=0.01)
        rasterized = polygon_to_mask(polygons[0].tolist(), (100, 200, 3))
        original_area = int((mask > 0).sum())
        rasterized_area = int((rasterized > 0).sum())
        check(
            "polygon_to_mask ricostruisce un'area paragonabile a mask_to_polygons",
            abs(rasterized_area - original_area) / original_area < 0.05,
        )

        telemetry_csv = tmp_path / "telemetry.csv"
        pd.DataFrame({
            "sec_video": [0, 1, 2],
            "laser_altitude_fixed": [10.0, 11.0, 12.0],
            "flight_date": ["2026-07-01"] * 3,
        }).to_csv(telemetry_csv, index=False)
        telemetry = load_telemetry(
            str(telemetry_csv), "sec_video", "laser_altitude_fixed",
            use_tilt_filter=False, measure_in_meters=True,
        )
        check("load_telemetry disponibile", telemetry.available)
        check("get_altitude takes that second's sample, without interpolating",
              get_altitude(telemetry, 0.5) == 10.0)
        check("get_altitude never looks ahead",
              get_altitude(telemetry, 1.9) == 11.0)
        check(
            "extract_flight_date finds a date",
            extract_flight_date(telemetry.dataframe) == "2026-07-01",
        )
        # Regression: dayfirst=True swapped day and month even on unambiguous
        # ISO strings "YYYY-MM-DD hh:mm:ss". Real bug, found on real data: files
        # exported as "2025-05-04" instead of "2025-04-05". The column is named
        # "second" on purpose: it is the first name checked in preferred_names,
        # the same as in the telemetry where the bug appeared.
        real_world_dates = pd.DataFrame({
            "second": ["2025-04-05 11:02:57", "2025-04-05 11:02:58"],
        })
        check(
            "extract_flight_date does not swap day and month on ISO dates",
            extract_flight_date(real_world_dates) == "2025-04-05",
        )

        annotated = build_annotated_result_frame(
            np.zeros((50, 50, 3), dtype=np.uint8), None, [(1, 1), (20, 20)], [], {}
        )
        check("build_annotated_result_frame", annotated.shape == (50, 50, 3))
        df = pd.DataFrame({"measured_length": [10.0], "_internal": [1]})
        check("prepare_reviewed_results rimuove colonne interne", "_internal" not in prepare_reviewed_results(df).columns)

        # Real regression (2026-09): with several animals in one frame, the
        # training image saved for the reviewed animal still showed the others
        # in its pixels, but unlabelled. That is a false negative that confuses
        # training. other_whale_instance_lines adds a line for each other animal
        # from its automatic screening mask.
        other_whale_mask_crop = np.zeros((30, 30), dtype=np.uint8)
        cv2.circle(other_whale_mask_crop, (15, 15), 10, 255, -1)
        multi_whale_pool = pd.DataFrame({
            "frame_pos": [100, 100, 100, 200],
            "whale_id": [1, 2, 2, 1],
            "_auto_mask_crop": [None, other_whale_mask_crop, other_whale_mask_crop, None],
            "_auto_mask_crop_x1": [np.nan, 40.0, 40.0, np.nan],
            "_auto_mask_crop_y1": [np.nan, 40.0, 40.0, np.nan],
        })
        lines_for_whale_1 = other_whale_instance_lines(
            multi_whale_pool, frame_pos=100, exclude_whale_id=1, frame_shape=(100, 100, 3)
        )
        check(
            "other_whale_instance_lines returns a line for the other animal "
            "in the same frame (whale 2, excluding whale 1 itself)",
            len(lines_for_whale_1) == 1 and lines_for_whale_1[0].startswith("0 "),
        )
        check(
            "other_whale_instance_lines returns nothing for a frame "
            "with no other animals",
            other_whale_instance_lines(
                multi_whale_pool, frame_pos=200, exclude_whale_id=1, frame_shape=(100, 100, 3)
            ) == [],
        )
        check(
            "other_whale_instance_lines survives an empty or missing pool",
            other_whale_instance_lines(None, 100, 1, (100, 100, 3)) == [],
        )

        # Real regression (2026-09): the user must be able to correct the other
        # animal's outline by hand rather than being stuck with the raw
        # automatic mask. confirmed_polygons holds the corrected outline (or the
        # automatic one accepted as-is) and is used instead.
        confirmed_extra_polygon = [[45.0, 45.0], [55.0, 45.0], [55.0, 55.0], [45.0, 55.0]]
        lines_with_confirmed = other_whale_instance_lines(
            multi_whale_pool,
            frame_pos=100,
            exclude_whale_id=1,
            frame_shape=(100, 100, 3),
            confirmed_polygons={2: confirmed_extra_polygon},
        )
        check(
            "other_whale_instance_lines prefers the hand-confirmed polygon over "
            "the raw automatic mask when there is one",
            len(lines_with_confirmed) == 1
            and lines_with_confirmed[0] == "0 0.450000 0.450000 0.550000 0.450000 0.550000 0.550000 0.450000 0.550000",
        )

        # Real regression (2026-09): an animal added BY HAND during review
        # (never detected or accepted by screening, see start_manual_other_whale)
        # has no row in candidate_pool at all. It must still appear in the label
        # if its polygon is in confirmed_polygons.
        manual_whale_polygon = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
        lines_manual_only = other_whale_instance_lines(
            multi_whale_pool,
            frame_pos=100,
            exclude_whale_id=1,
            frame_shape=(100, 100, 3),
            confirmed_polygons={99: manual_whale_polygon},
        )
        check(
            "other_whale_instance_lines includes an animal added by hand "
            "(no candidate_pool row, only confirmed_polygons)",
            any("0.100000 0.100000" in line for line in lines_manual_only),
        )
        check(
            "other_whale_instance_lines survives a missing candidate_pool with "
            "only a hand-added animal",
            other_whale_instance_lines(
                None, 100, 1, (100, 100, 3), confirmed_polygons={99: manual_whale_polygon}
            )
            != [],
        )

        check(
            "reconstruct_mask_from_crop puts the crop back in the right place "
            "on the full-size canvas",
            reconstruct_mask_from_crop(other_whale_mask_crop, 40, 40, 100, 100)[55, 55] == 255,
        )
        check(
            "reconstruct_mask_from_crop returns an empty canvas for a missing or "
            "invalid crop, without failing",
            not reconstruct_mask_from_crop(None, np.nan, np.nan, 100, 100).any(),
        )

        landmark_csv = tmp_path / "landmarks.csv"
        # A deliberately sub-pixel point (10.5, 20.25): a placed or dragged
        # landmark is that accurate (see review.py and landmarks.py), and
        # make_landmark_record must not round it to whole pixels when writing
        # the training CSV; see the comment there for why.
        row = make_landmark_record("frame_0001.jpg", {"BH": (10.5, 20.25)}, "video_2026-07-01")
        # "BH" is the third bodypart in LANDMARK_EXPORT_ORDER (after S and F),
        # so its x/y are row[7] and row[8]: row[0:3] is metadata and the empty
        # S and F take row[3:7].
        check("make_landmark_record does not round sub-pixel coordinates", row[7:9] == [10.5, 20.25])
        write_landmark_csv({row[2]: row}, landmark_csv)
        loaded = load_existing_landmark_records(landmark_csv)
        check("landmark CSV round-trip", "frame_0001.jpg" in loaded)
        check(
            "landmark CSV round-trip keeps sub-pixel accuracy",
            loaded["frame_0001.jpg"][7:9] == ["10.5", "20.25"],
        )
        check("LANDMARK_DEFINITIONS has 11 points", len(LANDMARK_DEFINITIONS) == 11)

    print("\nAll checks passed (pure functions/synthetic data only).")
    print("Remember to still run the app on a real video before trusting it in the field.")


if __name__ == "__main__":
    main()
