"""Interactive review: the single panel, the mouse callback, mask correction.

`ReviewState` replaces the ~25 global variables the original file reset by
hand for every frame (see the reset block in `refine_mask_interactive`). That
reset now just builds a fresh instance, shared with `mouse_callback`,
`ReviewActionPanel` and `LandmarkControlPanel` through `bind_state()`.

`_review_action_panel` and `_landmark_panel` stay private module-level
singletons instead: they live longer, being created once and reused for every
frame, and are not part of the state that resets per frame.
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import device, quality
from .frame_source import open_capture, use_sequential_indexing
from .landmarks import LandmarkControlPanel, LANDMARK_LABELS
from .measurement import (
    measure_length_pca,
    bezier_curve_points,
    measure_widths_along_curve,
    clean_review_mask,
    create_manual_width_guides,
    draw_cross_marker,
    draw_readable_text,
    enhance_review_frame,
    expand_bbox,
    manual_measure_length_px,
    mask_to_polygons,
    polygon_epsilon_fraction,
    polygon_to_mask,
    sanitize_name,
    SUBPIXEL_SHIFT,
    subpixel_point,
)
from .settings_ui import ask_confirm, bind_vertical_scroll, get_shared_tk_root

_review_action_panel = None
_landmark_panel = None


@dataclass
class ReviewState:
    drawing_points: list = field(default_factory=list)
    drawing_labels: list = field(default_factory=list)
    current_view_roi: "tuple | None" = None
    last_mouse_pos: tuple = (0, 0)
    manual_measure_points: list = field(default_factory=list)
    # Whether the manual length follows the Bezier curve through the clicked
    # points rather than the straight polyline. Set from the panel checkbox and
    # kept here because mouse_callback and the key handlers need it too.
    bezier_length_mode: bool = True
    is_measure_mode: bool = False
    info_points: "tuple | None" = None
    mask_final: "np.ndarray | None" = None
    is_width_edit_mode: bool = False
    active_widths_cache: list = field(default_factory=list)
    dragged_pt_info: "dict | None" = None
    last_width_recalc_at: float = 0.0
    current_gsd: float = 0.0
    current_win_w: int = 1920
    current_win_h: int = 1080
    current_display_offset_x: int = 0
    current_display_offset_y: int = 0
    zoom_center_x: float = 0.0
    zoom_center_y: float = 0.0
    zoom_factor: float = 1.0
    img_w: int = 0
    img_h: int = 0
    sam_dirty: bool = True
    mask_correction_enabled: bool = True
    landmark_mode: bool = False
    active_landmark_code: "str | None" = None
    pending_landmark_point: "tuple | None" = None
    has_mouse_position: bool = False
    quality_mode: bool = False
    quality_index: int = 0
    quality_scores: list = field(default_factory=list)
    # A counter bumped every time mask_final is replaced by a new segmentation.
    # Used instead of id(mask_final) to invalidate the caches below: id() on a
    # numpy array is its memory address, and CPython often reuses the address of
    # a just-freed array of the same shape, so a new mask could share the old
    # one's id() and the cache would miss the change.
    mask_generation: int = 0
    dataset_polygon: "list | None" = None
    dataset_polygon_generation: "int | None" = None
    dataset_polygon_epsilon: "float | None" = None
    dragging_polygon_index: "int | None" = None
    mask_contour_cache_generation: "int | None" = None
    mask_contours_cache: list = field(default_factory=list)
    polygon_edit_mode: bool = False
    polygon_edit_confirmed: "bool | None" = None
    polygon_edit_skipped: bool = False
    # Drawing an outline from scratch, for a frame with no automatic mask at
    # all (excluded for altitude or tilt and added by hand for annotation, or
    # rejected by the mean-shape filter).
    #
    # polygon_drawing_from_scratch stays True from the moment the user chooses
    # to draw until the polygon is rasterized into a real mask
    # (apply_polygon_edit_to_mask). polygon_scratch_building separates "still
    # adding points" (a click adds a point) from "polygon closed, now refining
    # it" (a click drags or inserts, like the automatic polygon).
    polygon_drawing_from_scratch: bool = False
    polygon_scratch_building: bool = False
    custom_segment_mode: bool = False
    custom_segment_queue: list = field(default_factory=list)
    custom_segment_index: int = 0
    custom_segment_pending: list = field(default_factory=list)
    custom_segment_awaiting_confirm: bool = False
    custom_segments: list = field(default_factory=list)
    # The names of this frame's steps in the order they were actually visited.
    # The polygon and extra-segment steps are optional, so the list varies. The
    # "Previous phase" button uses it to know where to go back to.
    phase_history: list = field(default_factory=list)
    # Correcting the outlines of the OTHER whales in the same frame, not the
    # one under review. Holds the queue still to be offered (one at a time,
    # yes/no plus dragging as for dataset_polygon), the active one, and the
    # final result for each. Only used when exporting THIS animal's training
    # image, see other_whale_instance_lines in export.py.
    other_whale_queue: list = field(default_factory=list)
    other_whale_edit_mode: bool = False
    other_whale_confirmed: "bool | None" = None
    active_other_whale_id: "int | None" = None
    other_whale_polygon: "list | None" = None
    # The WHOLE automatic mask (not an already simplified polygon) of the whale
    # active in other_whale_edit_mode, plus the detail level the slider had when
    # other_whale_polygon was last extracted. Same mask/epsilon cache pair as
    # dataset_polygon_generation and epsilon above, so it follows the slider
    # like the main polygon does (see sync_other_whale_polygon).
    other_whale_mask: "np.ndarray | None" = None
    other_whale_polygon_epsilon: "float | None" = None
    dragging_other_whale_polygon_index: "int | None" = None
    other_whale_results: dict = field(default_factory=dict)
    # Drawing another whale's outline from scratch: the same idea as
    # polygon_drawing_from_scratch and polygon_scratch_building above, but for
    # an animal in other_whale_edit_mode with no automatic mask
    # (other_whale_mask is None). That happens when the automatic queue is
    # exhausted, the shape was rejected, or the animal was never detected in
    # this frame; see other_whale_manual_prompt_mode below.
    other_whale_drawing_from_scratch: bool = False
    other_whale_scratch_building: bool = False
    # Landmarks for the OTHER whales in the frame. Right after one animal's
    # outline (above), if automatic head and fluke points exist for THAT whale
    # (or the user places them by hand when they are missing, see
    # other_whale_landmark_manual_start in LandmarkControlPanel), the user is
    # offered the landmark step for it too, reusing the main animal's
    # LandmarkControlPanel. See doing_other_whale_landmarks for how the two uses
    # are told apart in refine_mask_interactive.
    other_whale_landmark_mode: bool = False
    other_whale_landmark_confirmed: "bool | None" = None
    other_whale_landmark_head: "tuple | None" = None
    other_whale_landmark_fluke: "tuple | None" = None
    other_whale_landmark_results: dict = field(default_factory=dict)
    doing_other_whale_landmarks: bool = False
    # After the automatic queue of other whales (or straight away if there was
    # none), ask whether there is still another animal in this frame that was
    # never detected (filtered out by screening, or missed entirely) to add by
    # hand, with its own ID, an outline drawn from scratch and landmarks.
    # other_whale_manual_added remembers the IDs already added by hand in THIS
    # frame, so they are not offered again if the user steps back with
    # "Previous phase".
    other_whale_manual_prompt_mode: bool = False
    other_whale_manual_confirmed: "bool | None" = None
    other_whale_manual_added: list = field(default_factory=list)


def increase_review_zoom(state):
    """Zoom in around the pointer, or around the centre if there is none."""
    if state.current_view_roi is not None:
        view_x, view_y, view_w, view_h = state.current_view_roi
        if state.has_mouse_position and state.current_win_w > 0 and state.current_win_h > 0:
            mouse_x, mouse_y = state.last_mouse_pos
            state.zoom_center_x = view_x + (mouse_x / state.current_win_w) * view_w
            state.zoom_center_y = view_y + (mouse_y / state.current_win_h) * view_h
        else:
            state.zoom_center_x = view_x + view_w / 2
            state.zoom_center_y = view_y + view_h / 2
    # A multiplying step rather than a fixed +0.5. With a ceiling of 100x a
    # fixed step would take about 200 presses to get there; 1.3x per press
    # covers 1x to 100x in about twenty, while staying fine-grained near 1x.
    state.zoom_factor = min(round(state.zoom_factor * 1.3, 1), 100.0)


def decrease_review_zoom(state):
    state.zoom_factor = max(round(state.zoom_factor / 1.3, 1), 1.0)
    if state.zoom_factor == 1.0:
        state.zoom_center_x, state.zoom_center_y = state.img_w / 2, state.img_h / 2


def reset_review_zoom(state):
    state.zoom_factor = 1.0
    state.zoom_center_x, state.zoom_center_y = state.img_w / 2, state.img_h / 2


def calculate_review_view_dimensions(state, canvas_width, canvas_height):
    """Fit the zoomed view to the panel, avoiding empty bands without
    stretching it."""
    base_width = max(1, int(state.img_w / state.zoom_factor))
    base_height = max(1, int(state.img_h / state.zoom_factor))
    if state.zoom_factor <= 1.0 or canvas_width <= 0 or canvas_height <= 0:
        return base_width, base_height

    canvas_aspect = canvas_width / canvas_height
    view_aspect = base_width / base_height
    if view_aspect > canvas_aspect:
        base_width = max(1, int(round(base_height * canvas_aspect)))
    elif view_aspect < canvas_aspect:
        base_height = max(1, int(round(base_width / canvas_aspect)))
    return min(base_width, state.img_w), min(base_height, state.img_h)


def measuring_line_points(state):
    """The line the measurements actually run along.

    The user clicks a few points, but with the Bezier box ticked the length is
    measured along the curve through them, not along the straight polyline. The
    widths have to follow the SAME curve, or "35% of the body" means two
    different places for length and for widths, and the width segments do not
    even sit on the curve drawn on screen.

    It deliberately uses the same number of samples as the length: measuring
    both on exactly the same points is the whole purpose of this function.
    """
    points = state.manual_measure_points
    if state.bezier_length_mode and len(points) >= 3:
        return bezier_curve_points(points).tolist()
    return points


def toggle_manual_measurement(state):
    """Turn on manual measurement, even when there is no mask."""
    state.is_measure_mode = not state.is_measure_mode
    state.is_width_edit_mode = False
    state.dragged_pt_info = None
    if state.is_measure_mode:
        if state.mask_final is not None and len(state.manual_measure_points) >= 3:
            state.active_widths_cache = measure_widths_along_curve(
                state.mask_final,
                measuring_line_points(state),
                state.current_gsd,
            )
        else:
            state.active_widths_cache = []
    else:
        state.active_widths_cache = (
            state.info_points[3] if state.info_points is not None else []
        )


def toggle_width_editing(state):
    """Turn on the automatic widths, or create manual guides when there is no
    mask."""
    if state.is_width_edit_mode:
        state.is_width_edit_mode = False
        state.dragged_pt_info = None
        return

    if state.is_measure_mode and state.mask_final is None:
        if len(state.manual_measure_points) < 3:
            if _review_action_panel is not None:
                _review_action_panel.status.set(
                    "First mark head, at least one intermediate point and fluke."
                )
            return
        if not state.active_widths_cache:
            state.active_widths_cache = create_manual_width_guides(
                measuring_line_points(state),
                state.current_gsd,
                state.img_w,
                state.img_h,
            )
    elif not state.active_widths_cache:
        if _review_action_panel is not None:
            _review_action_panel.status.set(
                "No widths available to edit."
            )
        return

    state.is_width_edit_mode = True
    state.dragged_pt_info = None
    if _review_action_panel is not None:
        _review_action_panel.status.set(
            "Drag the ends of the purple segments; press W to finish."
        )


def sync_dataset_polygon(state, review_action_panel, mask_final):
    """Rebuild state.dataset_polygon only when it really has to change.

    That means when the mask changed (a new segmentation, tracked by
    mask_generation) or the detail slider moved. On every other tick of the
    review loop the version already in memory is reused, since it may hold
    vertices the user moved by hand. Otherwise each redraw, about 50 times a
    second, would rerun findContours and approxPolyDP over the whole frame and
    wipe those edits out.

    Outside polygon_edit_mode this function does nothing at all: the step runs
    at the start of the review, before quality and landmarks, and the confirmed
    polygon has to survive until it is saved to the dataset at the end of the
    frame, long after the mode closed.
    """
    if not state.polygon_edit_mode:
        return
    if state.polygon_drawing_from_scratch:
        # Drawn point by point by the user, with no automatic mask to start
        # from (see enter_polygon_phase). There is nothing to regenerate:
        # mouse_callback builds dataset_polygon click by click, and rewriting it
        # here would erase it on every tick.
        return
    if mask_final is None or not state.polygon_edit_confirmed:
        state.dataset_polygon = None
        state.dataset_polygon_generation = None
        state.dataset_polygon_epsilon = None
        state.dragging_polygon_index = None
        return

    epsilon_fraction = review_action_panel.get_polygon_epsilon_fraction()
    if (
        state.dataset_polygon is None
        or state.dataset_polygon_generation != state.mask_generation
        or state.dataset_polygon_epsilon != epsilon_fraction
    ):
        # ponytail: only the largest polygon can be edited by hand. With
        # several separate outlines, rare for one whale, the rest stay exactly
        # as generated.
        polygons = mask_to_polygons(mask_final, epsilon_fraction)
        state.dataset_polygon = (
            [list(map(int, point)) for point in polygons[0]] if polygons else None
        )
        state.dataset_polygon_generation = state.mask_generation
        state.dataset_polygon_epsilon = epsilon_fraction
        state.dragging_polygon_index = None


def sync_other_whale_polygon(state, review_action_panel):
    """sync_dataset_polygon's logic, applied to ANOTHER whale's outline.

    It rebuilds state.other_whale_polygon from that animal's automatic mask
    (state.other_whale_mask, which is fixed for this animal and, unlike
    mask_final, never resegmented) whenever the detail slider changes. That way
    it follows the slider like the main animal's polygon, instead of staying
    stuck at the one simplification computed when the queue was built.
    """
    if not state.other_whale_edit_mode or not state.other_whale_confirmed:
        return
    if state.other_whale_drawing_from_scratch:
        # Drawn point by point by the user, with no automatic mask (see
        # start_manual_other_whale). There is nothing to regenerate:
        # mouse_callback builds other_whale_polygon click by click, and
        # rewriting it here would erase it on every tick.
        return
    if state.other_whale_mask is None:
        state.other_whale_polygon = None
        return
    epsilon_fraction = review_action_panel.get_polygon_epsilon_fraction()
    if (
        state.other_whale_polygon is None
        or state.other_whale_polygon_epsilon != epsilon_fraction
    ):
        polygons = mask_to_polygons(state.other_whale_mask, epsilon_fraction)
        state.other_whale_polygon = (
            [list(map(int, point)) for point in polygons[0]] if polygons else None
        )
        state.other_whale_polygon_epsilon = epsilon_fraction
        state.dragging_other_whale_polygon_index = None


def find_dataset_polygon_vertex(polygon, x, y, scale_x):
    """The polygon vertex nearest (x, y), within a click tolerance."""
    if not polygon:
        return None
    min_dist = max(10, 15 * scale_x)
    best_index = None
    for index, (point_x, point_y) in enumerate(polygon):
        distance = math.hypot(point_x - x, point_y - y)
        if distance < min_dist:
            min_dist = distance
            best_index = index
    return best_index


def find_dataset_polygon_edge_insertion_index(polygon, x, y, scale_x):
    """Where to insert a new vertex when (x, y) lands on one of the polygon's
    sides, including the closing one.

    Returns that insertion index, so detail can be added to a too-coarse side
    without first moving an existing vertex.
    """
    if not polygon or len(polygon) < 2:
        return None
    min_dist = max(10, 15 * scale_x)
    best_index = None
    point = np.array([x, y], dtype=np.float32)
    vertex_count = len(polygon)
    for index in range(vertex_count):
        start = np.array(polygon[index], dtype=np.float32)
        end = np.array(polygon[(index + 1) % vertex_count], dtype=np.float32)
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-6:
            continue
        t = float(np.clip(np.dot(point - start, segment) / segment_length_sq, 0.0, 1.0))
        projection = start + t * segment
        distance = float(np.hypot(*(point - projection)))
        if distance < min_dist:
            min_dist = distance
            best_index = index + 1
    return best_index


def get_mask_contours(state, mask_final):
    """Contours of the WHOLE mask for drawing, recomputed only when the mask
    changed since the last tick (same reasoning as sync_dataset_polygon).

    Always taken from the whole mask, never from a crop of the current view. A
    contour computed on a crop that cuts the shape in half (high zoom on part
    of the whale) would be closed off at the crop edge instead of continuing
    out of view, drawing a straight segment that does not exist in the real
    mask. Callers pass `offset` to cv2.drawContours to shift these contours (in
    image coordinates) onto the crop rather than recomputing them there.
    """
    if state.mask_contour_cache_generation != state.mask_generation:
        contours, _ = cv2.findContours(
            mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        state.mask_contours_cache = contours
        state.mask_contour_cache_generation = state.mask_generation
    return state.mask_contours_cache


def mouse_callback(event, x, y, flags, state):
    if state is None:
        return

    x -= state.current_display_offset_x
    y -= state.current_display_offset_y
    if not (0 <= x < state.current_win_w and 0 <= y < state.current_win_h):
        state.has_mouse_position = False
        if event == cv2.EVENT_LBUTTONUP:
            state.dragged_pt_info = None
        return
    state.has_mouse_position = True

    if flags & cv2.EVENT_FLAG_SHIFTKEY and event == cv2.EVENT_MOUSEMOVE:
        view_x, view_y, view_w, view_h = state.current_view_roi
        scale_x = view_w / state.current_win_w
        scale_y = view_h / state.current_win_h
        dx = x - state.last_mouse_pos[0]
        dy = y - state.last_mouse_pos[1]
        state.zoom_center_x -= dx * scale_x
        state.zoom_center_y -= dy * scale_y
        state.zoom_center_x = max(0, min(state.zoom_center_x, state.img_w))
        state.zoom_center_y = max(0, min(state.zoom_center_y, state.img_h))
        state.last_mouse_pos = (x, y)
        return

    state.last_mouse_pos = (x, y)
    if state.current_view_roi is None or state.current_win_w == 0 or state.current_win_h == 0:
        return

    view_x, view_y, view_w, view_h = state.current_view_roi
    scale_x = view_w / state.current_win_w
    scale_y = view_h / state.current_win_h
    # Sub-pixel, not int(): on screen one canvas pixel is often several real
    # pixels (a 4K video in a smaller window), so truncating here would make
    # dragged points (widths, length) jump several real pixels at a time instead
    # of following the mouse smoothly. Rounding happens later, only where it is
    # needed: drawing and indexing.
    real_x = view_x + x * scale_x
    real_y = view_y + y * scale_y

    if state.landmark_mode:
        if event == cv2.EVENT_LBUTTONDOWN and state.active_landmark_code is not None:
            state.pending_landmark_point = (real_x, real_y)
            if _landmark_panel is not None:
                _landmark_panel.set_pending_point(state.pending_landmark_point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.pending_landmark_point = None
            if _landmark_panel is not None and state.active_landmark_code is not None:
                _landmark_panel.status.set("Point cancelled: click again on the frame.")
        return

    if state.quality_mode:
        return

    if state.custom_segment_mode:
        if state.custom_segment_awaiting_confirm:
            # Both points are placed and waiting for the user to confirm them
            # (Enter/Space) or redo them (R); see the key handler. Clicks do
            # nothing here, so a third point cannot be dropped by accident while
            # still looking at the pair just drawn.
            return
        # Left click does nothing once the queue is empty (there is nothing
        # left to name), but right click to undo stays available: otherwise an
        # accidental double click on the last configured segment could not be
        # undone.
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and state.custom_segment_index < len(state.custom_segment_queue)
        ):
            state.custom_segment_pending.append((real_x, real_y))
            if len(state.custom_segment_pending) >= 2:
                # No longer confirmed automatically on the second click: it
                # waits until the user confirms explicitly, so they can still
                # look at it and undo before it becomes final.
                state.custom_segment_awaiting_confirm = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state.custom_segment_pending:
                state.custom_segment_pending.pop()
            elif state.custom_segments:
                state.custom_segments.pop()
                state.custom_segment_index -= 1
        return

    if state.polygon_edit_mode:
        if state.polygon_scratch_building:
            # No automatic mask to start from (see polygon_drawing_from_
            # scratch), so every click adds a new point rather than dragging an
            # existing one: there is no outline yet to drag or insert into.
            # Right click undoes the last point, the same as for extra segments.
            if event == cv2.EVENT_LBUTTONDOWN:
                if state.dataset_polygon is None:
                    state.dataset_polygon = []
                state.dataset_polygon.append([float(real_x), float(real_y)])
            elif event == cv2.EVENT_RBUTTONDOWN and state.dataset_polygon:
                state.dataset_polygon.pop()
            return
        if state.polygon_edit_confirmed and state.dataset_polygon:
            if event == cv2.EVENT_LBUTTONDOWN:
                vertex_index = find_dataset_polygon_vertex(
                    state.dataset_polygon, real_x, real_y, scale_x
                )
                if vertex_index is not None:
                    state.dragging_polygon_index = vertex_index
                else:
                    insertion_index = find_dataset_polygon_edge_insertion_index(
                        state.dataset_polygon, real_x, real_y, scale_x
                    )
                    if insertion_index is not None:
                        # Clicking a side rather than a vertex adds a point
                        # exactly there, immediately draggable, so a too-coarse
                        # side can gain detail without first moving an existing
                        # vertex.
                        #
                        # Float, not int(): like the width and length points, a
                        # vertex rounded to a whole pixel on every drag event
                        # would start jumping again (see the sub-pixel comment
                        # below).
                        state.dataset_polygon.insert(
                            insertion_index, [float(real_x), float(real_y)]
                        )
                        state.dragging_polygon_index = insertion_index
            elif event == cv2.EVENT_MOUSEMOVE and state.dragging_polygon_index is not None:
                state.dataset_polygon[state.dragging_polygon_index] = [
                    float(np.clip(real_x, 0, state.img_w - 1)),
                    float(np.clip(real_y, 0, state.img_h - 1)),
                ]
            elif event == cv2.EVENT_LBUTTONUP:
                state.dragging_polygon_index = None
        return

    if state.other_whale_edit_mode:
        # Exactly the logic of the block above (dataset_polygon), but pointed
        # at other_whale_polygon: dragging and inserting vertices on ANOTHER
        # whale's outline in the same frame, not the one under review.
        if state.other_whale_scratch_building:
            # No mask to start from (an animal added by hand, see
            # start_manual_other_whale): every click adds a point and right
            # click undoes the last, as in polygon_scratch_building above.
            if event == cv2.EVENT_LBUTTONDOWN:
                if state.other_whale_polygon is None:
                    state.other_whale_polygon = []
                state.other_whale_polygon.append([float(real_x), float(real_y)])
            elif event == cv2.EVENT_RBUTTONDOWN and state.other_whale_polygon:
                state.other_whale_polygon.pop()
            return
        if state.other_whale_confirmed and state.other_whale_polygon:
            if event == cv2.EVENT_LBUTTONDOWN:
                vertex_index = find_dataset_polygon_vertex(
                    state.other_whale_polygon, real_x, real_y, scale_x
                )
                if vertex_index is not None:
                    state.dragging_other_whale_polygon_index = vertex_index
                else:
                    insertion_index = find_dataset_polygon_edge_insertion_index(
                        state.other_whale_polygon, real_x, real_y, scale_x
                    )
                    if insertion_index is not None:
                        state.other_whale_polygon.insert(
                            insertion_index, [float(real_x), float(real_y)]
                        )
                        state.dragging_other_whale_polygon_index = insertion_index
            elif event == cv2.EVENT_MOUSEMOVE and state.dragging_other_whale_polygon_index is not None:
                state.other_whale_polygon[state.dragging_other_whale_polygon_index] = [
                    float(np.clip(real_x, 0, state.img_w - 1)),
                    float(np.clip(real_y, 0, state.img_h - 1)),
                ]
            elif event == cv2.EVENT_LBUTTONUP:
                state.dragging_other_whale_polygon_index = None
        return

    if state.is_width_edit_mode:
        # On macOS, switching between the Tk panel and the OpenCV window can
        # occasionally swallow LBUTTONDOWN. If the button is already held, grab
        # the point on the first move event instead.
        should_acquire_drag = (
            event == cv2.EVENT_LBUTTONDOWN
            or (
                event == cv2.EVENT_MOUSEMOVE
                and flags & cv2.EVENT_FLAG_LBUTTON
                and state.dragged_pt_info is None
            )
        )
        if should_acquire_drag:
            min_dist = max(10, 15 * scale_x)
            best_match = None
            for i, w_info in enumerate(state.active_widths_cache):
                for edge in ['pt_top', 'pt_bot']:
                    pt = w_info[edge]
                    dist = math.hypot(pt[0] - real_x, pt[1] - real_y)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = {
                            'type': 'width',
                            'idx': i, 'edge': edge,
                            'p1': w_info['pt_top'], 'p2': w_info['pt_bot']
                        }
            pts_to_check = []
            if state.is_measure_mode and len(state.manual_measure_points) >= 3:
                pts_to_check = [
                    ('head', state.manual_measure_points[0], state.manual_measure_points[1]),
                    ('fluke', state.manual_measure_points[-1], state.manual_measure_points[-2])
                ]
            elif not state.is_measure_mode and state.info_points is not None:
                pts_to_check = [
                    ('head', state.info_points[1], state.info_points[2]),
                    ('fluke', state.info_points[2], state.info_points[1])
                ]
            for edge, p1_pt, p2_pt in pts_to_check:
                dist = math.hypot(p1_pt[0] - real_x, p1_pt[1] - real_y)
                if dist < min_dist:
                    min_dist = dist
                    best_match = {
                        'type': 'length',
                        'edge': edge,
                        'p1': p1_pt,
                        'p2': p2_pt
                    }
            if best_match:
                state.dragged_pt_info = best_match

        if event == cv2.EVENT_MOUSEMOVE:
            if state.dragged_pt_info is not None:
                p1 = np.array(state.dragged_pt_info['p1'], dtype=np.float32)
                p2 = np.array(state.dragged_pt_info['p2'], dtype=np.float32)
                m = np.array([real_x, real_y], dtype=np.float32)
                v = p2 - p1
                sq_len = np.dot(v, v)
                if sq_len > 0:
                    t = np.dot(m - p1, v) / sq_len
                    proj = p1 + t * v
                    # Sub-pixel, not int(): a dragged point rounded to the
                    # nearest pixel on every move event makes the measurement
                    # jump, since one native pixel can be several centimetres.
                    # The value stays float until it is drawn.
                    new_pt = (float(proj[0]), float(proj[1]))
                    if state.dragged_pt_info['type'] == 'width':
                        idx = state.dragged_pt_info['idx']
                        edge = state.dragged_pt_info['edge']
                        state.active_widths_cache[idx][edge] = new_pt
                        new_p1 = state.active_widths_cache[idx]['pt_top']
                        new_p2 = state.active_widths_cache[idx]['pt_bot']
                        state.active_widths_cache[idx]['width_m'] = math.hypot(new_p2[0]-new_p1[0], new_p2[1]-new_p1[1]) * state.current_gsd
                    elif state.dragged_pt_info['type'] == 'length':
                        edge = state.dragged_pt_info['edge']
                        # measure_widths_along_curve walks the mask pixel by
                        # pixel (cast_ray, about 38 ray walks per call on a 4K
                        # frame). Recomputing it on every mouse move, often over
                        # 60 per second, was the main reason dragging head or
                        # fluke felt jerky. It is recomputed at most ~25 times a
                        # second here, and releasing the mouse below forces one
                        # exact final recomputation anyway.
                        now = time.monotonic()
                        allow_width_recalc = (now - state.last_width_recalc_at) >= 0.04
                        if allow_width_recalc:
                            state.last_width_recalc_at = now
                        if state.is_measure_mode:
                            if edge == 'head':
                                state.manual_measure_points[0] = new_pt
                            elif edge == 'fluke':
                                state.manual_measure_points[-1] = new_pt
                            if allow_width_recalc and len(state.manual_measure_points) >= 3:
                                if state.mask_final is not None:
                                    state.active_widths_cache = measure_widths_along_curve(state.mask_final, measuring_line_points(state), state.current_gsd)
                                else:
                                    state.active_widths_cache = create_manual_width_guides(
                                        measuring_line_points(state),
                                        state.current_gsd,
                                        state.img_w,
                                        state.img_h,
                                    )
                        else:
                            if state.info_points is not None:
                                lung_m, old_h, old_f, w_data = state.info_points
                                if edge == 'head':
                                    new_h = new_pt
                                    new_f = old_f
                                else:
                                    new_h = old_h
                                    new_f = new_pt
                                new_lung_m = math.hypot(new_f[0] - new_h[0], new_f[1] - new_h[1]) * state.current_gsd
                                new_w_data = w_data
                                if allow_width_recalc and state.mask_final is not None:
                                    new_w_data = measure_widths_along_curve(state.mask_final, [new_h, new_f], state.current_gsd)
                                state.info_points = (new_lung_m, new_h, new_f, new_w_data)
                                state.active_widths_cache = new_w_data

        elif event == cv2.EVENT_LBUTTONUP:
            if state.dragged_pt_info is not None and state.dragged_pt_info['type'] == 'length' and state.mask_final is not None:
                # The final recomputation is not throttled: the widths shown
                # must match the exact release position, even if the last mouse
                # move fell inside the throttle window above.
                if state.is_measure_mode and len(state.manual_measure_points) >= 3:
                    state.active_widths_cache = measure_widths_along_curve(state.mask_final, measuring_line_points(state), state.current_gsd)
                elif not state.is_measure_mode and state.info_points is not None:
                    lung_m, h, f, _ = state.info_points
                    new_w_data = measure_widths_along_curve(state.mask_final, [h, f], state.current_gsd)
                    state.info_points = (lung_m, h, f, new_w_data)
                    state.active_widths_cache = new_w_data
            state.dragged_pt_info = None

    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            if state.is_measure_mode:
                state.manual_measure_points.append((real_x, real_y))
                if state.mask_final is not None and len(state.manual_measure_points) >= 3:
                    state.active_widths_cache = measure_widths_along_curve(state.mask_final, measuring_line_points(state), state.current_gsd)
                elif state.mask_final is None:
                    state.active_widths_cache = []
            else:
                if not state.mask_correction_enabled:
                    return
                state.drawing_points.append([real_x, real_y])
                state.drawing_labels.append(1)
                state.sam_dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state.is_measure_mode:
                if len(state.manual_measure_points) > 0:
                    state.manual_measure_points.pop()
                if state.mask_final is not None and len(state.manual_measure_points) >= 3:
                    state.active_widths_cache = measure_widths_along_curve(state.mask_final, measuring_line_points(state), state.current_gsd)
                else:
                    state.active_widths_cache = []
            else:
                if not state.mask_correction_enabled:
                    return
                state.drawing_points.append([real_x, real_y])
                state.drawing_labels.append(0)
                state.sam_dirty = True


def copy_automatic_measurement(info_points_value):
    """Make an independent copy of the automatic measurement, for validation."""
    if info_points_value is None:
        return None
    length_m, head_point, fluke_point, widths = info_points_value
    frozen_widths = {}
    for width_info in widths:
        pct = int(width_info["pct"])
        frozen_widths[pct] = {
            "width_m": float(width_info["width_m"]),
            # Float, not int(): this copy feeds the automatic-versus-human
            # comparison in the final CSV. Rounding the automatic side while the
            # human side stays exact would bias the comparison with noise on one
            # side only.
            "pt_top": (float(width_info["pt_top"][0]), float(width_info["pt_top"][1])),
            "pt_bot": (float(width_info["pt_bot"][0]), float(width_info["pt_bot"][1])),
        }
    return {
        "length_m": float(length_m),
        "head": (float(head_point[0]), float(head_point[1])),
        "fluke": (float(fluke_point[0]), float(fluke_point[1])),
        "widths": frozen_widths,
    }


class ReviewActionPanel:
    """Pannello unico per immagine, azioni, zoom e landmark."""

    def __init__(self, config):
        import _tkinter
        import tkinter as tk
        from tkinter import ttk

        self._tkinter = _tkinter
        self.tk = tk
        self._config = config
        self._state = None
        self._landmark_panel = None
        self.root = tk.Toplevel(get_shared_tk_root())
        self.root.title("Whale Vision — Review")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1500, max(900, screen_width - 60))
        window_height = min(900, max(620, screen_height - 100))
        self.root.geometry(f"{window_width}x{window_height}+20+40")
        self.root.minsize(
            min(1050, window_width),
            min(650, window_height),
        )
        self.root.protocol("WM_DELETE_WINDOW", self.keep_open)
        self.settings_requested = False
        self.slider_dragging = False
        self.slider_release_counter = 0
        self.pending_keys = []
        self.dirty_external = True
        self.image_photo = None
        self.contrast = tk.IntVar(master=self.root, value=100)
        self.brightness = tk.IntVar(master=self.root, value=0)
        self.clahe = tk.BooleanVar(master=self.root, value=False)
        self.show_enhanced = tk.BooleanVar(master=self.root, value=False)
        self.mask_opacity = tk.IntVar(master=self.root, value=20)
        self.hide_geometry = tk.BooleanVar(master=self.root, value=False)
        self.bezier_length = tk.BooleanVar(master=self.root, value=True)
        # 7 gives markerSize=14, the same size as the cross used for the point
        # awaiting confirmation while placing landmarks.
        self.marker_size = tk.IntVar(master=self.root, value=7)
        self.marker_thickness = tk.IntVar(master=self.root, value=1)
        self.inference_quality = tk.IntVar(
            master=self.root,
            value=config.review_imgsz,
        )
        self.polygon_complexity = tk.IntVar(master=self.root, value=99)
        self.skip_dataset_save = tk.BooleanVar(master=self.root, value=False)
        self._measure_controls_visible = True

        main = tk.PanedWindow(
            self.root,
            orient="horizontal",
            sashwidth=9,
            sashrelief="raised",
            sashpad=2,
            showhandle=True,
            handlesize=14,
            handlepad=32,
            sashcursor="sb_h_double_arrow",
            background="#555555",
            borderwidth=0,
            opaqueresize=True,
        )
        main.pack(fill="both", expand=True)
        self.main_pane = main
        self.sash_initialized = False

        image_host = ttk.Frame(main, padding=(10, 10, 5, 10))
        controls_host = ttk.Frame(main, width=440, padding=(5, 10, 10, 10))
        main.add(image_host, minsize=420, stretch="always")
        main.add(controls_host, minsize=300, width=440, stretch="never")

        self.image_canvas = tk.Canvas(
            image_host,
            background="#101010",
            highlightthickness=1,
            highlightbackground="#444444",
            cursor="crosshair",
        )
        self.image_canvas.pack(fill="both", expand=True)
        self.image_item = self.image_canvas.create_image(
            0,
            0,
            anchor="nw",
        )
        self._bind_image_events()

        self.canvas = tk.Canvas(
            controls_host,
            width=420,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            controls_host,
            orient="vertical",
            command=self.canvas.yview,
        )
        container = ttk.Frame(self.canvas, padding=14)
        container.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        container_window = self.canvas.create_window(
            (0, 0),
            window=container,
            anchor="nw",
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(
                container_window,
                width=event.width,
            ),
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_vertical_scroll(self.root, self.canvas)
        self.status = tk.StringVar(master=self.root, value="")
        self.accepted_counter_text = tk.StringVar(master=self.root, value="")
        ttk.Label(
            container,
            text="↔ Drag the left edge to resize the menu",
            foreground="#666666",
        ).pack(anchor="w", pady=(0, 6))
        # Kept separate from the status line, which changes constantly ("running
        # inference..."), so this stays visible for the whole review.
        ttk.Label(
            container,
            textvariable=self.accepted_counter_text,
            font=("", 11, "bold"),
            foreground="#1a7a3c",
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            container,
            textvariable=self.status,
            font=("", 11, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        image_controls = ttk.LabelFrame(
            container,
            text="Image and overlays",
            padding=10,
        )
        image_controls.pack(fill="x")
        contrast_scale = tk.Scale(
            image_controls,
            label="Contrast (%)",
            variable=self.contrast,
            from_=50,
            to=200,
            orient="horizontal",
            resolution=1,
        )
        contrast_scale.pack(fill="x")
        self.register_slider(contrast_scale)
        brightness_scale = tk.Scale(
            image_controls,
            label="Brightness",
            variable=self.brightness,
            from_=-100,
            to=100,
            orient="horizontal",
            resolution=1,
        )
        brightness_scale.pack(fill="x")
        self.register_slider(brightness_scale)
        opacity_scale = tk.Scale(
            image_controls,
            label="Mask opacity (%)",
            variable=self.mask_opacity,
            from_=0,
            to=80,
            orient="horizontal",
            resolution=1,
        )
        opacity_scale.pack(fill="x")
        self.register_slider(opacity_scale, affects_inference=False)
        quality_scale = tk.Scale(
            image_controls,
            label="Review inference quality (px)",
            variable=self.inference_quality,
            from_=320,
            to=2048,
            orient="horizontal",
            resolution=32,
        )
        quality_scale.pack(fill="x")
        self.register_slider(quality_scale)
        option_row = ttk.Frame(image_controls)
        option_row.pack(fill="x", pady=(5, 0))
        ttk.Checkbutton(
            option_row,
            text="CLAHE",
            variable=self.clahe,
        ).pack(side="left")
        ttk.Checkbutton(
            option_row,
            text="Show enhanced image",
            variable=self.show_enhanced,
        ).pack(side="left", padx=(14, 0))
        ttk.Checkbutton(
            image_controls,
            text="Hide mask and length/width lines",
            variable=self.hide_geometry,
        ).pack(anchor="w", pady=(7, 0))
        marker_size_scale = tk.Scale(
            image_controls,
            # The same cross is used everywhere (widths, length, landmarks,
            # dataset polygon, SAM points, extra segments), so one slider sets
            # its arm length in pixels. The ceiling is 50, up from 15: at high
            # zoom a small cross can disappear into the mask detail.
            label="Cross marker length (px)",
            variable=self.marker_size,
            from_=2,
            to=50,
            orient="horizontal",
            resolution=1,
        )
        marker_size_scale.pack(fill="x")
        self.register_slider(marker_size_scale, affects_inference=False)
        marker_thickness_scale = tk.Scale(
            image_controls,
            # A wider scale (was 1-6) so the steps feel gradual: going from 1 to
            # 2px doubles the stroke, while 10 to 11px is barely noticeable.
            label="Cross marker thickness (px)",
            variable=self.marker_thickness,
            from_=1,
            to=15,
            orient="horizontal",
            resolution=1,
        )
        marker_thickness_scale.pack(fill="x")
        self.register_slider(marker_thickness_scale, affects_inference=False)
        polygon_scale = tk.Scale(
            image_controls,
            label="Dataset polygon complexity (1-100, used in the dedicated phase)",
            variable=self.polygon_complexity,
            from_=1,
            to=100,
            orient="horizontal",
            resolution=1,
        )
        polygon_scale.pack(fill="x")
        self.register_slider(polygon_scale, affects_inference=False)
        ttk.Checkbutton(
            image_controls,
            text="Don't save this mask in the YOLO-Seg dataset",
            variable=self.skip_dataset_save,
        ).pack(anchor="w", pady=(4, 0))
        reset_button = ttk.Button(
            image_controls,
            text="Reset adjustments",
            command=self.reset_image_controls,
        )
        reset_button.pack(fill="x", pady=(8, 0))
        self.bind_immediate(reset_button)

        review_controls = ttk.Frame(container)
        review_controls.pack(fill="x", pady=(10, 0))
        self.review_controls = review_controls
        manual_button = ttk.Button(
            review_controls,
            text="Manual measurement",
            command=lambda: toggle_manual_measurement(self._state),
        )
        manual_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.bind_immediate(manual_button)
        self.manual_button = manual_button
        widths_button = ttk.Button(
            review_controls,
            text="Edit widths (W)",
            command=lambda: toggle_width_editing(self._state),
        )
        widths_button.pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.bind_immediate(widths_button)
        self.widths_button = widths_button
        ttk.Checkbutton(
            container,
            text="Manual length as a Bezier curve (instead of the polyline)",
            variable=self.bezier_length,
        ).pack(anchor="w", pady=(4, 0))
        configured_segments = [
            name.strip() for name in config.custom_segment_names.split(",") if name.strip()
        ]
        if configured_segments:
            ttk.Label(
                container,
                text=(
                    "Configured extra segments (asked after the measurement): "
                    + ", ".join(configured_segments)
                ),
                foreground="#666666",
                wraplength=400,
            ).pack(anchor="w", pady=(6, 0))

        zoom_controls = ttk.Frame(container)
        zoom_controls.pack(fill="x", pady=(6, 0))
        self.zoom_controls = zoom_controls
        ttk.Button(
            zoom_controls,
            text="Zoom −",
            command=self.zoom_out,
        ).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(
            zoom_controls,
            text="Zoom +",
            command=self.zoom_in,
        ).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(
            zoom_controls,
            text="Zoom 1×",
            command=self.zoom_reset,
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))
        ttk.Label(
            container,
            text="Scroll over the image to zoom (centered on the cursor), hold Shift and move to pan",
            foreground="#666666",
        ).pack(anchor="w", pady=(2, 0))

        # Always visible, in every step (polygon, quality, measurement, extra
        # segments, landmarks). It accepts the step's current progress and moves
        # on, except in the landmark step, the last one, where it discards the
        # frame and moves to the next: there is no following step to skip to.
        phase_nav = ttk.Frame(container)
        phase_nav.pack(fill="x", pady=(6, 0))
        previous_phase_button = ttk.Button(
            phase_nav,
            text="← Previous phase (P)",
            command=lambda: self.queue_key(ord("p")),
        )
        previous_phase_button.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.bind_immediate(previous_phase_button)
        skip_phase_button = ttk.Button(
            phase_nav,
            text="Skip phase / next →  (K)",
            command=lambda: self.queue_key(ord("k")),
        )
        skip_phase_button.pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.bind_immediate(skip_phase_button)

        frame_actions = ttk.LabelFrame(
            container,
            text="Frame actions",
            padding=8,
        )
        frame_actions.pack(fill="x", pady=(10, 0))
        self.frame_actions = frame_actions
        confirm_button = ttk.Button(
            frame_actions,
            text="Confirm mask and continue",
            command=lambda: self.queue_key(13),
        )
        confirm_button.pack(fill="x")
        self.bind_immediate(confirm_button)
        self.confirm_measurements_button = confirm_button
        secondary_actions = ttk.Frame(frame_actions)
        secondary_actions.pack(fill="x", pady=(6, 0))
        undo_button = ttk.Button(
            secondary_actions,
            text="Undo last point",
            command=lambda: self.queue_key(ord("z")),
        )
        undo_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 3),
        )
        self.bind_immediate(undo_button)
        discard_button = ttk.Button(
            secondary_actions,
            text="Discard frame",
            command=lambda: self.queue_key(ord("x")),
        )
        discard_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(3, 0),
        )
        self.bind_immediate(discard_button)

        stop_button = ttk.Button(
            container,
            text="Save progress and return to settings",
            command=self.request_settings,
        )
        stop_button.pack(fill="x", pady=(10, 0))
        self.bind_immediate(stop_button)
        self.stop_button = stop_button
        self.landmark_host = ttk.Frame(container)
        self.landmark_host.pack(fill="both", expand=True, pady=(8, 0))
        for key_sequence in ("<plus>", "<equal>", "<KP_Add>"):
            self.root.bind(key_sequence, self.zoom_in)
        for key_sequence in ("<minus>", "<underscore>", "<KP_Subtract>"):
            self.root.bind(key_sequence, self.zoom_out)
        for key_sequence in ("<KeyPress-a>", "<KeyPress-A>"):
            self.root.bind(key_sequence, self.confirm_landmark_shortcut)
        for key_sequence in ("<KeyPress-s>", "<KeyPress-S>"):
            self.root.bind(
                key_sequence,
                lambda _event: self.request_settings(),
            )
        for key_sequence, key_code in (
            ("<space>", 32),
            ("<Return>", 13),
            ("<KeyPress-x>", ord("x")),
            ("<KeyPress-X>", ord("x")),
            ("<KeyPress-r>", ord("r")),
            ("<KeyPress-R>", ord("r")),
            ("<KeyPress-z>", ord("z")),
            ("<KeyPress-Z>", ord("z")),
            ("<KeyPress-h>", ord("h")),
            ("<KeyPress-H>", ord("h")),
            ("<KeyPress-m>", ord("m")),
            ("<KeyPress-M>", ord("m")),
            ("<KeyPress-w>", ord("w")),
            ("<KeyPress-W>", ord("w")),
            ("<KeyPress-n>", ord("n")),
            ("<KeyPress-N>", ord("n")),
            ("<KeyPress-y>", ord("y")),
            ("<KeyPress-Y>", ord("y")),
            ("<KeyPress-k>", ord("k")),
            ("<KeyPress-K>", ord("k")),
            ("<KeyPress-p>", ord("p")),
            ("<KeyPress-P>", ord("p")),
            ("<KeyPress-1>", ord("1")),
            ("<KeyPress-2>", ord("2")),
            ("<KeyPress-3>", ord("3")),
        ):
            self.root.bind(
                key_sequence,
                lambda _event, queued_key=key_code: self.queue_key(
                    queued_key
                ),
            )
        self.root.withdraw()

    def bind_state(self, state):
        """Attach the current frame's ReviewState (called once per frame)."""
        self._state = state

    def set_landmark_panel(self, panel):
        self._landmark_panel = panel

    def _bind_image_events(self):
        """Forward the Canvas events to the existing review logic."""
        def send(cv_event, event, extra_flags=0):
            flags = extra_flags
            if getattr(event, "state", 0) & 0x0001:
                flags |= cv2.EVENT_FLAG_SHIFTKEY
            mouse_callback(cv_event, event.x, event.y, flags, self._state)
            self.image_canvas.focus_set()
            return "break"

        self.image_canvas.bind(
            "<ButtonPress-1>",
            lambda event: send(
                cv2.EVENT_LBUTTONDOWN,
                event,
                cv2.EVENT_FLAG_LBUTTON,
            ),
        )
        self.image_canvas.bind(
            "<B1-Motion>",
            lambda event: send(
                cv2.EVENT_MOUSEMOVE,
                event,
                cv2.EVENT_FLAG_LBUTTON,
            ),
        )
        self.image_canvas.bind(
            "<ButtonRelease-1>",
            lambda event: send(cv2.EVENT_LBUTTONUP, event),
        )
        self.image_canvas.bind(
            "<Motion>",
            lambda event: send(cv2.EVENT_MOUSEMOVE, event),
        )
        self.image_canvas.bind(
            "<ButtonPress-2>",
            lambda event: send(cv2.EVENT_RBUTTONDOWN, event),
        )
        self.image_canvas.bind(
            "<ButtonPress-3>",
            lambda event: send(cv2.EVENT_RBUTTONDOWN, event),
        )
        self.image_canvas.bind(
            "<Control-ButtonPress-1>",
            lambda event: send(cv2.EVENT_RBUTTONDOWN, event),
        )

        def zoom_wheel(event):
            # Smooth wheel zoom, rather than only the buttons and keys with
            # their 1.3x step. last_mouse_pos is synced to the cursor's current
            # position first (the same normalization mouse_callback uses), so the
            # zoom stays centred on the pointer even if the mouse has not moved
            # since the last event.
            direction = 0
            if getattr(event, "num", None) == 4:
                direction = 1
            elif getattr(event, "num", None) == 5:
                direction = -1
            elif getattr(event, "delta", 0):
                direction = 1 if event.delta > 0 else -1
            if direction == 0:
                return "break"
            send(cv2.EVENT_MOUSEMOVE, event)
            if direction > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return "break"

        self.image_canvas.bind("<MouseWheel>", zoom_wheel)
        self.image_canvas.bind("<Button-4>", zoom_wheel)
        self.image_canvas.bind("<Button-5>", zoom_wheel)

    def present_image(self, bgr_image, offset_x=0, offset_y=0):
        """Show the OpenCV frame in the Canvas, without opening any window."""
        from PIL import Image, ImageTk

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        if (
            self.image_photo is None
            or self.image_photo.width() != pil_image.width
            or self.image_photo.height() != pil_image.height
        ):
            self.image_photo = ImageTk.PhotoImage(
                image=pil_image,
                master=self.root,
            )
            self.image_canvas.itemconfigure(
                self.image_item,
                image=self.image_photo,
            )
        else:
            self.image_photo.paste(pil_image)
        self.image_canvas.coords(
            self.image_item,
            int(offset_x),
            int(offset_y),
        )

    def get_image_size(self):
        width = self.image_canvas.winfo_width() - 2
        height = self.image_canvas.winfo_height() - 2
        if width <= 1 or height <= 1:
            return 960, 700
        return width, height

    def get_fitted_image_geometry(self, source_width, source_height):
        """Return the size and position that fit without stretching the frame."""
        canvas_width, canvas_height = self.get_image_size()
        if source_width <= 0 or source_height <= 0:
            return canvas_width, canvas_height, 0, 0
        scale = min(
            canvas_width / source_width,
            canvas_height / source_height,
        )
        display_width = max(1, int(round(source_width * scale)))
        display_height = max(1, int(round(source_height * scale)))
        offset_x = max(0, (canvas_width - display_width) // 2)
        offset_y = max(0, (canvas_height - display_height) // 2)
        return display_width, display_height, offset_x, offset_y

    def place_initial_sash(self):
        if self.sash_initialized:
            return
        try:
            panel_width = min(440, max(300, self.root.winfo_width() // 3))
            self.main_pane.sash_place(
                0,
                self.root.winfo_width() - panel_width,
                0,
            )
            self.sash_initialized = True
        except self.tk.TclError:
            pass

    def clear_image(self):
        self.image_photo = None
        self.image_canvas.itemconfigure(self.image_item, image="")

    def queue_key(self, key_code):
        self.pending_keys.append(key_code)
        self.image_canvas.focus_set()
        self.mark_dirty()
        return "break"

    def pop_key(self):
        if not self.pending_keys:
            return 255
        return self.pending_keys.pop(0)

    def mark_dirty(self):
        """Record that something visible changed outside the review loop:
        zoom, landmarks, keys.

        Tkinter bindings such as zoom_in write to the state directly, and
        without this the loop would not notice, leaving the view on the old
        frame.
        """
        self.dirty_external = True

    def pop_dirty(self):
        was_dirty = self.dirty_external
        self.dirty_external = False
        return was_dirty

    def start(self, count, total, accepted_count=None, max_accepted=None, subject_label=None):
        self.settings_requested = False
        self.pending_keys.clear()
        self.set_measurement_controls_locked(False)
        self.skip_dataset_save.set(False)
        # A per-frame choice, not a sticky setting: every new frame goes back
        # to contributing to the dataset, and exclusion is decided each time.
        self._measure_controls_visible = True
        self.review_controls.pack(fill="x", pady=(10, 0), before=self.zoom_controls)
        self.frame_actions.pack(fill="x", pady=(10, 0), before=self.stop_button)
        self.status.set(f"Reviewing frame {count}/{total}")
        if accepted_count is not None and max_accepted is not None:
            self.accepted_counter_text.set(f"Accepted: {accepted_count}/{max_accepted}")
        self.root.title(
            f"Whale Vision — Review {count}/{total}"
            + (f" — {subject_label}" if subject_label else "")
        )
        self.root.deiconify()
        self.root.lift()
        self.canvas.yview_moveto(0.0)
        self.root.after_idle(self.place_initial_sash)

    def request_settings(self):
        if self.settings_requested:
            return "break"
        confirmed = ask_confirm(
            self.root,
            "Confirm early stop",
            (
                "Do you really want to stop the review?\n\n"
                "Frames already confirmed will be saved and you will return "
                "to settings. The current frame, if not yet confirmed, "
                "will not be saved."
            ),
            yes_label="Yes, stop and return to settings",
            no_label="No, continue the review",
        )
        if not confirmed:
            self.status.set("Stop cancelled: the review continues.")
            self.image_canvas.focus_set()
            return "break"
        self.settings_requested = True
        self.status.set("Saving and returning to settings…")
        return "break"

    def zoom_in(self, _event=None):
        increase_review_zoom(self._state)
        self.mark_dirty()
        return "break"

    def zoom_out(self, _event=None):
        decrease_review_zoom(self._state)
        self.mark_dirty()
        return "break"

    def zoom_reset(self, _event=None):
        reset_review_zoom(self._state)
        self.mark_dirty()
        return "break"

    def confirm_landmark_shortcut(self, _event=None):
        if (
            self._state is not None
            and self._state.landmark_mode
            and self._landmark_panel is not None
        ):
            self._landmark_panel.confirm_and_advance()
            self.mark_dirty()
            return "break"
        return None

    def reset_image_controls(self):
        self.contrast.set(100)
        self.brightness.set(0)
        self.clahe.set(False)
        self.show_enhanced.set(False)
        self.mask_opacity.set(20)
        self.hide_geometry.set(False)
        self.inference_quality.set(self._config.review_imgsz)
        self.marker_size.set(7)
        self.marker_thickness.set(1)

    def get_control_values(self):
        return (
            int(self.contrast.get()),
            int(self.brightness.get()),
            bool(self.clahe.get()),
            bool(self.show_enhanced.get()),
            int(self.mask_opacity.get()),
            bool(self.hide_geometry.get()),
            int(self.inference_quality.get()),
            int(self.marker_size.get()),
            bool(self.bezier_length.get()),
            int(self.marker_thickness.get()),
        )

    def get_polygon_epsilon_fraction(self):
        return polygon_epsilon_fraction(int(self.polygon_complexity.get()))

    def register_slider(self, slider, affects_inference=True):
        slider.bind("<ButtonPress-1>", self.slider_pressed)
        slider.bind(
            "<ButtonRelease-1>",
            lambda event: self.slider_released(
                event,
                affects_inference,
            ),
        )

    def slider_pressed(self, _event=None):
        self.slider_dragging = True

    def slider_released(self, _event=None, affects_inference=True):
        self.slider_dragging = False
        if affects_inference:
            self.slider_release_counter += 1

    @staticmethod
    def bind_immediate(button):
        def invoke_on_press(_event):
            button.invoke()
            return "break"

        button.bind("<ButtonPress-1>", invoke_on_press)

    def show_landmarks(self):
        self.set_measurement_controls_locked(True)
        self.status.set(
            "Measurements locked — adjustments only change the display."
        )
        self.root.lift()
        self.root.after_idle(self.scroll_to_landmarks)

    def scroll_to_landmarks(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def hide_landmarks(self):
        self.set_measurement_controls_locked(False)
        self.canvas.yview_moveto(0.0)

    def set_measurement_controls_locked(self, locked):
        state = ["disabled"] if locked else ["!disabled"]
        for button in (
            self.manual_button,
            self.widths_button,
            self.confirm_measurements_button,
        ):
            button.state(state)

    def keep_open(self):
        self.request_settings()

    def hide(self):
        # Leave fullscreen before hiding the window. Withdrawn while still in
        # native fullscreen (its own space on macOS), the next window (the
        # altitude dialog, say) can end up stuck in that space instead of
        # appearing. This is the likely cause of the display problems at the end
        # of a review with fullscreen on.
        try:
            self.root.attributes("-fullscreen", False)
        except self.tk.TclError:
            pass
        self.root.withdraw()

    def _sync_phase_visibility(self):
        """Show only the controls that mean something in the current step.

        The "Frame actions" block (confirm, undo, discard) and the manual
        measurement and width buttons only apply during the measurement step.
        In the other steps (dataset polygon, quality, extra segments,
        landmarks), each with its own on-screen instructions or panel, they used
        to sit there visible but inert, just clutter.
        """
        state = self._state
        if state is None:
            return
        in_measure_phase = not (
            state.polygon_edit_mode
            or state.other_whale_edit_mode
            or state.other_whale_landmark_mode
            or state.other_whale_manual_prompt_mode
            or state.quality_mode
            or state.custom_segment_mode
            or state.landmark_mode
        )
        if in_measure_phase == self._measure_controls_visible:
            return
        self._measure_controls_visible = in_measure_phase
        if in_measure_phase:
            self.review_controls.pack(fill="x", pady=(10, 0), before=self.zoom_controls)
            self.frame_actions.pack(fill="x", pady=(10, 0), before=self.stop_button)
        else:
            self.review_controls.pack_forget()
            self.frame_actions.pack_forget()

    def update(self):
        """Process Tk events without the reentrant update() call."""
        self._sync_phase_visibility()
        try:
            for _event_index in range(24):
                event_processed = self.root.tk.dooneevent(
                    self._tkinter.DONT_WAIT
                )
                if not event_processed:
                    break
        except self.tk.TclError:
            self.settings_requested = True


def refine_mask_interactive(
    frame,
    bbox,
    sam_model,
    count,
    total,
    gsd,
    config,
    initial_mask=None,
    mask_predictor=None,
    bbox_predictor=None,
    initial_mask_attempted=False,
    accepted_count=None,
    max_accepted=None,
    subject_label=None,
    other_whale_queue_source=None,
    next_manual_whale_id=2,
):
    global _review_action_panel, _landmark_panel

    measure_in_meters = config.measurement_unit == "Meters"

    state = ReviewState()
    if _landmark_panel is not None:
        _landmark_panel.hide()
    base_final_package = None
    locked_mask = None
    automatic_measurement_snapshot = None
    # The quality step (Christiansen) runs before the measurement, so these two
    # hold its verdict and scores until finalize_measurement builds
    # base_final_package, by which time the quality step is long closed (see
    # finalize_quality_and_proceed).
    quality_verdict = "passed"
    quality_scores_by_key = {}
    current_bbox = np.asarray(
        bbox,
        dtype=np.float32,
    ).reshape(-1, 4).copy()
    state.current_gsd = gsd
    state.mask_correction_enabled = sam_model is not None
    resegment_enabled = state.mask_correction_enabled or mask_predictor is not None
    state.sam_dirty = state.mask_correction_enabled or (
        initial_mask is None
        and mask_predictor is not None
        and not initial_mask_attempted
    )
    show_help = True

    state.img_h, state.img_w = frame.shape[:2]
    state.zoom_factor = 1.0
    state.zoom_center_x, state.zoom_center_y = state.img_w / 2, state.img_h / 2
    state.has_mouse_position = False
    state.current_display_offset_x = 0
    state.current_display_offset_y = 0

    def close_review_windows():
        if _review_action_panel is not None:
            _review_action_panel.clear_image()

    def confirm_discard():
        """Ask before discarding a frame.

        X sits next to Z (undo point) and S (settings) and gets pressed in a
        hurry. Without a confirmation, one wrong key throws away the frame's
        mask and measurement with no way to get them back.
        """
        return ask_confirm(
            review_action_panel.root,
            "Confirm discard",
            "Discard this frame? The mask and any measurement done on it "
            "so far will be lost.",
            yes_label="Yes, discard the frame",
            no_label="No, keep working on it",
        )

    def start_landmark_phase():
        """The last step for this frame: after quality, extra segments and the
        dataset polygon (whichever apply), it is the landmarks' turn."""
        global _landmark_panel
        if _landmark_panel is None:
            _landmark_panel = LandmarkControlPanel(
                review_action_panel.landmark_host,
                review_action_panel=review_action_panel,
            )
            review_action_panel.set_landmark_panel(_landmark_panel)
        _landmark_panel.bind_state(state)
        # finalize_measurement always needs a non-empty spine to get this far,
        # but a (0, 0) fallback costs nothing and avoids an IndexError if that
        # ever changes.
        spine = base_final_package[1]
        head_point = spine[0] if spine else (0, 0)
        fluke_point = spine[-1] if spine else (0, 0)
        _landmark_panel.start(head_point, fluke_point)
        state.landmark_mode = True
        # This should already be False by now, since the other-animals step
        # resets it itself when taken. Getting it wrong would send the MAIN
        # animal's landmark confirmation down the other-animals branch later in
        # the loop, silently losing this measurement.
        state.doing_other_whale_landmarks = False
        state.phase_history.append("landmark")

    def enter_custom_segments_phase():
        """Start the extra-segments step, or restart it from the Previous phase
        button. Always from index zero: segments defined on an earlier pass
        through this step are not kept."""
        state.custom_segment_mode = True
        state.custom_segment_queue = custom_segment_names
        state.custom_segment_index = 0
        state.custom_segment_pending = []
        state.custom_segment_awaiting_confirm = False
        state.custom_segments = []
        state.phase_history.append("custom_segments")

    def start_next_post_confirm_phase():
        """Start the next queued step once the current one ends: extra segments
        if configured, then landmarks when the queue is empty.

        `phase_queue` is rebuilt by enter_measure_phase every time the
        measurement step is entered, not here; this only consumes it (see
        enter_measure_phase for why). The dataset polygon no longer goes through
        here: it is the very first step of the review, before the measurement
        (see apply_polygon_edit_to_mask, called when the frame opens).
        """
        if not phase_queue:
            start_landmark_phase()
            return
        next_phase = phase_queue.pop(0)
        if next_phase == "custom_segments":
            enter_custom_segments_phase()

    def apply_polygon_edit_to_mask():
        """Close the review's first step, correcting the dataset polygon that
        goes into the YOLO-Seg database.

        If the user edited it, that shape becomes the mask everything else is
        computed from, not just a separate saved annotation: it is rasterized
        back into a binary mask and head, fluke, length and widths are
        recomputed with the usual PCA method. If the user declined with N, the
        original mask is kept.
        """
        nonlocal mask_final

        # sam_dirty starts True when SAM refinement is available (see the setup
        # below). Without clearing it here, the first tick of the measurement
        # step would immediately resegment from the original bbox, silently
        # wiping out the polygon correction just made (or the starting mask, if
        # the user answered N).
        state.sam_dirty = False

        if not (state.polygon_edit_confirmed and state.dataset_polygon):
            return
        rasterized = clean_review_mask(
            polygon_to_mask(state.dataset_polygon, frame.shape), frame.shape
        )
        if not rasterized.any():
            return
        mask_final = rasterized
        state.mask_final = mask_final
        state.mask_generation += 1
        if state.is_measure_mode and len(state.manual_measure_points) >= 3:
            state.active_widths_cache = measure_widths_along_curve(
                mask_final, measuring_line_points(state), gsd
            )
        else:
            state.info_points = measure_length_pca(mask_final, gsd)
            state.active_widths_cache = (
                state.info_points[3] if state.info_points is not None else []
            )

    def enter_polygon_phase():
        state.polygon_edit_mode = True
        state.polygon_edit_confirmed = None
        state.polygon_drawing_from_scratch = False
        state.polygon_scratch_building = False
        state.phase_history.append("polygon")

    def enter_other_whale_polygon_phase():
        """Offer to correct the other whales' outlines, one at a time.

        Runs after the main animal's dataset polygon, or straight away if that
        step was skipped for want of a mask. Any other whale in the same frame
        with an automatic mask (other_whale_queue_source, built in run_review)
        is offered in turn. Once the queue is empty, or if it was empty to begin
        with, the user is still asked whether there is another undetected animal
        to add by hand (see enter_other_whale_manual_prompt) before the quality
        step.

        The queue is rebuilt from scratch on every entry, including when
        stepping back with "Previous phase", the same convention as this
        frame's other steps: earlier answers, hand-added animals included, are
        not kept.
        """
        state.other_whale_queue = [dict(entry) for entry in (other_whale_queue_source or [])]
        state.other_whale_results = {}
        state.other_whale_landmark_results = {}
        state.other_whale_manual_added = []
        state.phase_history.append("other_whale_polygon")
        _advance_other_whale_queue()

    def _advance_other_whale_queue():
        if not state.other_whale_queue:
            state.other_whale_edit_mode = False
            state.active_other_whale_id = None
            enter_other_whale_manual_prompt()
            return
        entry = state.other_whale_queue.pop(0)
        _start_other_whale_entry(entry)

    def _start_other_whale_entry(entry):
        """Set up the state for ONE animal's outline.

        The animal either comes from the automatic queue, where entry["mask"]
        is its screening mask, or was just added by hand, where entry["mask"]
        is None and the outline is drawn from scratch.
        """
        state.active_other_whale_id = entry["whale_id"]
        state.other_whale_mask = entry["mask"]
        # Left as None rather than extracted here: this frame's first
        # sync_other_whale_polygon builds it at the slider's current detail
        # level. That avoids reading the slider in build_other_whale_queue,
        # which runs before the review panel even exists for the session's first
        # candidate.
        state.other_whale_polygon = None
        state.other_whale_polygon_epsilon = None
        state.other_whale_confirmed = None
        state.other_whale_drawing_from_scratch = False
        state.other_whale_scratch_building = False
        state.dragging_other_whale_polygon_index = None
        state.other_whale_edit_mode = True
        state.other_whale_landmark_head = entry.get("head_point")
        state.other_whale_landmark_fluke = entry.get("fluke_point")

    def enter_other_whale_manual_prompt():
        """Ask whether there is another whale in this frame that screening
        never detected or accepted: filtered out by shape, centring or
        confidence, or missed entirely.

        Runs after the automatic queue, whether it ran out or never existed, and
        again after each hand-added animal (see
        finish_other_whale_landmark_step) so several can be added.
        """
        state.other_whale_manual_confirmed = None
        state.other_whale_manual_prompt_mode = True

    def start_manual_other_whale(whale_id):
        """Add an animal with this ID to the review, with no starting mask.

        It enters the same other_whale_edit_mode the automatic queue uses, but
        with mask=None. The outline's yes/no block recognises that and offers to
        draw it from scratch rather than correct an existing one, the same as
        for the main animal when mask_final is missing.
        """
        state.other_whale_manual_prompt_mode = False
        state.other_whale_manual_added.append(whale_id)
        _start_other_whale_entry(
            {"whale_id": whale_id, "mask": None, "head_point": None, "fluke_point": None}
        )

    def enter_other_whale_landmark_step():
        """Offer the landmarks for one animal, after its outline.

        Then moves on to the next animal in the queue, or to the "another
        animal?" prompt if it was the last. When automatic head and fluke points
        exist (from the screening PCA, or computed on the spot if the outline
        was drawn from scratch and the shape is clear enough, see the end of the
        editing block) they start already placed. Otherwise the user places them
        like any other anatomical point: they are no longer skipped just for
        being missing.
        """
        state.other_whale_landmark_confirmed = None
        state.other_whale_landmark_mode = True

    def finish_other_whale_landmark_step(landmark_points):
        """Close the landmark session for the current animal and move on.

        Used both when the user confirms every point and, indirectly through
        its callers, when they decline with N or K. Moves to the next animal in
        the queue, or to the "another animal?" prompt if there were no more.
        """
        state.other_whale_landmark_results[state.active_other_whale_id] = landmark_points
        state.other_whale_landmark_mode = False
        state.doing_other_whale_landmarks = False
        state.landmark_mode = False
        if _landmark_panel is not None:
            _landmark_panel.hide()
        _advance_other_whale_queue()

    def finalize_other_whale_polygon():
        """Close the outline of the animal active in other_whale_edit_mode and
        move on to offering its landmarks.

        If the outline was drawn from scratch (no automatic mask, see
        start_manual_other_whale), this also tries to find head and fluke
        automatically by rasterizing the polygon just drawn and running the same
        PCA as for the main animal. When the shape is clear enough the landmarks
        start already placed; otherwise the user places them by hand
        (enter_other_whale_landmark_step never skips them, see its comment).
        """
        state.other_whale_results[state.active_other_whale_id] = state.other_whale_polygon
        if state.other_whale_drawing_from_scratch and state.other_whale_polygon:
            rasterized = clean_review_mask(
                polygon_to_mask(state.other_whale_polygon, frame.shape), frame.shape
            )
            if rasterized.any():
                pca_info = measure_length_pca(rasterized, gsd)
                if pca_info is not None:
                    state.other_whale_landmark_head = (
                        float(pca_info[1][0]), float(pca_info[1][1])
                    )
                    state.other_whale_landmark_fluke = (
                        float(pca_info[2][0]), float(pca_info[2][1])
                    )
        state.other_whale_edit_mode = False
        enter_other_whale_landmark_step()

    def set_measure_mode_for_current_mask():
        """Switch to manual measurement when there is no automatic one, and say
        why.

        That happens both when there is no mask at all and when there is one,
        even after editing the polygon, but the PCA found no head or fluke. The
        latter is common on candidates screening flagged for exactly that reason
        (see screening.commit_candidate).
        """
        if mask_final is None:
            if initial_mask_attempted or not resegment_enabled:
                state.is_measure_mode = True
                review_action_panel.status.set(
                    f"Reviewing frame {count}/{total} — no mask: "
                    "measure manually or discard."
                )
        elif state.info_points is None:
            state.is_measure_mode = True
            review_action_panel.status.set(
                f"Reviewing frame {count}/{total} — mask found but "
                "head/fluke not detected: measure manually, or "
                "fix the mask to retry the automatic measurement."
            )

    def start_quality_phase():
        """The first step after the dataset polygon: the Christiansen quality
        scoring, which runs before the measurement rather than after.

        The verdict is informative only, written to the CSV, and never blocks
        the measurement that follows; see finalize_quality_and_proceed.
        """
        nonlocal show_help
        state.quality_mode = True
        state.quality_index = 0
        state.quality_scores = [None] * len(quality.ATTRIBUTES)
        show_help = False
        state.phase_history.append("quality")

    def enter_measure_phase():
        """Enter the measurement step, the default when no other step flag is
        set.

        Unlike the others it has no flag of its own, so all that is cleared here
        is the state that shapes how it starts. Forced manual mode is restored
        by set_measure_mode_for_current_mask if it is still needed.

        `phase_queue` (extra segments, when configured) is also rebuilt on every
        entry, not just the first one after quality.
        `start_next_post_confirm_phase` consumes the queue with .pop(0), so
        without rebuilding it here, coming back with "Previous phase" and
        confirming again would find it already empty and silently skip the extra
        segments the second time.
        """
        nonlocal phase_queue
        state.is_measure_mode = False
        set_measure_mode_for_current_mask()
        state.phase_history.append("measure")
        phase_queue = []
        if custom_segment_names:
            phase_queue.append("custom_segments")

    def go_to_previous_phase():
        """Go back to this frame's previous step ("Previous phase" button, or
        the P key) and redo it from the start.

        Quality scores, extra segments and the polygon confirmation are all
        cleared: earlier answers from that pass are not kept, only the step
        itself goes back. `phase_history` records every step actually visited
        (the polygon and extra-segment steps are optional), so working out which
        step came before never has to guess the order: it reads what really
        happened for this frame.
        """
        if len(state.phase_history) < 2:
            review_action_panel.status.set(
                "Already at the first phase of this frame: there is no previous phase."
            )
            return
        state.phase_history.pop()
        target_phase = state.phase_history.pop()
        state.landmark_mode = False
        if _landmark_panel is not None:
            _landmark_panel.hide()
        state.quality_mode = False
        state.custom_segment_mode = False
        state.polygon_edit_mode = False
        state.other_whale_edit_mode = False
        state.other_whale_landmark_mode = False
        state.other_whale_manual_prompt_mode = False
        state.doing_other_whale_landmarks = False
        phase_entry_points = {
            "polygon": enter_polygon_phase,
            "other_whale_polygon": enter_other_whale_polygon_phase,
            "quality": start_quality_phase,
            "measure": enter_measure_phase,
            "custom_segments": enter_custom_segments_phase,
            "landmark": start_landmark_phase,
        }
        phase_entry_points[target_phase]()
        review_action_panel.status.set("Went back to the previous phase of this frame.")

    def finalize_quality_and_proceed(verdict, scores_by_key):
        """Close the quality step with its verdict ("passed",
        "failed: <reason>" or "skipped") and go on to the normal measurement
        regardless.

        The Christiansen verdict is recorded in the CSV for reference but no
        longer blocks the measurement or its count towards the quota: on some
        flights every frame is "poor" by those criteria and still has to be
        measured and kept.
        """
        nonlocal quality_verdict, quality_scores_by_key
        quality_verdict = verdict
        quality_scores_by_key = scores_by_key
        enter_measure_phase()

    def finalize_measurement():
        """Close the measurement step (free mask correction plus the manual or
        automatic measurement, the step after quality) and move on to extra
        segments and landmarks.

        It behaves the same whether quality passed or not (see
        finalize_quality_and_proceed): the Christiansen verdict is informative
        only. Returns False when there is no valid measurement to confirm (no
        mask and no manual points), leaving the user where they were.
        """
        nonlocal base_final_package, locked_mask
        nonlocal show_help, pending_resegment_at

        review_settings = {
            "contrast": review_contrast,
            "brightness": review_brightness,
            "clahe": review_clahe,
            "imgsz": review_quality,
            "quality_verdict": quality_verdict,
            "quality_scores": dict(quality_scores_by_key),
        }
        if state.is_measure_mode and len(state.manual_measure_points) >= 3:
            final_package = (
                lung_m,
                state.manual_measure_points,
                state.active_widths_cache,
                True,
                review_settings,
            )
        elif state.info_points is not None:
            final_package = (
                state.info_points[0],
                [state.info_points[1], state.info_points[2]],
                state.active_widths_cache,
                False,
                review_settings,
            )
        else:
            final_package = None
        if final_package is None:
            print("Cannot confirm: base measurement not available.")
            return False

        # Float, not int(): these frozen points feed both the final annotated
        # frame (which rounds for itself when drawing, see
        # build_annotated_result_frame) and the automatic-versus-human columns
        # in the final CSV. Rounding here would throw away the sub-pixel
        # accuracy of the manual placement in the saved data.
        frozen_spine = [
            (float(spine_point[0]), float(spine_point[1])) for spine_point in final_package[1]
        ]
        frozen_widths = []
        for width_info in final_package[2]:
            frozen_width = dict(width_info)
            frozen_width["pt_top"] = (float(width_info["pt_top"][0]), float(width_info["pt_top"][1]))
            frozen_width["pt_bot"] = (float(width_info["pt_bot"][0]), float(width_info["pt_bot"][1]))
            frozen_widths.append(frozen_width)
        base_final_package = (
            final_package[0],
            frozen_spine,
            frozen_widths,
            final_package[3],
            dict(final_package[4]),
        )
        locked_mask = mask_final.copy() if mask_final is not None else None
        state.sam_dirty = False
        pending_resegment_at = None
        state.is_width_edit_mode = False
        show_help = False
        start_next_post_confirm_phase()
        return True

    if _review_action_panel is None:
        _review_action_panel = ReviewActionPanel(config)
    review_action_panel = _review_action_panel
    review_action_panel.bind_state(state)
    review_action_panel.start(count, total, accepted_count, max_accepted, subject_label=subject_label)
    (
        contrast_position,
        review_brightness,
        review_clahe,
        show_enhanced,
        mask_opacity_position,
        hide_review_geometry,
        review_quality,
        marker_size,
        bezier_length_mode,
        marker_thickness,
    ) = review_action_panel.get_control_values()
    state.bezier_length_mode = bezier_length_mode
    review_contrast = contrast_position / 100.0
    enhanced_frame = enhance_review_frame(
        frame,
        review_contrast,
        review_brightness,
        review_clahe,
    )
    previous_image_settings = (
        contrast_position,
        review_brightness,
        review_clahe,
        review_quality,
    )
    previous_slider_release_counter = (
        review_action_panel.slider_release_counter
    )
    pending_resegment_at = None
    mask_opacity = mask_opacity_position / 100.0
    phase_queue = []
    # Fixed for this frame's whole review: it comes from config and does not
    # depend on what the user does. Computed once here rather than at the end of
    # every quality step, so enter_measure_phase can rebuild phase_queue even
    # when the measurement step is re-entered by stepping back (see its own
    # comment).
    custom_segment_names = [
        name.strip()
        for name in config.custom_segment_names.split(",")
        if name.strip()
    ]
    # The loop keeps running to process Tk events, but while the user is doing
    # nothing (just looking at the frame, which is most of a review) there is no
    # point redrawing 50 times a second. After a second of inactivity the loop
    # slows down, leaving the CPU and GPU free. The constant load is also the
    # main suspect behind the external SSD dropping out during interactive
    # review.
    last_activity_at = time.monotonic()
    previous_mouse_pos = state.last_mouse_pos
    previous_has_mouse_position = state.has_mouse_position
    needs_redraw = True

    mask_final = None
    if initial_mask is not None:
        mask_final = clean_review_mask(initial_mask, frame.shape)
    state.mask_final = mask_final
    state.mask_generation += 1
    lung_m = 0
    state.info_points = (
        measure_length_pca(mask_final, gsd)
        if mask_final is not None
        else None
    )
    automatic_measurement_snapshot = copy_automatic_measurement(state.info_points)
    state.active_widths_cache = state.info_points[3] if state.info_points is not None else []

    # The very first thing shown: correcting the dataset polygon, the one that
    # ends up in the YOLO-Seg database. Only once the user has fixed it (or
    # declined with N) does the measurement step begin, using that mask; see
    # apply_polygon_edit_to_mask, called on leaving this step.
    #
    # This step is always entered, even with no mask_final. In that case it
    # offers to draw one from scratch instead of correcting an automatic one
    # (see the yes/no block in the loop below), so a candidate with no mask
    # (added by hand from the preview, or rejected by the mean-shape filter) is
    # not stuck without a polygon just because screening found none.
    enter_polygon_phase()

    print(f"--- Reviewing Frame {count}/{total} ---")

    while True:
        review_action_panel.update()
        if review_action_panel.settings_requested:
            if _landmark_panel is not None:
                _landmark_panel.hide()
            state.landmark_mode = False
            review_action_panel.hide()
            close_review_windows()
            return None, None, "settings"

        if _landmark_panel is not None:
            if state.landmark_mode and _landmark_panel.finish_requested and state.doing_other_whale_landmarks:
                # A landmark session for ANOTHER whale (see
                # enter_other_whale_landmark_step), not the main animal. It does
                # not end the frame, only move to the next animal in the queue,
                # or to the quality step if that was the last.
                finish_other_whale_landmark_step(dict(_landmark_panel.points))
                continue
            if state.landmark_mode and _landmark_panel.finish_requested:
                landmark_points = dict(_landmark_panel.points)
                # The Y/N answer given in the dedicated step, not a sticky
                # checkbox: every frame asks again. Nothing is saved if the step
                # was skipped with K (answered N, or an edit confirmed with K
                # instead of Enter/Space), or if the panel's "Don't save this
                # mask in the dataset" box is ticked, which applies even when
                # the polygon was edited after answering Y.
                base_final_package[4]["save_training_mask"] = (
                    bool(state.polygon_edit_confirmed)
                    and not state.polygon_edit_skipped
                    and not review_action_panel.skip_dataset_save.get()
                )
                base_final_package[4]["polygon_epsilon_fraction"] = (
                    review_action_panel.get_polygon_epsilon_fraction()
                )
                base_final_package[4]["dataset_polygon"] = (
                    list(state.dataset_polygon) if state.dataset_polygon else None
                )
                base_final_package[4]["custom_segments"] = list(state.custom_segments)
                # Outlines of the OTHER whales in the same frame, either edited
                # or accepted as-is (see other_whale_edit_mode), so they can be
                # labelled in the training dataset alongside the main animal.
                # See other_whale_instance_lines in export.py.
                base_final_package[4]["other_whale_polygons"] = dict(state.other_whale_results)
                # Landmarks for those same other whales, or None when skipped
                # with N or K; see other_whale_landmark_mode. Only the ones
                # answered Yes have a dict of points here, and the rest produce
                # no landmark row at all (see run_review).
                base_final_package[4]["other_whale_landmarks"] = dict(state.other_whale_landmark_results)
                _landmark_panel.hide()
                state.landmark_mode = False
                return (
                    locked_mask,
                    (
                        *base_final_package,
                        landmark_points,
                        automatic_measurement_snapshot,
                    ),
                    "ok",
                )

        # Finishing with the mouse (the second point of the last configured
        # segment) has to be noticed here, not only on a key press, or the step
        # would stay active until the user happened to press some key after the
        # final click.
        if (
            state.custom_segment_mode
            and state.custom_segment_index >= len(state.custom_segment_queue)
        ):
            state.custom_segment_mode = False
            start_next_post_confirm_phase()

        # No resegmentation during the frame's guided steps. Editing the
        # dataset polygon (which now runs first), quality, extra segments and
        # landmarks all work on a frozen mask_final; otherwise a resegmentation,
        # triggered by a slider touched by accident, would silently overwrite
        # the hand-made polygon edits.
        in_post_confirm_phase = (
            state.quality_mode
            or state.custom_segment_mode
            or state.polygon_edit_mode
            or state.other_whale_edit_mode
            or state.other_whale_landmark_mode
            or state.other_whale_manual_prompt_mode
            or state.landmark_mode
        )

        canvas_width, canvas_height = review_action_panel.get_image_size()
        view_w, view_h = calculate_review_view_dimensions(
            state,
            canvas_width,
            canvas_height,
        )
        if state.zoom_factor > 1.0:
            state.current_win_w, state.current_win_h = canvas_width, canvas_height
            state.current_display_offset_x, state.current_display_offset_y = 0, 0
        else:
            (
                state.current_win_w,
                state.current_win_h,
                state.current_display_offset_x,
                state.current_display_offset_y,
            ) = review_action_panel.get_fitted_image_geometry(
                view_w,
                view_h,
            )

        if (
            not in_post_confirm_phase
            and pending_resegment_at is not None
            and time.monotonic() >= pending_resegment_at
            and not review_action_panel.slider_dragging
        ):
            state.sam_dirty = resegment_enabled
            pending_resegment_at = None

        if (
            state.sam_dirty
            and resegment_enabled
            and not state.is_width_edit_mode
            and not in_post_confirm_phase
        ):
            review_action_panel.status.set(
                f"Reviewing frame {count}/{total} — inference at "
                f"{review_quality}px in progress…"
            )
            review_action_panel.update()
            if state.mask_correction_enabled:
                candidate_bbox = current_bbox.copy()
                if bbox_predictor is not None:
                    review_action_panel.status.set(
                        f"Reviewing frame {count}/{total} — YOLO recomputing "
                        f"the bounding box at {review_quality}px…"
                    )
                    review_action_panel.update()
                    refreshed_bbox = bbox_predictor(
                        enhanced_frame,
                        review_quality,
                        current_bbox.copy(),
                    )
                    if refreshed_bbox is None:
                        state.sam_dirty = False
                        if mask_final is None:
                            state.is_measure_mode = True
                            state.active_widths_cache = []
                            review_action_panel.status.set(
                                f"Reviewing frame {count}/{total} — YOLO did "
                                "not find the animal: measure manually or "
                                "discard."
                            )
                        else:
                            review_action_panel.status.set(
                                f"Reviewing frame {count}/{total} — YOLO did "
                                "not find a new box: keeping the previous "
                                "mask."
                            )
                        continue
                    candidate_bbox = np.asarray(
                        refreshed_bbox,
                        dtype=np.float32,
                    ).reshape(-1, 4).copy()

                review_action_panel.status.set(
                    f"Reviewing frame {count}/{total} — SAM segmenting the "
                    f"new bounding box at {review_quality}px…"
                )
                review_action_panel.update()
                sam_predictor = getattr(sam_model, "predictor", None)
                if (
                    sam_predictor is not None
                    and hasattr(sam_predictor, "reset_image")
                ):
                    sam_predictor.reset_image()
                # A slightly larger box used only as SAM's prompt. The tight box
                # stays candidate_bbox/current_bbox, which tracking uses from one
                # frame to the next.
                sam_bbox = expand_bbox(
                    candidate_bbox, enhanced_frame.shape, config.sam_box_margin / 100.0
                )
                if len(state.drawing_points) > 0:
                    sam_res = device.predict_adaptive(
                        sam_model,
                        enhanced_frame,
                        review_quality,
                        "SAM review",
                        bboxes=sam_bbox,
                        points=[state.drawing_points],
                        labels=[state.drawing_labels],
                    )
                else:
                    sam_res = device.predict_adaptive(
                        sam_model,
                        enhanced_frame,
                        review_quality,
                        "SAM review",
                        bboxes=sam_bbox,
                    )

                if sam_res.masks is None or len(sam_res.masks.data) == 0:
                    print("SAM did not produce a mask for this frame.")
                    del sam_res
                    device.release_prediction(sam_model, clear_cache=False)
                    if mask_final is None:
                        state.is_measure_mode = True
                        state.active_widths_cache = []
                        review_action_panel.status.set(
                            f"Reviewing frame {count}/{total} — SAM did not "
                            "produce a mask: measure manually or discard."
                        )
                    else:
                        review_action_panel.status.set(
                            f"Reviewing frame {count}/{total} — SAM did not "
                            "produce a new mask: keeping the previous "
                            "one."
                        )
                    state.sam_dirty = False
                    continue
                mask_raw = (
                    (sam_res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                    * 255
                )
                del sam_res
                device.release_prediction(sam_model, clear_cache=False)
                current_bbox = candidate_bbox.copy()
            else:
                mask_raw = mask_predictor(
                    enhanced_frame,
                    review_quality,
                )
                if mask_raw is None:
                    print("YOLO did not produce a mask with these settings.")
                    if mask_final is None:
                        state.is_measure_mode = True
                        state.active_widths_cache = []
                        review_action_panel.status.set(
                            f"Reviewing frame {count}/{total} — YOLO did not "
                            "produce a mask: measure manually or discard."
                        )
                    state.sam_dirty = False
                    continue

            mask_final = clean_review_mask(mask_raw, frame.shape)
            state.mask_final = mask_final
            state.mask_generation += 1

            state.info_points = measure_length_pca(mask_final, gsd)
            if not state.drawing_points:
                automatic_measurement_snapshot = copy_automatic_measurement(
                    state.info_points
                )

            if state.is_measure_mode and not state.manual_measure_points:
                state.is_measure_mode = False
            review_action_panel.status.set(
                f"Reviewing frame {count}/{total} — mask obtained at "
                f"{review_quality}px."
            )

            if state.is_measure_mode:
                if len(state.manual_measure_points) >= 3 and mask_final is not None:
                    state.active_widths_cache = measure_widths_along_curve(mask_final, measuring_line_points(state), gsd)
                else:
                    state.active_widths_cache = []
            else:
                state.active_widths_cache = state.info_points[3] if state.info_points is not None else []

            state.sam_dirty = False

        (
            contrast_position,
            review_brightness,
            review_clahe,
            show_enhanced,
            mask_opacity_position,
            hide_review_geometry,
            review_quality,
            marker_size,
            bezier_length_mode,
            marker_thickness,
        ) = review_action_panel.get_control_values()
        state.bezier_length_mode = bezier_length_mode
        mask_opacity = mask_opacity_position / 100.0
        current_image_settings = (
            contrast_position,
            review_brightness,
            review_clahe,
            review_quality,
        )
        if current_image_settings != previous_image_settings:
            review_contrast = contrast_position / 100.0
            enhanced_frame = enhance_review_frame(
                frame,
                review_contrast,
                review_brightness,
                review_clahe,
            )
            if in_post_confirm_phase or review_action_panel.slider_dragging:
                pending_resegment_at = None
            else:
                pending_resegment_at = time.monotonic() + 0.25
            previous_image_settings = current_image_settings
        if (
            review_action_panel.slider_release_counter
            != previous_slider_release_counter
        ):
            previous_slider_release_counter = (
                review_action_panel.slider_release_counter
            )
            if not in_post_confirm_phase:
                pending_resegment_at = time.monotonic() + 0.25

        sync_dataset_polygon(state, review_action_panel, mask_final)
        sync_other_whale_polygon(state, review_action_panel)
        is_dragging_polygon = state.dragging_polygon_index is not None
        is_dragging_other_whale_polygon = state.dragging_other_whale_polygon_index is not None
        # Hidden only once the user has answered yes, not during the Y/N
        # question, when there is no polygon to show yet: the frame underneath
        # has to be clean to place vertices accurately. Same reasoning for the
        # other animal's outline.
        hide_mask_for_polygon_phase = (
            state.polygon_edit_mode and state.polygon_edit_confirmed
        ) or (state.other_whale_edit_mode and state.other_whale_confirmed)

        # lung_m feeds the final package if the user confirms (Space/Enter) in
        # this same iteration, so it must stay current even when the redraw
        # below is skipped, or the confirmation would use a stale value. It is
        # only arithmetic over a few points and never touches pixels, so it is
        # cheap enough to do every iteration.
        if not in_post_confirm_phase:
            if state.is_measure_mode and len(state.manual_measure_points) >= 1:
                manual_lung_px = manual_measure_length_px(
                    state.manual_measure_points, bezier_length_mode
                )
                lung_m = manual_lung_px * gsd
            elif state.info_points is not None:
                lung_m = state.info_points[0]

        # ponytail: the redraw (copy frame, blend, contours, resize) is the
        # main cost of the review loop and used to run unconditionally up to 50
        # times a second even on a still screen. That sustained load is the main
        # suspect behind the external SSD dropping out on the M1. It now redraws
        # only when something visible has changed since last time, using the
        # same signal the inactivity check below uses.
        # Known gap: resizing the window without moving the mouse (rare). If it
        # matters, add canvas_width/canvas_height to the comparison.
        if needs_redraw:
            display_base = enhanced_frame if show_enhanced else frame

            # Crop to the visible part (the current zoom view) FIRST, then blend
            # and draw the outline on that crop only. The old code copied,
            # blended and drew on the whole frame (up to 27ms per redraw on a 4K
            # frame) and then threw away everything but the crop: a cost that
            # did not shrink with zoom, and what made Shift-panning jerky.
            # Measured at about 0.3ms cropping first, roughly 100x faster, with
            # exactly the same result on screen.
            x_start = max(0, min(int(state.zoom_center_x - view_w / 2), state.img_w - view_w))
            y_start = max(0, min(int(state.zoom_center_y - view_h / 2), state.img_h - view_h))
            state.current_view_roi = (x_start, y_start, view_w, view_h)
            cropped_base = display_base[y_start : y_start + view_h, x_start : x_start + view_w]
            if cropped_base.size == 0:
                # Should never happen, since calculate_review_view_dimensions
                # guarantees view_w/h <= img_w/h, but if it does, fall back to
                # the whole frame rather than show an empty image.
                cropped_base = display_base
                x_start, y_start = 0, 0
                state.current_view_roi = (0, 0, state.img_w, state.img_h)
            cropped_mask = (
                mask_final[y_start : y_start + cropped_base.shape[0], x_start : x_start + cropped_base.shape[1]]
                if mask_final is not None
                else None
            )
            overlay = cropped_base.copy()
            if (
                mask_final is not None
                and not state.landmark_mode
                and not hide_review_geometry
                and not hide_mask_for_polygon_phase
            ):
                overlay[cropped_mask > 0] = [0, 0, 255]
                overlay = cv2.addWeighted(
                    overlay,
                    mask_opacity,
                    cropped_base,
                    1.0 - mask_opacity,
                    0,
                )
                # Contours come from the WHOLE mask (cached, see
                # get_mask_contours), not recomputed on the crop. At high zoom
                # the shape usually continues past the edge of the view, and a
                # contour recomputed on the crop alone would be closed off there,
                # drawing a straight edge that does not exist in the real mask.
                # `offset` shifts the contours (in image coordinates) onto the
                # small cropped canvas, so the drawing matches what the old
                # whole-image version produced without copying and blending the
                # whole frame.
                contours = get_mask_contours(state, mask_final)
                cv2.drawContours(
                    overlay, contours, -1, (255, 255, 0), 1, cv2.LINE_AA,
                    offset=(-x_start, -y_start),
                )

            # Bright orange at rest, dimmer while a point is being dragged (same
            # BGR, lower brightness). Used below for the polygon outline and its
            # crosses, which are drawn after the zoom resize (see
            # "interactive_screen_draws").
            #
            # The "Dataset polygon: N vertices..." label that used to be here was
            # removed: it was drawn BEFORE the zoom resize, so it grew and moved
            # with the image instead of staying readable. The same message is
            # shown below with the step instructions, after the resize.
            polygon_color = (0, 100, 160) if is_dragging_polygon else (0, 140, 255)
            # A different colour from the main animal (cyan rather than orange):
            # this outline belongs to ANOTHER whale and must not be mistaken for
            # the dataset polygon of the one under review.
            other_whale_polygon_color = (
                (160, 100, 0) if is_dragging_other_whale_polygon else (255, 140, 0)
            )

            txt_y = state.current_win_h - 100
            if txt_y < 50: txt_y = 50

            measurement_summary = None
            if not in_post_confirm_phase and state.is_measure_mode:
                # "Hide mask and lines" does not hide the manual line being
                # drawn: while drawing it is the only thing showing what is being
                # marked, whatever the checkbox says.
                if len(state.manual_measure_points) >= 1:
                    # The actual drawing (line, crosses, H-F labels) happens
                    # after the zoom resize, in the "interactive_screen_draws"
                    # block below. Only the arithmetic for the summary text is
                    # done here.
                    manual_lung_px = manual_measure_length_px(
                        state.manual_measure_points, bezier_length_mode
                    )
                    lung_m = manual_lung_px * gsd
                    measurement_summary = (
                        f"MANUAL LENGTH: {lung_m:.2f} "
                        f"{'m' if measure_in_meters else 'px'}"
                    )
            elif not in_post_confirm_phase and state.info_points:
                lung_m, p_head, p_notch, _ = state.info_points
                measurement_summary = (
                    f"AUTOMATIC LENGTH: {lung_m:.2f} "
                    f"{'m' if measure_in_meters else 'px'}"
                )
            elif not in_post_confirm_phase and mask_final is None:
                measurement_summary = (
                    "NO MASK — use Manual measurement or discard the frame"
                )

            if not in_post_confirm_phase and measure_in_meters and not np.isfinite(gsd):
                # gsd comes from the nearest laser altitude in time (see
                # screening.commit_candidate). If there is none for this second,
                # for instance when the video starts before the telemetry, every
                # length computed here would silently be NaN. The user needs to
                # know before measuring or accepting by mistake.
                laser_warning = (
                    "NO LASER MEASUREMENT FOR THIS SECOND — "
                    "length in meters not available"
                )
                measurement_summary = (
                    f"{measurement_summary} | {laser_warning}"
                    if measurement_summary
                    else laser_warning
                )

            # Widths stay visible while being edited (W) even with the "hide"
            # box ticked, or the user could not see what they are dragging.
            # Everything else (mask, SAM points) still obeys the checkbox.
            #
            # The points themselves (SAM, segments, landmarks) are drawn below in
            # "interactive_screen_draws"; this only decides WHICH to show. The
            # if/elif chain is deliberate: these are mutually exclusive, not
            # independent conditions.
            if not in_post_confirm_phase and (
                not hide_review_geometry or state.is_width_edit_mode
            ):
                interactive_markers_mode = (
                    None if state.is_width_edit_mode else "sam_points"
                )
            elif state.custom_segment_mode:
                interactive_markers_mode = "custom_segments"
            elif (
                state.landmark_mode
                and state.active_landmark_code is None
                and _landmark_panel is not None
            ):
                interactive_markers_mode = "landmarks_all"
            elif state.landmark_mode and state.pending_landmark_point is not None:
                interactive_markers_mode = "landmark_pending"
            else:
                interactive_markers_mode = None

            # `overlay` is already just the current view, cropped at the top of
            # this block before blending and drawing, so all that is left here is
            # the resize for the screen, not a second crop.
            if overlay.shape[0] > 0 and overlay.shape[1] > 0:
                # INTER_AREA when shrinking, to avoid aliasing. INTER_NEAREST
                # when enlarging, not INTER_LINEAR: at high zoom the latter blurs
                # both the image pixels and the mask, lines and crosses drawn on
                # top at native resolution, instead of keeping them sharp.
                interpolation = (
                    cv2.INTER_AREA
                    if overlay.shape[1] > state.current_win_w
                    else cv2.INTER_NEAREST
                )
                display_img = cv2.resize(
                    overlay,
                    (state.current_win_w, state.current_win_h),
                    interpolation=interpolation,
                )
            else:
                display_img = overlay

            # interactive_screen_draws: EVERY interactive point, cross and line
            # — length, widths, dataset polygon, SAM points, extra segments,
            # landmarks — is drawn HERE, after the crop and the zoom resize,
            # rather than on the full-resolution overlay above.
            #
            # Why: at high zoom that overlay is enlarged with INTER_NEAREST (to
            # stay sharp on the mask), so a marker drawn earlier ends up as a
            # block as wide as the zoom factor — thick and blurry — and dragging
            # it by less than one real pixel changes nothing on screen. Drawn
            # here, at the resolution the user actually sees, markers stay thin
            # at any zoom and follow the drag smoothly, free of the original
            # image's pixel grid. draw_cross_marker is still the same crisp,
            # non-anti-aliased function used everywhere, just fed screen
            # coordinates instead of image ones.
            view_x, view_y, roi_w, roi_h = state.current_view_roi
            screen_scale_x = state.current_win_w / max(roi_w, 1)
            screen_scale_y = state.current_win_h / max(roi_h, 1)

            def to_screen(point):
                return (
                    (point[0] - view_x) * screen_scale_x,
                    (point[1] - view_y) * screen_scale_y,
                )

            if (
                (mask_final is not None or state.polygon_drawing_from_scratch)
                and state.polygon_edit_mode
                and state.polygon_edit_confirmed
                and state.dataset_polygon
            ):
                # Closed (True) even while still drawing from scratch
                # (polygon_scratch_building): with one or two points it just
                # looks like a point or a segment, and becomes a real outline as
                # soon as there are enough. No special case needed.
                polygon_screen_pts = np.array(
                    [subpixel_point(to_screen(p)) for p in state.dataset_polygon],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    display_img, [polygon_screen_pts], True, polygon_color,
                    1, cv2.LINE_AA, SUBPIXEL_SHIFT,
                )
                for point in state.dataset_polygon:
                    draw_cross_marker(display_img, to_screen(point), marker_size, polygon_color, thickness=marker_thickness)

            if (
                state.other_whale_edit_mode
                and state.other_whale_confirmed
                and state.other_whale_polygon
            ):
                other_whale_screen_pts = np.array(
                    [subpixel_point(to_screen(p)) for p in state.other_whale_polygon],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    display_img, [other_whale_screen_pts], True, other_whale_polygon_color,
                    1, cv2.LINE_AA, SUBPIXEL_SHIFT,
                )
                for point in state.other_whale_polygon:
                    draw_cross_marker(
                        display_img, to_screen(point), marker_size,
                        other_whale_polygon_color, thickness=marker_thickness,
                    )

            if interactive_markers_mode == "sam_points":
                for pt, lbl in zip(state.drawing_points, state.drawing_labels):
                    draw_cross_marker(
                        display_img, to_screen(pt), marker_size,
                        (0, 255, 0) if lbl == 1 else (0, 0, 255),
                        thickness=marker_thickness,
                    )
            elif interactive_markers_mode == "custom_segments":
                for segment_index, segment in enumerate(state.custom_segments):
                    screen_a, screen_b = to_screen(segment["pt_a"]), to_screen(segment["pt_b"])
                    cv2.line(display_img, subpixel_point(screen_a), subpixel_point(screen_b), (0, 128, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT)
                    draw_cross_marker(display_img, screen_a, marker_size, (0, 128, 255), thickness=marker_thickness)
                    draw_cross_marker(display_img, screen_b, marker_size, (0, 128, 255), thickness=marker_thickness)
                    draw_readable_text(
                        display_img,
                        f"{segment.get('name', f'segment_{segment_index + 1}')}: "
                        f"{segment['length_m']:.2f} {'m' if measure_in_meters else 'px'}",
                        (int(screen_b[0] + 8), int(screen_b[1] - 8)),
                        0.55, (0, 128, 255),
                    )
                if state.custom_segment_awaiting_confirm and len(state.custom_segment_pending) >= 2:
                    # Yellow until confirmed, so it is obvious at a glance which
                    # segments are final (orange) and which is not.
                    screen_a = to_screen(state.custom_segment_pending[0])
                    screen_b = to_screen(state.custom_segment_pending[1])
                    pending_length = math.hypot(
                        state.custom_segment_pending[1][0] - state.custom_segment_pending[0][0],
                        state.custom_segment_pending[1][1] - state.custom_segment_pending[0][1],
                    ) * state.current_gsd
                    cv2.line(display_img, subpixel_point(screen_a), subpixel_point(screen_b), (0, 255, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT)
                    draw_cross_marker(display_img, screen_a, marker_size, (0, 255, 255), thickness=marker_thickness)
                    draw_cross_marker(display_img, screen_b, marker_size, (0, 255, 255), thickness=marker_thickness)
                    draw_readable_text(
                        display_img,
                        f"{pending_length:.2f} {'m' if measure_in_meters else 'px'} — "
                        "Enter/Space confirm, R redo",
                        (int(screen_b[0] + 8), int(screen_b[1] - 8)),
                        0.55, (0, 255, 255),
                    )
                elif state.custom_segment_pending:
                    screen_pending = to_screen(state.custom_segment_pending[0])
                    draw_cross_marker(display_img, screen_pending, marker_size, (0, 128, 255), thickness=marker_thickness)
            elif interactive_markers_mode == "landmarks_all":
                for landmark_code, landmark_point in _landmark_panel.points.items():
                    if landmark_point is None:
                        continue
                    screen_point = to_screen(landmark_point)
                    point_color = (
                        (0, 220, 255)
                        if landmark_code in {"S", "F"}
                        else (80, 255, 80)
                    )
                    draw_cross_marker(display_img, screen_point, marker_size, point_color, thickness=marker_thickness)
                    draw_readable_text(
                        display_img, landmark_code,
                        (int(screen_point[0] + 6), int(screen_point[1] - 6)),
                        0.45, point_color,
                    )
            elif interactive_markers_mode == "landmark_pending":
                screen_point = to_screen(state.pending_landmark_point)
                draw_cross_marker(display_img, screen_point, marker_size, (0, 255, 255), thickness=marker_thickness)
                draw_readable_text(
                    display_img, state.active_landmark_code or "",
                    (int(screen_point[0] + 8), int(screen_point[1] - 8)),
                    0.45, (0, 255, 255),
                )

            if not in_post_confirm_phase:
                if state.is_measure_mode:
                    # Nested under is_measure_mode rather than a flat elif with
                    # info_points below: with no points placed yet it must not
                    # show the leftover automatic line.
                    if len(state.manual_measure_points) >= 1:
                        # The manual line stays visible while being drawn even
                        # with "hide" ticked; see the comment on manual
                        # measurement above.
                        screen_pts = [to_screen(p) for p in state.manual_measure_points]
                        if bezier_length_mode and len(state.manual_measure_points) >= 3:
                            curve_screen_pts = np.array(
                                [
                                    subpixel_point(to_screen(p))
                                    for p in bezier_curve_points(state.manual_measure_points)
                                ],
                                dtype=np.int32,
                            ).reshape(-1, 1, 2)
                            cv2.polylines(display_img, [curve_screen_pts], False, (0, 165, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT)
                        else:
                            for idx_pt in range(len(screen_pts) - 1):
                                cv2.line(
                                    display_img,
                                    subpixel_point(screen_pts[idx_pt]),
                                    subpixel_point(screen_pts[idx_pt + 1]),
                                    (0, 165, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT,
                                )
                        for screen_pt in screen_pts:
                            draw_cross_marker(display_img, screen_pt, marker_size, (0, 165, 255), thickness=marker_thickness)
                        if not state.is_width_edit_mode:
                            draw_readable_text(
                                display_img, "H",
                                (int(screen_pts[0][0] - 18), int(screen_pts[0][1] - 15)),
                                0.9, (255, 0, 255),
                            )
                            if len(screen_pts) >= 2:
                                draw_readable_text(
                                    display_img, "F",
                                    (int(screen_pts[-1][0] + 8), int(screen_pts[-1][1] + 15)),
                                    0.9, (255, 0, 255),
                                )
                elif state.info_points and (not hide_review_geometry or state.is_width_edit_mode):
                    _, p_head, p_notch, _ = state.info_points
                    screen_head, screen_notch = to_screen(p_head), to_screen(p_notch)
                    cv2.line(
                        display_img,
                        subpixel_point(screen_head), subpixel_point(screen_notch),
                        (0, 255, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT,
                    )
                    draw_cross_marker(display_img, screen_head, marker_size, (0, 255, 0), thickness=marker_thickness)
                    draw_cross_marker(display_img, screen_notch, marker_size, (0, 0, 255), thickness=marker_thickness)
                    if not state.is_width_edit_mode:
                        draw_readable_text(
                            display_img, "H",
                            (int(screen_head[0] - 18), int(screen_head[1] + 15)),
                            0.9, (255, 0, 255),
                        )
                        draw_readable_text(
                            display_img, "F",
                            (int(screen_notch[0] + 8), int(screen_notch[1] + 15)),
                            0.9, (255, 0, 255),
                        )

                if not hide_review_geometry or state.is_width_edit_mode:
                    for w_info in state.active_widths_cache:
                        screen_top, screen_bot = to_screen(w_info['pt_top']), to_screen(w_info['pt_bot'])
                        cv2.line(
                            display_img,
                            subpixel_point(screen_top), subpixel_point(screen_bot),
                            (255, 0, 255), 1, cv2.LINE_AA, SUBPIXEL_SHIFT,
                        )
                        width_color = (0, 255, 255) if state.is_width_edit_mode else (255, 100, 255)
                        draw_cross_marker(display_img, screen_top, marker_size, width_color, thickness=marker_thickness)
                        draw_cross_marker(display_img, screen_bot, marker_size, width_color, thickness=marker_thickness)
                        if not state.is_width_edit_mode:
                            draw_readable_text(
                                display_img,
                                str(round(w_info["width_m"], 2)),
                                (int(screen_bot[0] - 16), int(screen_bot[1] - 14)),
                                0.55, (255, 255, 255),
                            )

            if state.landmark_mode and state.active_landmark_code is not None:
                landmark_name = LANDMARK_LABELS.get(
                    state.active_landmark_code,
                    state.active_landmark_code,
                )
                active_text = (
                    f"SELECT: {landmark_name} ({state.active_landmark_code})"
                )
                font_scale = 0.68
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    active_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness,
                )
                cv2.rectangle(
                    display_img,
                    (14, 12),
                    (
                        min(display_img.shape[1] - 10, 34 + text_width),
                        28 + text_height + baseline,
                    ),
                    (45, 75, 85),
                    thickness=cv2.FILLED,
                )
                draw_readable_text(
                    display_img,
                    active_text,
                    (24, 22 + text_height),
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    outline=False,
                )

            if not in_post_confirm_phase:
                text_background = display_img.copy()
                cv2.rectangle(
                    text_background,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 1160), 170),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                cv2.rectangle(
                    text_background,
                    (15, max(0, txt_y - 38)),
                    (
                        min(display_img.shape[1] - 15, 620),
                        min(display_img.shape[0] - 10, txt_y + 62),
                    ),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                display_img = cv2.addWeighted(
                    text_background,
                    0.62,
                    display_img,
                    0.38,
                    0,
                )
                draw_readable_text(
                    display_img,
                    f"ZOOM: {state.zoom_factor:.1f}x",
                    (30, txt_y),
                    1.2,
                    (0, 255, 255),
                )
                base_mode = "SAM MASK CORRECTION" if state.mask_correction_enabled else "YOLO MASK"
                mode_text = (
                    "EDITING MEASUREMENTS (W)"
                    if state.is_width_edit_mode
                    else ("MANUAL MEASUREMENT (M)" if state.is_measure_mode else base_mode)
                )
                color_mode = (
                    (0, 0, 255)
                    if state.is_width_edit_mode
                    else ((0, 165, 255) if state.is_measure_mode else (0, 255, 0))
                )
                draw_readable_text(
                    display_img,
                    mode_text,
                    (30, txt_y + 40),
                    1.0,
                    color_mode,
                )
                settings_text = (
                    f"Contrast {review_contrast:.1f} | Brightness {review_brightness:+d} | "
                    f"CLAHE {'ON' if review_clahe else 'OFF'} | "
                    f"Quality {review_quality}px | "
                    f"Preview {'ENHANCED' if show_enhanced else 'ORIGINAL'}"
                    + (" | LINES HIDDEN" if hide_review_geometry else "")
                    + (" | updating mask…" if pending_resegment_at else "")
                )
                draw_readable_text(
                    display_img,
                    settings_text,
                    (30, 38),
                    0.75,
                    (255, 255, 255),
                    outline=False,
                )
                if measurement_summary is not None:
                    draw_readable_text(
                        display_img,
                        measurement_summary,
                        (30, 140),
                        0.95,
                        (
                            (0, 80, 255)
                            if mask_final is None
                            else (
                                (0, 220, 255)
                                if not state.is_measure_mode
                                else (0, 165, 255)
                            )
                        ),
                        2,
                    )
            if show_help and not in_post_confirm_phase:
                help_lines = [
                    "Image, mask and lines: use the single REVIEW CONTROLS panel",
                    "Z undo last point   M manual measurement   W edit measurements   "
                    "K skip phase   H hide help",
                ]
                for help_index, help_text in enumerate(help_lines):
                    draw_readable_text(
                        display_img,
                        help_text,
                        (30, 72 + help_index * 30),
                        0.65,
                        (230, 230, 230),
                        1,
                        outline=False,
                    )

            if state.quality_mode:
                current_attribute = quality.ATTRIBUTES[state.quality_index]
                quality_lines = [
                    f"Quality check {state.quality_index + 1}/{len(quality.ATTRIBUTES)}: "
                    f"{current_attribute.label}",
                    "Press 1 (good)  2 (medium)  3 (poor)   Z undo   "
                    "K skip scoring (measure anyway)   X discard frame",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(quality_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (0, 255, 255),
                        2,
                        outline=False,
                    )

            if state.custom_segment_mode and state.custom_segment_index < len(state.custom_segment_queue):
                current_segment_name = state.custom_segment_queue[state.custom_segment_index]
                segment_lines = [
                    f"Custom segment {state.custom_segment_index + 1}/{len(state.custom_segment_queue)}: "
                    f"{current_segment_name}",
                    (
                        "Enter/Space/K to confirm this segment   R to redo it"
                        if state.custom_segment_awaiting_confirm
                        else "Click two points to measure it   Right-click to undo   N/K to skip it"
                    ),
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(segment_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (0, 128, 255),
                        2,
                        outline=False,
                    )

            if state.polygon_edit_mode and state.polygon_edit_confirmed is None:
                if mask_final is None:
                    polygon_lines = [
                        "No automatic mask for this frame — draw its outline from scratch?",
                        "Y = yes, draw it   N/K = no, leave this frame without a mask/polygon   X discard frame",
                    ]
                else:
                    polygon_lines = [
                        "Adjust the mask polygon? It becomes the mask used for length/width",
                        "Y = yes, edit it   N/K = no, use the automatic mask as-is   X discard frame",
                    ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(polygon_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (0, 255, 0),
                        2,
                        outline=False,
                    )
            elif state.polygon_edit_mode and state.polygon_scratch_building:
                polygon_lines = [
                    f"Drawing the outline from scratch — click to add a point "
                    f"({len(state.dataset_polygon or [])} so far, at least 3 needed)",
                    "Right-click to undo the last point   Enter/Space/K to close it   "
                    "R to start over   X to discard   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(polygon_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (0, 255, 0),
                        2,
                        outline=False,
                    )
            elif state.polygon_edit_mode and state.polygon_edit_confirmed:
                polygon_lines = [
                    "Mask polygon editing — drag an orange point to adjust "
                    "it, click a side to add one",
                    "Enter/Space/K to confirm and go to quality   R to reset   X to discard   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(polygon_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (0, 140, 255),
                        2,
                        outline=False,
                    )

            if state.other_whale_edit_mode and state.other_whale_confirmed is None:
                if state.other_whale_mask is None:
                    other_whale_lines = [
                        f"Whale {state.active_other_whale_id} has no automatic mask — "
                        "draw its outline from scratch?",
                        "Y = yes, draw it   N/K = no, drop this whale   S for settings",
                    ]
                else:
                    other_whale_lines = [
                        f"Another sperm whale (whale {state.active_other_whale_id}) is in this "
                        "frame — adjust its outline too?",
                        "Y = yes, edit it   N/K = no, use its automatic mask as-is   S for settings",
                    ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(other_whale_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (255, 140, 0),
                        2,
                        outline=False,
                    )
            elif state.other_whale_edit_mode and state.other_whale_scratch_building:
                other_whale_lines = [
                    f"Whale {state.active_other_whale_id} — click to add a point "
                    f"({len(state.other_whale_polygon or [])} so far, at least 3 needed)",
                    "Right-click to undo   Enter/Space/K to close it   R to start over   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(other_whale_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (255, 140, 0),
                        2,
                        outline=False,
                    )
            elif state.other_whale_edit_mode and state.other_whale_confirmed:
                other_whale_lines = [
                    f"Whale {state.active_other_whale_id} outline editing — drag an orange "
                    "point to adjust it, click a side to add one",
                    "Enter/Space/K to confirm   R to reset   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(other_whale_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (255, 140, 0),
                        2,
                        outline=False,
                    )

            if state.other_whale_manual_prompt_mode and state.other_whale_manual_confirmed is None:
                manual_prompt_lines = [
                    "Is there another sperm whale in this frame that wasn't detected/kept "
                    "automatically?",
                    "Y = yes, add it (pick an ID)   N/K = no   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(manual_prompt_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (255, 140, 0),
                        2,
                        outline=False,
                    )

            if state.other_whale_landmark_mode and state.other_whale_landmark_confirmed is None:
                other_whale_landmark_lines = [
                    f"Place anatomical landmarks for whale {state.active_other_whale_id} too?",
                    "Y = yes, open the landmark panel   N/K = no, skip its landmarks   S for settings",
                ]
                cv2.rectangle(
                    display_img,
                    (15, 10),
                    (min(display_img.shape[1] - 15, 900), 90),
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                )
                for line_index, line_text in enumerate(other_whale_landmark_lines):
                    draw_readable_text(
                        display_img,
                        line_text,
                        (30, 38 + line_index * 32),
                        0.75,
                        (255, 140, 0),
                        2,
                        outline=False,
                    )

            review_action_panel.present_image(
                display_img,
                state.current_display_offset_x,
                state.current_display_offset_y,
            )
        key = review_action_panel.pop_key()

        # pop_dirty() must always be called, never inside the `or`, which would
        # short-circuit it. It covers zoom, the landmark buttons and other
        # Tkinter buttons that write to the state directly, bypassing the key and
        # mouse queue below. Without it the view would stay on the old frame
        # until some tracked input arrived.
        external_change = review_action_panel.pop_dirty()
        is_active_now = (
            key != 255
            or external_change
            or state.last_mouse_pos != previous_mouse_pos
            or state.has_mouse_position != previous_has_mouse_position
            or review_action_panel.slider_dragging
            or pending_resegment_at is not None
            or state.sam_dirty
            or state.dragging_polygon_index is not None
        )
        previous_mouse_pos = state.last_mouse_pos
        previous_has_mouse_position = state.has_mouse_position
        needs_redraw = is_active_now
        if is_active_now:
            last_activity_at = time.monotonic()
        # ponytail: a fixed 1s threshold instead of real system-load detection.
        # If that is not enough, the next step is a more aggressive idle check.
        idle_for = time.monotonic() - last_activity_at
        time.sleep(0.02 if idle_for < 1.0 else 0.15)

        if key in (ord("+"), ord("="), 43, 61, 171):
            increase_review_zoom(state)
            continue
        if key in (ord("-"), ord("_"), 45, 95, 173):
            decrease_review_zoom(state)
            continue
        if key == ord("p"):
            # Works in every step (polygon, quality, measurement, extra
            # segments, landmarks), which is why it is handled here before the
            # per-step blocks, like zoom.
            go_to_previous_phase()
            continue

        if state.landmark_mode:
            if key in (ord("a"), ord("A")):
                _landmark_panel.confirm_and_advance()
                continue
            if key in (ord("x"), ord("k")):
                # Last step, so "skip" means discard this frame and move to the
                # next: there is no following step to skip to.
                if not confirm_discard():
                    continue
                _landmark_panel.hide()
                state.landmark_mode = False
                return None, None, "discard"
            if key == ord("s"):
                _landmark_panel.hide()
                state.landmark_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.quality_mode:
            if key in (ord("1"), ord("2"), ord("3")):
                state.quality_scores[state.quality_index] = key - ord("0")
                state.quality_index += 1
                if state.quality_index >= len(quality.ATTRIBUTES):
                    scores_by_key = {
                        attribute.key: score
                        for attribute, score in zip(
                            quality.ATTRIBUTES, state.quality_scores
                        )
                    }
                    passed, reason = quality.evaluate(scores_by_key)
                    state.quality_mode = False
                    # Moved ahead of the measurement, but the verdict is only
                    # informative and ends up in the CSV: failing the
                    # Christiansen criteria does not stop the frame being
                    # measured and kept (see finalize_quality_and_proceed).
                    if not passed:
                        print(
                            f"Quality check not passed (Christiansen criteria): {reason} "
                            "— measuring anyway."
                        )
                    finalize_quality_and_proceed(
                        "passed" if passed else f"failed: {reason}",
                        scores_by_key,
                    )
            elif key == ord("z") and state.quality_index > 0:
                state.quality_index -= 1
                state.quality_scores[state.quality_index] = None
            elif key == ord("x"):
                if confirm_discard():
                    state.quality_mode = False
                    return None, None, "discard"
            elif key == ord("s"):
                state.quality_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            elif key == ord("k"):
                # Skip the quality scoring entirely and go on to the normal
                # measurement. The verdict is "skipped", which is informative
                # only in the CSV and distinct from a real failure.
                state.quality_mode = False
                scores_by_key = {
                    attribute.key: score
                    for attribute, score in zip(
                        quality.ATTRIBUTES, state.quality_scores
                    )
                    if score is not None
                }
                finalize_quality_and_proceed("skipped", scores_by_key)
            continue

        if state.custom_segment_mode:
            if state.custom_segment_awaiting_confirm:
                if key in (32, 13, ord("k")):
                    point_a, point_b = state.custom_segment_pending[:2]
                    length = math.hypot(
                        point_b[0] - point_a[0], point_b[1] - point_a[1]
                    ) * state.current_gsd
                    segment_name = state.custom_segment_queue[state.custom_segment_index]
                    state.custom_segments.append(
                        {
                            "name": segment_name,
                            "pt_a": point_a,
                            "pt_b": point_b,
                            "length_m": length,
                        }
                    )
                    state.custom_segment_pending = []
                    state.custom_segment_awaiting_confirm = False
                    state.custom_segment_index += 1
                elif key == ord("r"):
                    # Redo this segment: drop the two points just placed and
                    # click again, keeping the same name and index.
                    state.custom_segment_pending = []
                    state.custom_segment_awaiting_confirm = False
                elif key == ord("x"):
                    if confirm_discard():
                        state.custom_segment_mode = False
                        return None, None, "discard"
                elif key == ord("s"):
                    state.custom_segment_mode = False
                    review_action_panel.hide()
                    close_review_windows()
                    return None, None, "settings"
                if state.custom_segment_index >= len(state.custom_segment_queue):
                    state.custom_segment_mode = False
                    start_next_post_confirm_phase()
                continue

            if key in (ord("n"), ord("k")) and state.custom_segment_index < len(state.custom_segment_queue):
                # Skip this segment: not every configured feature is visible in
                # every frame.
                state.custom_segment_index += 1
                state.custom_segment_pending = []
            elif key == ord("x"):
                if confirm_discard():
                    state.custom_segment_mode = False
                    return None, None, "discard"
            elif key == ord("s"):
                state.custom_segment_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            if state.custom_segment_index >= len(state.custom_segment_queue):
                state.custom_segment_mode = False
                start_next_post_confirm_phase()
            continue

        if state.polygon_edit_mode and state.polygon_edit_confirmed is None:
            # The step's first question. With no automatic mask (see
            # mask_final) it becomes "draw one from scratch?" instead of
            # "correct the automatic one?", with the same yes/no answer.
            if key == ord("y"):
                state.polygon_edit_confirmed = True
                if mask_final is None:
                    state.polygon_drawing_from_scratch = True
                    state.polygon_scratch_building = True
                    state.dataset_polygon = []
            elif key in (ord("n"), ord("k")):
                state.polygon_edit_confirmed = False
                if key == ord("k"):
                    state.polygon_edit_skipped = True
                state.polygon_edit_mode = False
                apply_polygon_edit_to_mask()
                enter_other_whale_polygon_phase()
            elif key == ord("x"):
                if confirm_discard():
                    state.polygon_edit_mode = False
                    return None, None, "discard"
            elif key == ord("s"):
                state.polygon_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.polygon_edit_mode and state.polygon_scratch_building:
            # Still placing the outline's points, with no automatic mask to
            # start from. Each click adds one (see mouse_callback); only closing
            # and undoing are handled here.
            if key in (32, 13, ord("k")) and len(state.dataset_polygon or []) >= 3:
                state.polygon_scratch_building = False
                state.dragging_polygon_index = None
            elif key == ord("r"):
                state.dataset_polygon = []
            elif key == ord("x"):
                if confirm_discard():
                    state.polygon_edit_mode = False
                    return None, None, "discard"
            elif key == ord("s"):
                state.polygon_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.polygon_edit_mode:
            if key in (32, 13, ord("k")):
                # K confirms the current polygon and moves on like Enter, but
                # marks the step skipped: the mask is not added to the YOLO-Seg
                # dataset (see save_training_mask below), only used for this
                # frame's measurement.
                if key == ord("k"):
                    state.polygon_edit_skipped = True
                state.polygon_edit_mode = False
                apply_polygon_edit_to_mask()
                enter_other_whale_polygon_phase()
            elif key == ord("r"):
                if state.polygon_drawing_from_scratch:
                    # Start the drawing again from nothing: there is no
                    # automatic mask to regenerate from.
                    state.polygon_scratch_building = True
                    state.dataset_polygon = []
                    state.dragging_polygon_index = None
                else:
                    # Go back to the automatically generated polygon, throwing
                    # away this frame's edits so far.
                    state.dataset_polygon = None
                    state.dataset_polygon_generation = None
            elif key == ord("x"):
                if confirm_discard():
                    state.polygon_edit_mode = False
                    return None, None, "discard"
            elif key == ord("s"):
                state.polygon_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.other_whale_edit_mode and state.other_whale_confirmed is None:
            # The same logic as the dataset polygon block above, applied to
            # ANOTHER whale's outline. With no automatic mask (an animal added by
            # hand, see start_manual_other_whale) the question becomes "draw it
            # from scratch?" instead of "correct the automatic one?".
            if key == ord("y"):
                state.other_whale_confirmed = True
                if state.other_whale_mask is None:
                    state.other_whale_drawing_from_scratch = True
                    state.other_whale_scratch_building = True
                    state.other_whale_polygon = []
            elif key in (ord("n"), ord("k")):
                if state.other_whale_mask is not None:
                    # No or K: the automatic outline is kept as it is. This is
                    # not a rejection; other_whale_polygon already holds the
                    # unmodified automatic version.
                    state.other_whale_results[state.active_other_whale_id] = state.other_whale_polygon
                    state.other_whale_edit_mode = False
                    enter_other_whale_landmark_step()
                else:
                    # An animal added by hand and cancelled straight away:
                    # nothing to save and no landmarks to offer.
                    state.other_whale_edit_mode = False
                    _advance_other_whale_queue()
            elif key == ord("s"):
                state.other_whale_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.other_whale_edit_mode and state.other_whale_scratch_building:
            # Still placing the outline's points for an animal added by hand,
            # with no automatic mask to start from. Each click adds one (see
            # mouse_callback); only closing and undoing are handled here.
            if key in (32, 13, ord("k")) and len(state.other_whale_polygon or []) >= 3:
                state.other_whale_scratch_building = False
                state.dragging_other_whale_polygon_index = None
            elif key == ord("r"):
                state.other_whale_polygon = []
            elif key == ord("s"):
                state.other_whale_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.other_whale_edit_mode:
            if key in (32, 13, ord("k")):
                finalize_other_whale_polygon()
            elif key == ord("r"):
                if state.other_whale_drawing_from_scratch:
                    # Start the drawing again from nothing: there is no
                    # automatic mask to regenerate from.
                    state.other_whale_scratch_building = True
                    state.other_whale_polygon = []
                    state.dragging_other_whale_polygon_index = None
                else:
                    # Go back to the automatically generated polygon, throwing
                    # away this animal's edits so far.
                    state.other_whale_polygon = None
                    state.other_whale_polygon_epsilon = None
            elif key == ord("s"):
                state.other_whale_edit_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.other_whale_landmark_mode and state.other_whale_landmark_confirmed is None:
            # The same yes/no question as the outline above, for that animal's
            # landmarks. Only on Yes does the landmark panel open: the same one
            # used for the main animal, restarted with THIS animal's automatic
            # head and fluke points.
            if key == ord("y"):
                state.other_whale_landmark_confirmed = True
                if _landmark_panel is None:
                    _landmark_panel = LandmarkControlPanel(
                        review_action_panel.landmark_host,
                        review_action_panel=review_action_panel,
                    )
                    review_action_panel.set_landmark_panel(_landmark_panel)
                _landmark_panel.bind_state(state)
                _landmark_panel.start(
                    state.other_whale_landmark_head,
                    state.other_whale_landmark_fluke,
                    editable_head_fluke=True,
                )
                state.landmark_mode = True
                state.doing_other_whale_landmarks = True
            elif key in (ord("n"), ord("k")):
                finish_other_whale_landmark_step(None)
            elif key == ord("s"):
                state.other_whale_landmark_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if state.other_whale_manual_prompt_mode and state.other_whale_manual_confirmed is None:
            # Asked after each animal in the automatic queue, or straight away
            # if there was none: is there another whale in this frame that
            # screening never detected or accepted?
            if key == ord("y"):
                state.other_whale_manual_confirmed = True
                from tkinter import simpledialog

                # next_manual_whale_id is computed once per frame (in
                # run_review). When more than one animal is added to the same
                # frame, the suggestion also accounts for the ones added so far,
                # not only the first.
                suggested_id = max(
                    [next_manual_whale_id - 1] + state.other_whale_manual_added
                ) + 1
                whale_id = simpledialog.askinteger(
                    "Add another whale",
                    "ID for this whale (must be different from the others "
                    "already used in this video):",
                    initialvalue=suggested_id,
                    parent=review_action_panel.root,
                )
                state.other_whale_manual_prompt_mode = False
                if whale_id is None:
                    # Cancelled from the dialog: treated as answering No, with
                    # no animal added.
                    start_quality_phase()
                else:
                    start_manual_other_whale(whale_id)
            elif key in (ord("n"), ord("k")):
                state.other_whale_manual_confirmed = False
                state.other_whale_manual_prompt_mode = False
                start_quality_phase()
            elif key == ord("s"):
                state.other_whale_manual_prompt_mode = False
                review_action_panel.hide()
                close_review_windows()
                return None, None, "settings"
            continue

        if key in (32, 13, ord("k")):
            if pending_resegment_at is not None:
                state.sam_dirty = resegment_enabled
                pending_resegment_at = None
                continue
            if state.is_measure_mode and len(state.manual_measure_points) < 3:
                print("Warning: at least 3 points (Head, Fluke and one intermediate) are needed before accepting!")
                continue

            # The dataset polygon and quality steps have already run (the first
            # two, see the block just before the while True). All that is
            # confirmed here is the measurement on the mask they produced.
            finalize_measurement()
            continue

        elif key == ord('x'):
            if confirm_discard():
                return None, None, 'discard'

        elif key == ord('s'):
            review_action_panel.hide()
            close_review_windows()
            return None, None, 'settings'

        elif key == ord('r'):
            state.drawing_points, state.drawing_labels, state.manual_measure_points = [], [], []
            state.is_width_edit_mode = False
            state.dragged_pt_info = None
            state.sam_dirty = resegment_enabled

        elif key == ord('z'):
            if state.is_measure_mode and state.manual_measure_points:
                state.manual_measure_points.pop()
                if len(state.manual_measure_points) >= 3 and mask_final is not None:
                    state.active_widths_cache = measure_widths_along_curve(
                        mask_final, measuring_line_points(state), gsd
                    )
                else:
                    state.active_widths_cache = []
                    state.is_width_edit_mode = False
            elif state.mask_correction_enabled and state.drawing_points:
                state.drawing_points.pop()
                state.drawing_labels.pop()
                state.sam_dirty = True

        elif key == ord('h'):
            show_help = not show_help

        elif key == ord('m'):
            toggle_manual_measurement(state)

        elif key == ord('w'):
            toggle_width_editing(state)


def run_review(
    final_df,
    config,
    video_path,
    paths,
    previously_accepted=0,
    previously_accepted_per_second=None,
    subject_label=None,
    include_whale_suffix=False,
    full_candidate_pool=None,
):
    """Phase 3: review each candidate by hand and write the results to final_df.

    Also returns `decided`, a map `_candidate_id` -> "accepted"/"discarded" for
    every candidate actually shown to the user in this round (used to lock them
    if the user goes back to the timeline for more), and an updated
    `accepted_per_second` to pass into the next round so the per-second limit
    holds across the whole session, not just one round.

    `full_candidate_pool` (optional) is EVERY candidate of EVERY animal found
    during screening, not only this round's or this animal's. It is used solely
    to label the OTHER whales that happen to share a frame with the one under
    review, in the exported training image (see other_whale_instance_lines in
    export.py). Without it, a second animal in the same frame would be visible
    but unlabelled: a false negative for training.

    `subject_label` (e.g. "Whale 1/2") appears in the window title when the
    video has more than one animal. `include_whale_suffix` adds the animal's ID
    to the names of exported frames and masks, which is needed only with
    several whales, since two animals can share a frame number and would
    otherwise overwrite each other.
    """
    from pathlib import Path

    import pandas as pd

    from .export import (
        build_annotated_result_frame,
        other_whale_instance_lines,
        reconstruct_mask_from_crop,
        save_yolo_seg_training_example,
    )
    from .landmarks import (
        load_existing_landmark_records,
        make_landmark_record,
        write_landmark_csv,
    )
    from .measurement import (
        extract_detection_mask,
        sanitize_name,
        select_detection_near_reference,
    )

    global _review_action_panel, _landmark_panel

    measure_in_meters = config.measurement_unit == "Meters"
    unit_label = "meters" if measure_in_meters else "pixels"

    final_df = final_df.copy()
    # Sorted by position in the file, not by the shuffled order the candidates
    # were offered in. Otherwise each frame is read by jumping to a random point
    # in the video, which for a compressed file forces the decoder back to the
    # nearest keyframe and forward from there: far more data read from disk than
    # needed, and heavy bursty I/O on external drives (especially bus-powered
    # ones) for the whole review. Which candidates get reviewed does not change,
    # only the order.
    final_df = final_df.sort_values("frame_pos")
    if "_screening_quartile" in final_df.columns:
        final_df["screening_quartile"] = final_df["_screening_quartile"]
    final_df["measured_length"] = np.nan
    final_df["measured_length_px"] = np.nan
    final_df["review_contrast"] = np.nan
    final_df["review_brightness"] = np.nan
    final_df["review_clahe"] = False
    final_df["review_imgsz"] = np.nan
    for attribute in quality.ATTRIBUTES:
        final_df[f"quality_{attribute.key}"] = np.nan
    final_df["quality_verdict"] = ""
    validation_columns = {
        "_human_measurement_mode": "",
        "_model_auto_length_m": np.nan,
    }
    for point_name in ("head", "fluke"):
        validation_columns[f"_human_{point_name}_x"] = np.nan
        validation_columns[f"_human_{point_name}_y"] = np.nan
        validation_columns[f"_model_auto_{point_name}_x"] = np.nan
        validation_columns[f"_model_auto_{point_name}_y"] = np.nan
    for pct in range(5, 100, 5):
        validation_columns[f"width_{pct}pct"] = np.nan
        validation_columns[f"width_{pct}pct_px"] = np.nan
        for source_name in ("human", "model_auto"):
            validation_columns[f"_{source_name}_width_{pct}pct_m"] = np.nan
            validation_columns[f"_{source_name}_width_{pct}pct_top_x"] = np.nan
            validation_columns[f"_{source_name}_width_{pct}pct_top_y"] = np.nan
            validation_columns[f"_{source_name}_width_{pct}pct_bottom_x"] = np.nan
            validation_columns[f"_{source_name}_width_{pct}pct_bottom_y"] = np.nan
    final_df = pd.concat(
        [
            final_df,
            pd.DataFrame(validation_columns, index=final_df.index),
        ],
        axis=1,
    )

    print("PHASE 3: Interactive Review...")
    yolo_model = device.get_yolo_model()
    cap = open_capture(video_path, sequential=use_sequential_indexing(config))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen the video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    def read_frame_near(target_frame):
        """Read a frame, skipping a costly seek when the target is close to the
        current one, which is common now that final_df is sorted by position in
        the file."""
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gap = target_frame - current
        if 0 <= gap <= max(1, int(video_fps * 2)):
            for _ in range(gap):
                if not cap.grab():
                    return False, None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return cap.read()

    accepted_count = previously_accepted
    accepted_per_second = dict(previously_accepted_per_second or {})
    restart_requested = False
    decided = {}
    landmark_records = load_existing_landmark_records(paths.landmark_csv_path)

    def build_other_whale_queue(frame_pos, exclude_whale_id, frame_shape):
        """Build the queue of OTHER whales sharing a frame with the one under
        review.

        One entry {"whale_id", "mask", "head_point", "fluke_point"} per animal
        that has an automatic screening mask. They are offered for hand
        correction before the training dataset is exported (see
        other_whale_edit_mode in refine_mask_interactive).

        The whole mask is passed, not an already simplified polygon:
        sync_other_whale_polygon extracts the polygon on demand, so it follows
        the detail slider like the main animal's instead of being stuck at one
        simplification computed here.

        head_point and fluke_point come from the screening PCA on THAT animal's
        mask, or are None if it found neither, in which case that animal's
        landmark step is skipped for want of a starting point.
        """
        # Diagnostics: without a line here there is no way to tell "no other
        # animal in this exact frame" (normal, and true of most frames) from a
        # bug that hides the step by mistake.
        if full_candidate_pool is None:
            print("  > Other-whale check: full_candidate_pool not passed to this round "
                  "(multi-whale off?).")
            return []
        if not len(full_candidate_pool):
            print("  > Other-whale check: full_candidate_pool is empty.")
            return []
        if (
            "whale_id" not in full_candidate_pool.columns
            or "_auto_mask_crop" not in full_candidate_pool.columns
        ):
            print("  > Other-whale check: the pool has no whale_id/_auto_mask_crop columns.")
            return []
        same_frame = full_candidate_pool[
            (full_candidate_pool["frame_pos"] == frame_pos)
            & (full_candidate_pool["whale_id"] != exclude_whale_id)
        ]
        if same_frame.empty:
            other_frames = sorted(
                full_candidate_pool.loc[
                    full_candidate_pool["whale_id"] != exclude_whale_id, "frame_pos"
                ].unique().tolist()
            )
            nearest = min(other_frames, key=lambda f: abs(f - frame_pos)) if other_frames else None
            print(
                f"  > Other-whale check: no candidate from another animal at frame "
                f"{frame_pos} exactly (whale {exclude_whale_id} under review). "
                f"Nearest frame with another animal: {nearest}."
                if nearest is not None
                else f"  > Other-whale check: no candidate from ANY other animal in the "
                f"whole pool (whale {exclude_whale_id} under review) — only this "
                "animal has valid candidates."
            )
            return []
        frame_h, frame_w = frame_shape[:2]
        queue = []
        seen_whale_ids = set()
        for _, other_row in same_frame.iterrows():
            other_whale_id = other_row.get("whale_id")
            if pd.isna(other_whale_id) or other_whale_id in seen_whale_ids:
                continue
            mask = reconstruct_mask_from_crop(
                other_row.get("_auto_mask_crop"),
                other_row.get("_auto_mask_crop_x1"),
                other_row.get("_auto_mask_crop_y1"),
                frame_h,
                frame_w,
            )
            if not mask.any():
                print(
                    f"  > Other-whale check: whale {int(other_whale_id)} has a candidate "
                    f"at frame {frame_pos} but no usable automatic mask "
                    "(crop missing or invalid) — skipped."
                )
                continue
            head_x, head_y = other_row.get("_auto_head_x"), other_row.get("_auto_head_y")
            notch_x, notch_y = other_row.get("_auto_notch_x"), other_row.get("_auto_notch_y")
            head_point = (
                (float(head_x), float(head_y))
                if pd.notna(head_x) and pd.notna(head_y)
                else None
            )
            fluke_point = (
                (float(notch_x), float(notch_y))
                if pd.notna(notch_x) and pd.notna(notch_y)
                else None
            )
            queue.append(
                {
                    "whale_id": int(other_whale_id),
                    "mask": mask,
                    "head_point": head_point,
                    "fluke_point": fluke_point,
                }
            )
            seen_whale_ids.add(other_whale_id)
        print(
            f"  > Other-whale check: {len(queue)} other animal(s) ready to offer "
            f"at frame {frame_pos} (whale {exclude_whale_id} under review): "
            f"{[entry['whale_id'] for entry in queue]}."
        )
        return queue

    def next_available_whale_id(exclude_whale_id):
        """Suggest an ID for an animal added by hand (see
        other_whale_manual_prompt_mode): a number no known whale_id in this
        video uses, so it cannot be confused with an animal tracked elsewhere."""
        known_ids = {1}
        if pd.notna(exclude_whale_id):
            known_ids.add(int(exclude_whale_id))
        if full_candidate_pool is not None and "whale_id" in full_candidate_pool.columns:
            known_ids.update(
                int(value)
                for value in full_candidate_pool["whale_id"].dropna().unique().tolist()
            )
        return max(known_ids) + 1

    for i, (index, row) in enumerate(final_df.iterrows()):

        if accepted_count >= config.max_accepted:
            print(f"Reached {config.max_accepted} accepted frames — review finished.")
            break

        f_id = int(row['frame_pos'])
        current_sec = int(row['video_second'])

        # Keeps the confirmed frames from clustering in time.
        if accepted_per_second.get(current_sec, 0) >= config.max_frames_per_sec:
            print("Frame skipped")
            continue

        success, frame = read_frame_near(f_id)
        if not success: continue

        initial_mask = None
        review_mask_predictor = None
        review_bbox_predictor = None
        review_sam_model = None
        manual_only = bool(row.get("_manual_only", False))
        bbox_xyxy = np.array(
            [[row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]],
            dtype=np.float32,
        )
        if manual_only:
            # A frame added from the preview with no YOLO detection at all:
            # nothing to point YOLO or SAM at, so go straight to manual
            # measurement. refine_mask_interactive already handles
            # mask_final=None, see set_measure_mode_for_current_mask.
            print(
                f"Frame {f_id}: no automatic detection — "
                "fully manual annotation."
            )
        elif config.pipeline_mode == "YOLO segmentation only":
            def review_mask_predictor(
                adjusted_frame,
                inference_quality=config.review_imgsz,
            ):
                # Run on the whole frame, exactly as screening does, which never
                # crops. Cropping around the bbox changes the apparent scale at
                # which YOLO sees the animal (much more zoomed in than during the
                # scan), and on this model that can degrade the segmentation
                # badly. It is the most likely reason review used to produce
                # different, worse masks than the preview.
                review_result = device.predict_adaptive(
                    yolo_model,
                    adjusted_frame,
                    inference_quality,
                    "YOLO review on the full frame",
                    conf=config.confidence,
                    max_det=5,
                    retina_masks=True,
                )
                detection_index = select_detection_near_reference(
                    review_result,
                    bbox_xyxy,
                )
                if detection_index is None:
                    del review_result
                    device.release_prediction(yolo_model, clear_cache=False)
                    return None
                mask = extract_detection_mask(
                    review_result,
                    detection_index,
                    adjusted_frame.shape,
                )
                del review_result
                device.release_prediction(yolo_model, clear_cache=False)
                return mask

            initial_mask = review_mask_predictor(frame)
            if initial_mask is None:
                print(
                    f"Frame {f_id}: YOLO did not produce a mask; "
                    "the frame will still be shown for manual measurement."
                )
            review_sam_model = None
        else:
            def review_bbox_predictor(
                adjusted_frame,
                inference_quality,
                reference_bbox,
            ):
                review_result = device.predict_adaptive(
                    yolo_model,
                    adjusted_frame,
                    inference_quality,
                    "YOLO new bounding box",
                    conf=config.confidence,
                    max_det=5,
                    retina_masks=False,
                )
                detection_index = select_detection_near_reference(
                    review_result,
                    reference_bbox,
                )
                if detection_index is None:
                    del review_result
                    device.release_prediction(yolo_model, clear_cache=False)
                    return None
                refreshed_bbox = (
                    review_result.boxes.xyxy[detection_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                    .reshape(1, 4)
                )
                del review_result
                device.release_prediction(yolo_model, clear_cache=False)
                return refreshed_bbox

            review_sam_model = device.get_sam_model()

        accepted_mask, info_final, status = refine_mask_interactive(
            frame,
            bbox_xyxy,
            review_sam_model,
            i + 1,
            len(final_df),
            row["gsd"],
            config,
            initial_mask=initial_mask,
            mask_predictor=review_mask_predictor,
            bbox_predictor=review_bbox_predictor,
            initial_mask_attempted=(
                manual_only or config.pipeline_mode == "YOLO segmentation only"
            ),
            accepted_count=accepted_count,
            max_accepted=config.max_accepted,
            subject_label=subject_label,
            other_whale_queue_source=build_other_whale_queue(
                f_id, row.get("whale_id"), frame.shape
            ),
            next_manual_whale_id=next_available_whale_id(row.get("whale_id")),
        )

        if status == 'settings':
            print("Review stopped: returning to settings.")
            restart_requested = True
            del accepted_mask, info_final, frame
            break

        candidate_id = row.get("_candidate_id", index)

        if status == 'discard':
            decided[candidate_id] = 'discarded'

        if status == 'ok' and info_final:
            (
                lung_m,
                spine,
                widths,
                is_man,
                review_settings,
                landmark_points,
                automatic_measurement,
            ) = info_final
            # The Christiansen criteria no longer block the measurement or its
            # count (see finalize_quality_and_proceed): on some flights every
            # frame is "poor" by those criteria and the user still wants them
            # measured and kept. The verdict is still written to the CSV
            # (quality_verdict, below) for reference.
            decided[candidate_id] = 'accepted'
            # --- Counters are only updated now that the frame is confirmed ---
            accepted_count += 1
            accepted_per_second[current_sec] = accepted_per_second.get(current_sec, 0) + 1

            # Add the animal ID suffix only when needed: two whales can each
            # have a candidate on the same frame number, and without it the
            # second one saved would overwrite the first, both the annotated PNG
            # and its row in the landmark CSV, which is keyed on this file name.
            whale_suffix = (
                f"_W{row['whale_id']}"
                if include_whale_suffix and pd.notna(row.get("whale_id"))
                else ""
            )
            landmark_frame_name = (
                sanitize_name(
                    f"{Path(video_path).stem}_F{f_id}_{paths.flight_date_label}"
                    f"{whale_suffix}"
                )
                + ".png"
            )
            annotated_frame = build_annotated_result_frame(
                frame,
                accepted_mask,
                spine,
                widths,
                landmark_points,
            )
            annotated_path = paths.frame_output_folder / landmark_frame_name
            if not cv2.imwrite(str(annotated_path), annotated_frame):
                print(
                    f"Warning: could not save the annotated frame "
                    f"{annotated_path}."
                )

            # Hand-corrected outlines (or ones accepted as-is) for ANY animal in
            # this frame, the main one included. They are preferred over the raw
            # automatic mask when labelling the OTHER animals in each one's
            # image. See other_whale_edit_mode and other_whale_instance_lines.
            all_confirmed_polygons = dict(review_settings.get("other_whale_polygons") or {})
            primary_whale_id = row.get("whale_id")
            primary_dataset_polygon = review_settings.get("dataset_polygon")
            if (
                pd.notna(primary_whale_id)
                and primary_dataset_polygon
                and len(primary_dataset_polygon) >= 3
            ):
                all_confirmed_polygons[int(primary_whale_id)] = primary_dataset_polygon

            if review_settings.get("save_training_mask"):
                training_saved = save_yolo_seg_training_example(
                    frame,
                    accepted_mask,
                    landmark_frame_name,
                    paths.mask_database_folder,
                    review_settings.get("polygon_epsilon_fraction", 0.0005),
                    review_settings.get("dataset_polygon"),
                    extra_instance_lines=other_whale_instance_lines(
                        full_candidate_pool,
                        f_id,
                        row.get("whale_id"),
                        frame.shape,
                        review_settings.get("polygon_epsilon_fraction", 0.0005),
                        confirmed_polygons=all_confirmed_polygons,
                    ),
                )
                if training_saved:
                    print(
                        f"YOLO-Seg example saved: "
                        f"{paths.mask_database_folder / 'images' / landmark_frame_name}"
                    )
                else:
                    print(
                        "Mask selected for the database, but not available "
                        "or without a valid contour: example not saved."
                    )

            landmark_records[landmark_frame_name] = make_landmark_record(
                landmark_frame_name,
                landmark_points,
                paths.video_dataset_label,
            )

            # Landmarks for the OTHER whales in the same frame (see
            # other_whale_landmark_mode). Unlike the polygon and mask, which go
            # in as EXTRA instances in the SAME image as the main animal,
            # landmarks follow the DeepLabCut format of one row per image. So
            # each other animal needs its own image and row: the same frame, the
            # same pixels, but ITS label and ITS landmarks.
            for other_whale_id, other_landmark_points in (
                review_settings.get("other_whale_landmarks") or {}
            ).items():
                if not other_landmark_points:
                    continue
                other_whale_frame_name = (
                    sanitize_name(
                        f"{Path(video_path).stem}_F{f_id}_{paths.flight_date_label}"
                        f"_W{other_whale_id}"
                    )
                    + ".png"
                )
                if review_settings.get("save_training_mask"):
                    other_polygon = all_confirmed_polygons.get(other_whale_id)
                    if other_polygon and len(other_polygon) >= 3:
                        other_training_saved = save_yolo_seg_training_example(
                            frame,
                            accepted_mask,
                            other_whale_frame_name,
                            paths.mask_database_folder,
                            review_settings.get("polygon_epsilon_fraction", 0.0005),
                            other_polygon,
                            extra_instance_lines=other_whale_instance_lines(
                                full_candidate_pool,
                                f_id,
                                other_whale_id,
                                frame.shape,
                                review_settings.get("polygon_epsilon_fraction", 0.0005),
                                confirmed_polygons=all_confirmed_polygons,
                            ),
                        )
                        if other_training_saved:
                            print(
                                f"YOLO-Seg example saved (whale {other_whale_id}): "
                                f"{paths.mask_database_folder / 'images' / other_whale_frame_name}"
                            )
                landmark_records[other_whale_frame_name] = make_landmark_record(
                    other_whale_frame_name,
                    other_landmark_points,
                    paths.video_dataset_label,
                )

            write_landmark_csv(landmark_records, paths.landmark_csv_path)

            print(
                f"Length in {unit_label}: {lung_m:.2f} "
                f"(accepted: {accepted_count}/{config.max_accepted})"
            )
            final_df.at[index, "measured_length"] = lung_m
            final_df.at[index, "measured_length_px"] = lung_m / row['gsd']
            final_df.at[index, "review_contrast"] = review_settings["contrast"]
            final_df.at[index, "review_brightness"] = review_settings["brightness"]
            final_df.at[index, "review_clahe"] = review_settings["clahe"]
            final_df.at[index, "review_imgsz"] = review_settings["imgsz"]
            # The verdict has to be written even when quality_scores is empty,
            # for instance when K skipped the quality step before any score was
            # given. Otherwise "skipped" never reached the CSV.
            for attribute_key, score in review_settings.get("quality_scores", {}).items():
                final_df.at[index, f"quality_{attribute_key}"] = score
            final_df.at[index, "quality_verdict"] = review_settings.get(
                "quality_verdict", "passed"
            )
            segment_name_counts = {}
            for segment_index, segment in enumerate(review_settings.get("custom_segments", [])):
                base_name = sanitize_name(
                    segment.get("name") or f"segment_{segment_index + 1}"
                )
                # Names repeated in one frame (the user did not change the name
                # field between segments) are numbered rather than overwriting
                # each other in the dataframe.
                segment_name_counts[base_name] = segment_name_counts.get(base_name, 0) + 1
                column_name = (
                    base_name
                    if segment_name_counts[base_name] == 1
                    else f"{base_name}_{segment_name_counts[base_name]}"
                )
                final_df.at[index, f"custom_segment_{column_name}_m"] = (
                    segment["length_m"]
                )
                final_df.at[index, f"custom_segment_{column_name}_px"] = (
                    segment["length_m"] / row["gsd"]
                )
                # Float, not int(): these are reference columns for the
                # automatic-versus-human comparison, and rounding would add up to
                # about a pixel of avoidable noise to it.
                segment_point_a = (float(segment["pt_a"][0]), float(segment["pt_a"][1]))
                segment_point_b = (float(segment["pt_b"][0]), float(segment["pt_b"][1]))
                final_df.at[index, f"custom_segment_{column_name}_pt_a_x"] = segment_point_a[0]
                final_df.at[index, f"custom_segment_{column_name}_pt_a_y"] = segment_point_a[1]
                final_df.at[index, f"custom_segment_{column_name}_pt_b_x"] = segment_point_b[0]
                final_df.at[index, f"custom_segment_{column_name}_pt_b_y"] = segment_point_b[1]
            final_df.at[index, "_human_measurement_mode"] = (
                "manual" if is_man else "automatic_reviewed"
            )
            # finalize_measurement always guarantees a non-empty spine, but not
            # assuming it here costs nothing.
            if spine:
                human_head = (float(spine[0][0]), float(spine[0][1]))
                human_fluke = (float(spine[-1][0]), float(spine[-1][1]))
                final_df.at[index, "_human_head_x"] = human_head[0]
                final_df.at[index, "_human_head_y"] = human_head[1]
                final_df.at[index, "_human_fluke_x"] = human_fluke[0]
                final_df.at[index, "_human_fluke_y"] = human_fluke[1]
            for w in widths:
                pct = int(w["pct"])
                final_df.at[index, f"width_{pct}pct"] = w["width_m"]
                final_df.at[index, f"width_{pct}pct_px"] = (
                    w["width_m"] / row["gsd"]
                )
                human_top = (float(w["pt_top"][0]), float(w["pt_top"][1]))
                human_bottom = (float(w["pt_bot"][0]), float(w["pt_bot"][1]))
                final_df.at[index, f"_human_width_{pct}pct_m"] = w["width_m"]
                final_df.at[index, f"_human_width_{pct}pct_top_x"] = human_top[0]
                final_df.at[index, f"_human_width_{pct}pct_top_y"] = human_top[1]
                final_df.at[index, f"_human_width_{pct}pct_bottom_x"] = human_bottom[0]
                final_df.at[index, f"_human_width_{pct}pct_bottom_y"] = human_bottom[1]
            if automatic_measurement is not None:
                final_df.at[index, "_model_auto_length_m"] = (
                    automatic_measurement["length_m"]
                )
                auto_head = automatic_measurement["head"]
                auto_fluke = automatic_measurement["fluke"]
                final_df.at[index, "_model_auto_head_x"] = auto_head[0]
                final_df.at[index, "_model_auto_head_y"] = auto_head[1]
                final_df.at[index, "_model_auto_fluke_x"] = auto_fluke[0]
                final_df.at[index, "_model_auto_fluke_y"] = auto_fluke[1]
                for pct, auto_width in automatic_measurement["widths"].items():
                    auto_top = auto_width["pt_top"]
                    auto_bottom = auto_width["pt_bot"]
                    final_df.at[index, f"_model_auto_width_{pct}pct_m"] = (
                        auto_width["width_m"]
                    )
                    final_df.at[
                        index,
                        f"_model_auto_width_{pct}pct_top_x",
                    ] = auto_top[0]
                    final_df.at[
                        index,
                        f"_model_auto_width_{pct}pct_top_y",
                    ] = auto_top[1]
                    final_df.at[
                        index,
                        f"_model_auto_width_{pct}pct_bottom_x",
                    ] = auto_bottom[0]
                    final_df.at[
                        index,
                        f"_model_auto_width_{pct}pct_bottom_y",
                    ] = auto_bottom[1]
            del annotated_frame

        del accepted_mask, info_final, frame
        device.clear_accelerator_memory()

    cap.release()
    cv2.destroyAllWindows()
    if _review_action_panel is not None:
        _review_action_panel.hide()
    if _landmark_panel is not None:
        _landmark_panel.hide()
    device.clear_accelerator_memory()

    return final_df, restart_requested, decided, accepted_per_second
