"""geometry functions"""

import math
import re

import cv2
import numpy as np


def bezier_curve_points(control_points, num_samples=40):
    """Bezier curve through the manual points used as control polygon.

    Degree is (number of points - 1); the curve is sampled at `num_samples`
    points using the Bernstein form.
    """
    pts = np.asarray(control_points, dtype=np.float64)
    n = len(pts) - 1
    if n < 1:
        return pts.copy()
    t = np.linspace(0.0, 1.0, num_samples)[:, None]
    i = np.arange(n + 1)
    coeffs = np.array([math.comb(n, k) for k in i], dtype=np.float64)
    basis = coeffs * (t ** i) * ((1 - t) ** (n - i))
    return basis @ pts


def manual_measure_length_px(points_list, use_bezier=False):
    """Length in pixels of the manual measurement line.

    Measured along the polyline, or along the Bezier curve fitted to the same
    points if that option is on.
    """
    if len(points_list) < 2:
        return 0.0
    if use_bezier and len(points_list) >= 3:
        curve = bezier_curve_points(points_list)
        diffs = curve[1:] - curve[:-1]
        return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
    total = 0.0
    for i in range(len(points_list) - 1):
        p1, p2 = points_list[i], points_list[i + 1]
        total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return total


def measure_widths_along_curve(mask_uint8, points_list, gsd):
    if len(points_list) < 2: return []
    segments = []
    total_length = 0.0
    for i in range(len(points_list) - 1):
        p1 = points_list[i]
        p2 = points_list[i+1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist > 0:
            segments.append({'p1': p1, 'p2': p2, 'length': dist, 'accumulated': total_length})
            total_length += dist
    if total_length == 0: return []
    widths_data = []
    h, w = mask_uint8.shape

    def cast_ray(start_x, start_y, dir_x, dir_y, step=1):
        curr_x, curr_y = float(start_x), float(start_y)
        while True:
            next_x = curr_x + dir_x * step
            next_y = curr_y + dir_y * step
            if not (0 <= int(next_x) < w and 0 <= int(next_y) < h): break
            if mask_uint8[int(next_y), int(next_x)] == 0: break
            curr_x, curr_y = next_x, next_y
        return (int(curr_x), int(curr_y))

    for pct in range(5, 100, 5):
        target_dist = total_length * (pct / 100.0)
        current_seg = None
        for seg in segments:
            if seg['accumulated'] <= target_dist <= (seg['accumulated'] + seg['length']):
                current_seg = seg
                break
        if current_seg is None: current_seg = segments[-1]
        p1 = current_seg['p1']
        p2 = current_seg['p2']
        seg_len = current_seg['length']
        local_dist = target_dist - current_seg['accumulated']
        t = local_dist / seg_len
        cx = p1[0] + t * (p2[0] - p1[0])
        cy = p1[1] + t * (p2[1] - p1[1])
        dx = (p2[0] - p1[0]) / seg_len
        dy = (p2[1] - p1[1]) / seg_len
        nx, ny = -dy, dx
        if not (0 <= int(cx) < w and 0 <= int(cy) < h) or mask_uint8[int(cy), int(cx)] == 0:
            continue
        pt_edge1 = cast_ray(cx, cy, nx, ny)
        pt_edge2 = cast_ray(cx, cy, -nx, -ny)
        width_px = math.hypot(pt_edge2[0] - pt_edge1[0], pt_edge2[1] - pt_edge1[1])
        widths_data.append({
            'pct': pct, 'pt_top': pt_edge1, 'pt_bot': pt_edge2, 'width_m': width_px * gsd
        })
    return widths_data


def measure_length_pca(mask_uint8, gsd):
    y_coords, x_coords = np.nonzero(mask_uint8)
    if len(x_coords) < 25: return None
    pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
    pca_result = cv2.PCACompute(pts, mean=None)
    mean, eigenvectors = pca_result[0], pca_result[1]
    angle_deg = np.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w = mask_uint8.shape
    M_rot = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), angle_deg, 1.0)
    M_rev = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), -angle_deg, 1.0)
    mask_rot = cv2.warpAffine(mask_uint8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_r, x_r = np.nonzero(mask_rot)
    if len(x_r) == 0: return None
    x_min, x_max, y_min, y_max = np.min(x_r), np.max(x_r), np.min(y_r), np.max(y_r)
    img_crop = mask_rot[y_min:y_max+1, x_min:x_max+1]
    w_c = img_crop.shape[1]
    is_flipped = False
    if np.sum(img_crop[:, :max(1, int(w_c * 0.10))] > 0) > np.sum(img_crop[:, -max(1, int(w_c * 0.10)):] > 0):
        img_crop = cv2.flip(img_crop, 1)
        is_flipped = True

    def to_rotated_coords(p_crop):
        return (x_min + p_crop[0] if not is_flipped else x_min + (w_c - 1 - p_crop[0]), y_min + p_crop[1])
    def to_original_coords(p_rot):
        return tuple(M_rev.dot(np.array([p_rot[0], p_rot[1], 1]))[:2].astype(int))

    first_x = np.min(np.nonzero(img_crop)[1])
    p_head_crop = (first_x, int(np.mean(np.where(img_crop[:, first_x] > 0)[0])))

    def get_column_info(x_pos):
        if x_pos >= w_c or x_pos < 0: return True, None
        y_pix = np.where(img_crop[:, x_pos] > 0)[0]
        if len(y_pix) < 2: return True, None
        max_gap, gap_center = 0, None
        for i in range(len(y_pix) - 1):
            if y_pix[i+1] - y_pix[i] > max_gap:
                max_gap, gap_center = y_pix[i+1] - y_pix[i], (y_pix[i] + y_pix[i+1]) // 2
        if max_gap <= 5:
            return False, None
        # The gap has to sit roughly halfway down the column (between 25% and
        # 75%). A gap near the top or bottom edge is more likely a flaw in the
        # outline than the real notch, which is centred between the two flukes.
        column_span = y_pix[-1] - y_pix[0]
        if column_span <= 0:
            return False, None
        relative_position = (gap_center - y_pix[0]) / column_span
        if not (0.25 <= relative_position <= 0.75):
            return False, None
        return True, gap_center
    look_ahead = max(1, int(w_c * 0.05))
    p_notch_crop = None
    for x in range(w_c - 2, look_ahead, -1):
        has_hole_curr, _ = get_column_info(x)
        if not has_hole_curr:
            check_ahead = True
            for next_x in range(x - look_ahead, x):
                if get_column_info(next_x)[0]: check_ahead = False; break
            if check_ahead:
                has_hole_prev, gap_center_prev = get_column_info(x + 1)
                if has_hole_prev and gap_center_prev is not None:
                    p_notch_crop = (x, gap_center_prev)
                    break
    if p_notch_crop is None:
        # No gap -> no-gap transition with a centred, deep enough gap: return
        # no automatic measurement for this frame rather than invent a notch
        # by falling back on the last column.
        return None

    widths_data = []
    dist_x_crop = p_notch_crop[0] - p_head_crop[0]
    for pct in range(5, 100, 5):
        target_x = p_head_crop[0] + int(dist_x_crop * (pct / 100.0))
        if 0 <= target_x < w_c:
            y_pix = np.where(img_crop[:, target_x] > 0)[0]
            if len(y_pix) > 0:
                pt_top_orig = to_original_coords(to_rotated_coords((target_x, np.min(y_pix))))
                pt_bot_orig = to_original_coords(to_rotated_coords((target_x, np.max(y_pix))))
                widths_data.append({'pct': pct, 'pt_top': pt_top_orig, 'pt_bot': pt_bot_orig, 'width_m': (np.max(y_pix) - np.min(y_pix)) * gsd})

    p_head_orig = to_original_coords(to_rotated_coords(p_head_crop))
    p_notch_orig = to_original_coords(to_rotated_coords(p_notch_crop))
    lung_m = math.hypot(p_notch_crop[0] - p_head_crop[0], p_notch_crop[1] - p_head_crop[1]) * gsd
    return lung_m, p_head_orig, p_notch_orig, widths_data


def measure_length_pca_along_axis(mask_uint8, gsd):
    """Length measured along the body axis instead of head-to-notch directly.

    Head and tail notch are found exactly as in `measure_length_pca`. The
    difference is the length itself: the distance along the main (PCA) axis
    between the head and the line through the notch perpendicular to that
    axis, not the straight-line distance between the two points.

    In the rotated mask the body axis is horizontal, so that perpendicular line
    is vertical (x = the notch's x) and the shortest link from the head is
    purely horizontal. The length then depends only on how far along the body
    the tail ends, never on the notch's y, which is the least reliable part of
    the gap-based notch search.
    """
    y_coords, x_coords = np.nonzero(mask_uint8)
    if len(x_coords) < 25: return None
    pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
    pca_result = cv2.PCACompute(pts, mean=None)
    mean, eigenvectors = pca_result[0], pca_result[1]
    angle_deg = np.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w = mask_uint8.shape
    M_rot = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), angle_deg, 1.0)
    M_rev = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), -angle_deg, 1.0)
    mask_rot = cv2.warpAffine(mask_uint8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_r, x_r = np.nonzero(mask_rot)
    if len(x_r) == 0: return None
    x_min, x_max, y_min, y_max = np.min(x_r), np.max(x_r), np.min(y_r), np.max(y_r)
    img_crop = mask_rot[y_min:y_max+1, x_min:x_max+1]
    w_c = img_crop.shape[1]
    is_flipped = False
    if np.sum(img_crop[:, :max(1, int(w_c * 0.10))] > 0) > np.sum(img_crop[:, -max(1, int(w_c * 0.10)):] > 0):
        img_crop = cv2.flip(img_crop, 1)
        is_flipped = True

    def to_rotated_coords(p_crop):
        return (x_min + p_crop[0] if not is_flipped else x_min + (w_c - 1 - p_crop[0]), y_min + p_crop[1])
    def to_original_coords(p_rot):
        return tuple(M_rev.dot(np.array([p_rot[0], p_rot[1], 1]))[:2].astype(int))

    first_x = np.min(np.nonzero(img_crop)[1])
    p_head_crop = (first_x, int(np.mean(np.where(img_crop[:, first_x] > 0)[0])))

    def get_column_info(x_pos):
        if x_pos >= w_c or x_pos < 0: return True, None
        y_pix = np.where(img_crop[:, x_pos] > 0)[0]
        if len(y_pix) < 2: return True, None
        max_gap, gap_center = 0, None
        for i in range(len(y_pix) - 1):
            if y_pix[i+1] - y_pix[i] > max_gap:
                max_gap, gap_center = y_pix[i+1] - y_pix[i], (y_pix[i] + y_pix[i+1]) // 2
        return (max_gap > 3), gap_center
    look_ahead = max(1, int(w_c * 0.05))
    p_notch_crop = (w_c - 1, img_crop.shape[0] // 2)
    for x in range(w_c - 2, look_ahead, -1):
        has_hole_curr, _ = get_column_info(x)
        if not has_hole_curr:
            check_ahead = True
            for next_x in range(x - look_ahead, x):
                if get_column_info(next_x)[0]: check_ahead = False; break
            if check_ahead:
                _, gap_center_prev = get_column_info(x + 1)
                p_notch_crop = (x, gap_center_prev) if gap_center_prev is not None else (x, int(np.mean(np.where(img_crop[:, x] > 0)[0])))
                break

    # Foot of the perpendicular from the head to the line x = notch_x: same x
    # as the notch, same y as the head. It is the point on that line closest to
    # the head, so head-to-foot is the shortest head-to-line distance.
    p_notch_foot_crop = (p_notch_crop[0], p_head_crop[1])

    widths_data = []
    dist_x_crop = p_notch_crop[0] - p_head_crop[0]
    for pct in range(5, 100, 5):
        target_x = p_head_crop[0] + int(dist_x_crop * (pct / 100.0))
        if 0 <= target_x < w_c:
            y_pix = np.where(img_crop[:, target_x] > 0)[0]
            if len(y_pix) > 0:
                pt_top_orig = to_original_coords(to_rotated_coords((target_x, np.min(y_pix))))
                pt_bot_orig = to_original_coords(to_rotated_coords((target_x, np.max(y_pix))))
                widths_data.append({'pct': pct, 'pt_top': pt_top_orig, 'pt_bot': pt_bot_orig, 'width_m': (np.max(y_pix) - np.min(y_pix)) * gsd})

    p_head_orig = to_original_coords(to_rotated_coords(p_head_crop))
    p_notch_orig = to_original_coords(to_rotated_coords(p_notch_foot_crop))
    lung_m = abs(p_notch_crop[0] - p_head_crop[0]) * gsd
    return lung_m, p_head_orig, p_notch_orig, widths_data


def measure_length_pca_tail_concavity(mask_uint8, gsd):
    """Find the tail notch as the deepest dent in the outline.

    The head is found as in the other PCA variants. The notch, instead of being
    a "gap" in a column-by-column scan, is the point of greatest concavity of
    the contour (OpenCV "convexity defects", the same trick used to find the
    valleys between fingers on a hand).

    The column scan assumes the tail produces one specific two-lobe pattern in
    a single column. With imperfect masks (ragged edges, a faint notch, spray)
    it often finds nothing, or finds it in the wrong place. Convexity defects
    instead look for the deepest dent in the real outline, which is what a
    notch actually is. Limiting the search to the rear half of the body keeps
    it from latching onto the dorsal or pectoral fins near the head.

    The final length is still measured along the main axis, as in
    `measure_length_pca_along_axis`: the notch found here is more reliable, but
    its sideways position still should not affect the length.
    """
    y_coords, x_coords = np.nonzero(mask_uint8)
    if len(x_coords) < 25: return None
    pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
    pca_result = cv2.PCACompute(pts, mean=None)
    mean, eigenvectors = pca_result[0], pca_result[1]
    angle_deg = np.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w = mask_uint8.shape
    M_rot = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), angle_deg, 1.0)
    M_rev = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), -angle_deg, 1.0)
    mask_rot = cv2.warpAffine(mask_uint8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_r, x_r = np.nonzero(mask_rot)
    if len(x_r) == 0: return None
    x_min, x_max, y_min, y_max = np.min(x_r), np.max(x_r), np.min(y_r), np.max(y_r)
    img_crop = mask_rot[y_min:y_max+1, x_min:x_max+1]
    w_c = img_crop.shape[1]
    is_flipped = False
    if np.sum(img_crop[:, :max(1, int(w_c * 0.10))] > 0) > np.sum(img_crop[:, -max(1, int(w_c * 0.10)):] > 0):
        img_crop = cv2.flip(img_crop, 1)
        is_flipped = True

    def to_rotated_coords(p_crop):
        return (x_min + p_crop[0] if not is_flipped else x_min + (w_c - 1 - p_crop[0]), y_min + p_crop[1])
    def to_original_coords(p_rot):
        return tuple(M_rev.dot(np.array([p_rot[0], p_rot[1], 1]))[:2].astype(int))

    first_x = np.min(np.nonzero(img_crop)[1])
    p_head_crop = (first_x, int(np.mean(np.where(img_crop[:, first_x] > 0)[0])))

    p_notch_crop = None
    contours, _ = cv2.findContours(img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) >= 3:
                # convexityDefects needs increasing indices with no duplicates.
                hull_indices = np.unique(hull_indices.flatten())[::-1]
                try:
                    defects = cv2.convexityDefects(contour, hull_indices)
                except cv2.error:
                    defects = None
                if defects is not None:
                    # The returned shape depends on the OpenCV version,
                    # (n,1,4) or (n,4); normalize it once here.
                    defects = defects.reshape(-1, 4)
                    tail_region_start = p_head_crop[0] + 0.5 * (w_c - 1 - p_head_crop[0])
                    best_depth = 0
                    for i in range(defects.shape[0]):
                        _, _, far_index, depth = defects[i]
                        far_point = tuple(int(value) for value in contour[far_index][0])
                        if far_point[0] < tail_region_start:
                            continue
                        if depth > best_depth:
                            best_depth = depth
                            p_notch_crop = far_point
    if p_notch_crop is None:
        # Fallback when no dent is found (tail cut off by the frame edge, or
        # too smooth a shape): use the end of the mask, as the other variants do.
        p_notch_crop = (w_c - 1, img_crop.shape[0] // 2)

    p_notch_foot_crop = (p_notch_crop[0], p_head_crop[1])

    widths_data = []
    dist_x_crop = p_notch_crop[0] - p_head_crop[0]
    for pct in range(5, 100, 5):
        target_x = p_head_crop[0] + int(dist_x_crop * (pct / 100.0))
        if 0 <= target_x < w_c:
            y_pix = np.where(img_crop[:, target_x] > 0)[0]
            if len(y_pix) > 0:
                pt_top_orig = to_original_coords(to_rotated_coords((target_x, np.min(y_pix))))
                pt_bot_orig = to_original_coords(to_rotated_coords((target_x, np.max(y_pix))))
                widths_data.append({'pct': pct, 'pt_top': pt_top_orig, 'pt_bot': pt_bot_orig, 'width_m': (np.max(y_pix) - np.min(y_pix)) * gsd})

    p_head_orig = to_original_coords(to_rotated_coords(p_head_crop))
    p_notch_orig = to_original_coords(to_rotated_coords(p_notch_foot_crop))
    lung_m = abs(p_notch_crop[0] - p_head_crop[0]) * gsd
    return lung_m, p_head_orig, p_notch_orig, widths_data


def measure_length_pca_robust_notch(mask_uint8, gsd):
    """Like `measure_length_pca`, but harder to fool with a noisy mask.

    The notch is still found by scanning from the tail towards the body for the
    "gap -> no gap" transition, and the length is still the straight head-to-
    notch distance. Three changes make segmentation noise, which often creates
    small dents mistaken for the real notch, much less of a problem:

    1. the cropped mask is cleaned with a morphological closing (kernel sized
       from the body thickness) BEFORE the notch is looked for: a few stray
       pixels at the edge disappear, while a real notch, much wider, survives;
    2. the minimum gap size scales with the body thickness at that point
       instead of being a fixed 3px that ignores resolution and zoom, because
       a real notch is a steady fraction of the animal's height and leftover
       noise is not;
    3. the confirmation window (how many columns in a row must show no gap
       before the transition counts) is wider, so one narrow isolated gap can
       no longer pass for the notch.
    """
    y_coords, x_coords = np.nonzero(mask_uint8)
    if len(x_coords) < 25: return None
    pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
    pca_result = cv2.PCACompute(pts, mean=None)
    mean, eigenvectors = pca_result[0], pca_result[1]
    angle_deg = np.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w = mask_uint8.shape
    M_rot = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), angle_deg, 1.0)
    M_rev = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), -angle_deg, 1.0)
    mask_rot = cv2.warpAffine(mask_uint8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_r, x_r = np.nonzero(mask_rot)
    if len(x_r) == 0: return None
    x_min, x_max, y_min, y_max = np.min(x_r), np.max(x_r), np.min(y_r), np.max(y_r)
    img_crop = mask_rot[y_min:y_max+1, x_min:x_max+1]

    # Morphological closing: removes small edge defects (segmentation noise)
    # without touching a real notch, which is far wider.
    body_thickness = max(1, int(y_max - y_min))
    close_kernel_size = max(1, int(round(body_thickness * 0.06)))
    if close_kernel_size > 1:
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        img_crop = cv2.morphologyEx(img_crop, cv2.MORPH_CLOSE, close_kernel)

    w_c = img_crop.shape[1]
    is_flipped = False
    if np.sum(img_crop[:, :max(1, int(w_c * 0.10))] > 0) > np.sum(img_crop[:, -max(1, int(w_c * 0.10)):] > 0):
        img_crop = cv2.flip(img_crop, 1)
        is_flipped = True

    def to_rotated_coords(p_crop):
        return (x_min + p_crop[0] if not is_flipped else x_min + (w_c - 1 - p_crop[0]), y_min + p_crop[1])
    def to_original_coords(p_rot):
        return tuple(M_rev.dot(np.array([p_rot[0], p_rot[1], 1]))[:2].astype(int))

    nonzero_cols = np.nonzero(img_crop)[1]
    if len(nonzero_cols) == 0: return None
    first_x = np.min(nonzero_cols)
    p_head_crop = (first_x, int(np.mean(np.where(img_crop[:, first_x] > 0)[0])))

    # Minimum gap size scales with body thickness rather than being fixed: a
    # real notch is a steady fraction of the animal's height, leftover noise
    # (after the closing) is not.
    min_gap_depth = max(3, int(round(body_thickness * 0.12)))

    def get_column_info(x_pos):
        if x_pos >= w_c or x_pos < 0: return True, None
        y_pix = np.where(img_crop[:, x_pos] > 0)[0]
        if len(y_pix) < 2: return True, None
        max_gap, gap_center = 0, None
        for i in range(len(y_pix) - 1):
            if y_pix[i+1] - y_pix[i] > max_gap:
                max_gap, gap_center = y_pix[i+1] - y_pix[i], (y_pix[i] + y_pix[i+1]) // 2
        return (max_gap > min_gap_depth), gap_center

    # Wider confirmation window (10% of the body instead of 5%), so one narrow
    # isolated gap that survived the closing can no longer pass for the notch.
    look_ahead = max(1, int(w_c * 0.10))
    p_notch_crop = (w_c - 1, img_crop.shape[0] // 2)
    for x in range(w_c - 2, look_ahead, -1):
        has_hole_curr, _ = get_column_info(x)
        if not has_hole_curr:
            check_ahead = True
            for next_x in range(x - look_ahead, x):
                if get_column_info(next_x)[0]: check_ahead = False; break
            if check_ahead:
                _, gap_center_prev = get_column_info(x + 1)
                p_notch_crop = (x, gap_center_prev) if gap_center_prev is not None else (x, int(np.mean(np.where(img_crop[:, x] > 0)[0])))
                break

    widths_data = []
    dist_x_crop = p_notch_crop[0] - p_head_crop[0]
    for pct in range(5, 100, 5):
        target_x = p_head_crop[0] + int(dist_x_crop * (pct / 100.0))
        if 0 <= target_x < w_c:
            y_pix = np.where(img_crop[:, target_x] > 0)[0]
            if len(y_pix) > 0:
                pt_top_orig = to_original_coords(to_rotated_coords((target_x, np.min(y_pix))))
                pt_bot_orig = to_original_coords(to_rotated_coords((target_x, np.max(y_pix))))
                widths_data.append({'pct': pct, 'pt_top': pt_top_orig, 'pt_bot': pt_bot_orig, 'width_m': (np.max(y_pix) - np.min(y_pix)) * gsd})

    p_head_orig = to_original_coords(to_rotated_coords(p_head_crop))
    p_notch_orig = to_original_coords(to_rotated_coords(p_notch_crop))
    lung_m = math.hypot(p_notch_crop[0] - p_head_crop[0], p_notch_crop[1] - p_head_crop[1]) * gsd
    return lung_m, p_head_orig, p_notch_orig, widths_data


def measure_length_pca_notch_curve(mask_uint8, gsd):
    """Find the notch from a smoothed profile instead of column-by-column.

    The head is found as in the other PCA variants. The gap between the two
    tail lobes is traced as a curve along the body axis, smoothed with a moving
    average, and the transition is looked for on that curve.

    Compared with `measure_length_pca_robust_notch`: there each column passes
    or fails a threshold on its own, so one borderline column can sink the
    whole search even where the surrounding context clearly shows the notch.
    Here the depth profile is smoothed BEFORE the transition is looked for, so
    one noisy column barely moves the local average instead of deciding alone.

    The centring check (25%-75%, since a real notch sits between the two lobes)
    still applies, but only to the final chosen point, not to every column
    during smoothing.
    """
    y_coords, x_coords = np.nonzero(mask_uint8)
    if len(x_coords) < 25: return None
    pts = np.column_stack((x_coords, y_coords)).astype(np.float32)
    pca_result = cv2.PCACompute(pts, mean=None)
    mean, eigenvectors = pca_result[0], pca_result[1]
    angle_deg = np.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w = mask_uint8.shape
    M_rot = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), angle_deg, 1.0)
    M_rev = cv2.getRotationMatrix2D((mean[0, 0], mean[0, 1]), -angle_deg, 1.0)
    mask_rot = cv2.warpAffine(mask_uint8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_r, x_r = np.nonzero(mask_rot)
    if len(x_r) == 0: return None
    x_min, x_max, y_min, y_max = np.min(x_r), np.max(x_r), np.min(y_r), np.max(y_r)
    img_crop = mask_rot[y_min:y_max+1, x_min:x_max+1]
    w_c = img_crop.shape[1]
    is_flipped = False
    if np.sum(img_crop[:, :max(1, int(w_c * 0.10))] > 0) > np.sum(img_crop[:, -max(1, int(w_c * 0.10)):] > 0):
        img_crop = cv2.flip(img_crop, 1)
        is_flipped = True

    def to_rotated_coords(p_crop):
        return (x_min + p_crop[0] if not is_flipped else x_min + (w_c - 1 - p_crop[0]), y_min + p_crop[1])
    def to_original_coords(p_rot):
        return tuple(M_rev.dot(np.array([p_rot[0], p_rot[1], 1]))[:2].astype(int))

    nonzero_cols = np.nonzero(img_crop)[1]
    if len(nonzero_cols) == 0: return None
    first_x = np.min(nonzero_cols)
    p_head_crop = (first_x, int(np.mean(np.where(img_crop[:, first_x] > 0)[0])))

    body_thickness = max(1, int(y_max - y_min))

    # Raw gap profile per column: size and vertical centre of the largest gap
    # in that column, if any. No thresholds here, just the measurements.
    gap_sizes = np.zeros(w_c)
    gap_centers = np.full(w_c, np.nan)
    for x in range(w_c):
        y_pix = np.where(img_crop[:, x] > 0)[0]
        if len(y_pix) < 2:
            continue
        diffs = np.diff(y_pix)
        idx = int(np.argmax(diffs))
        gap_sizes[x] = diffs[idx]
        gap_centers[x] = (y_pix[idx] + y_pix[idx + 1]) / 2.0

    # Moving average over the profile: a real notch stays clear across many
    # columns, while one stray gap is flattened by its neighbours.
    smooth_window = max(1, int(round(w_c * 0.04)))
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        gap_sizes_smooth = np.convolve(gap_sizes, kernel, mode="same")
    else:
        gap_sizes_smooth = gap_sizes

    min_gap_depth = max(3, int(round(body_thickness * 0.12)))
    look_ahead = max(1, int(w_c * 0.05))

    p_notch_crop = None
    for x in range(w_c - 2, look_ahead, -1):
        if gap_sizes_smooth[x] <= min_gap_depth:
            if np.all(gap_sizes_smooth[max(0, x - look_ahead):x] <= min_gap_depth):
                candidate_x = x + 1
                if candidate_x < w_c and not np.isnan(gap_centers[candidate_x]):
                    p_notch_crop = (candidate_x, int(gap_centers[candidate_x]))
                    break
    if p_notch_crop is None:
        return None

    # Final centring check on the chosen point: a real notch sits midway
    # between the two tail lobes.
    notch_x, notch_y = p_notch_crop
    y_pix_at_notch = np.where(img_crop[:, notch_x] > 0)[0]
    if len(y_pix_at_notch) < 2:
        return None
    column_span = y_pix_at_notch[-1] - y_pix_at_notch[0]
    if column_span <= 0:
        return None
    relative_position = (notch_y - y_pix_at_notch[0]) / column_span
    if not (0.25 <= relative_position <= 0.75):
        return None

    widths_data = []
    dist_x_crop = p_notch_crop[0] - p_head_crop[0]
    for pct in range(5, 100, 5):
        target_x = p_head_crop[0] + int(dist_x_crop * (pct / 100.0))
        if 0 <= target_x < w_c:
            y_pix = np.where(img_crop[:, target_x] > 0)[0]
            if len(y_pix) > 0:
                pt_top_orig = to_original_coords(to_rotated_coords((target_x, np.min(y_pix))))
                pt_bot_orig = to_original_coords(to_rotated_coords((target_x, np.max(y_pix))))
                widths_data.append({'pct': pct, 'pt_top': pt_top_orig, 'pt_bot': pt_bot_orig, 'width_m': (np.max(y_pix) - np.min(y_pix)) * gsd})

    p_head_orig = to_original_coords(to_rotated_coords(p_head_crop))
    p_notch_orig = to_original_coords(to_rotated_coords(p_notch_crop))
    lung_m = math.hypot(p_notch_crop[0] - p_head_crop[0], p_notch_crop[1] - p_head_crop[1]) * gsd
    return lung_m, p_head_orig, p_notch_orig, widths_data


def create_manual_width_guides(points_list, gsd, img_w, img_h):
    """Place width guides perpendicular to the manual line, every 5% of it."""
    if len(points_list) < 2:
        return []

    segments = []
    total_length = 0.0
    for point_index in range(len(points_list) - 1):
        point_a = np.asarray(points_list[point_index], dtype=np.float64)
        point_b = np.asarray(points_list[point_index + 1], dtype=np.float64)
        vector = point_b - point_a
        segment_length = float(np.linalg.norm(vector))
        if segment_length <= 0:
            continue
        segments.append(
            {
                "start": point_a,
                "vector": vector,
                "length": segment_length,
                "offset": total_length,
            }
        )
        total_length += segment_length
    if not segments or total_length <= 0:
        return []

    # Start every guide at a fixed 100px rather than scaling it to the drawn
    # length: with no mask there is no shape to guess a sensible width from,
    # and the user adjusts each one by dragging its ends.
    initial_half_width = 50.0

    def clip_point(point):
        return (
            int(np.clip(round(point[0]), 0, max(0, img_w - 1))),
            int(np.clip(round(point[1]), 0, max(0, img_h - 1))),
        )

    guides = []
    for pct in range(5, 100, 5):
        target_distance = total_length * (pct / 100.0)
        segment = segments[-1]
        for candidate_segment in segments:
            if (
                target_distance
                <= candidate_segment["offset"] + candidate_segment["length"]
            ):
                segment = candidate_segment
                break
        local_distance = target_distance - segment["offset"]
        local_fraction = np.clip(
            local_distance / segment["length"],
            0.0,
            1.0,
        )
        center = segment["start"] + segment["vector"] * local_fraction
        tangent = segment["vector"] / segment["length"]
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        point_top = clip_point(center + normal * initial_half_width)
        point_bottom = clip_point(center - normal * initial_half_width)
        guides.append(
            {
                "pct": pct,
                "pt_top": point_top,
                "pt_bot": point_bottom,
                "width_m": math.hypot(
                    point_bottom[0] - point_top[0],
                    point_bottom[1] - point_top[1],
                )
                * gsd,
            }
        )
    return guides


def enhance_review_frame(frame, contrast=1.0, brightness=0, use_clahe=False):
    """Adjust the image shown and fed to the model, leaving the file untouched."""
    # A 256-byte lookup table avoids making a float32 copy of the whole 4K
    # frame (about 100 MB each), which matters on Macs with unified memory.
    lookup_table = np.clip(
        (
            np.arange(256, dtype=np.float32) - 127.5
        ) * contrast + 127.5 + brightness,
        0,
        255,
    ).astype(np.uint8)
    enhanced = cv2.LUT(frame, lookup_table)
    if use_clahe:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
    return enhanced


def draw_readable_text(
    image,
    text,
    origin,
    font_scale,
    color,
    thickness=2,
    outline=True,
):
    """Testo con contorno scuro, leggibile su acqua chiara o riflessi."""
    if outline:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


SUBPIXEL_SHIFT = 4  # cv2 fixed-point precision (1/16 px) per le linee di collegamento in review.py


def subpixel_point(point, shift=SUBPIXEL_SHIFT):
    """Convert a float point to the fixed-point coordinates cv2 expects when
    drawing with the `shift` argument (lines, polylines, circles)."""
    scale = 1 << shift
    return (int(round(point[0] * scale)), int(round(point[1] * scale)))


def draw_cross_marker(image, center, size, color, thickness=None):
    """Draw a cross (+) marker, the one marker style used everywhere in review.

    Widths, length, landmarks, SAM prompts and dataset polygon points all use
    it. A cross stays readable on top of the mask outline or another nearby
    point, which a filled dot would hide.

    Thickness comes from the UI slider ("Cross thickness", default 1), and
    anti-aliasing is deliberately off: the cross is made of horizontal and
    vertical segments only, with no diagonals to smooth, so AA would just
    spread the stroke over more pixels and make it fatter and blurrier for
    nothing.

    Crosses are drawn in review.py AFTER the zoom resize and crop, so they stay
    thin at any zoom level instead of being blown up along with the mask and
    contours by INTER_NEAREST.
    """
    x, y = int(round(center[0])), int(round(center[1]))
    size = max(1, int(round(size)))
    if thickness is None:
        thickness = 1
    cv2.drawMarker(
        image,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size * 2,
        thickness=thickness,
    )


def clean_review_mask(mask, frame_shape):
    """Tidy up the mask: keep the outline, drop stray fragments."""
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.shape != frame_shape[:2]:
        mask = cv2.resize(
            mask,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cleaned = np.zeros_like(mask)
        cv2.drawContours(
            cleaned,
            [max(contours, key=cv2.contourArea)],
            -1,
            255,
            thickness=cv2.FILLED,
        )
        mask = cleaned
    return mask


def sanitize_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "unknown"


def polygon_epsilon_fraction(complexity_pct):
    """Turn a detail level of 1 (rough) .. 100 (detailed) into the fraction of
    the perimeter approxPolyDP uses as epsilon. More detail means a smaller
    epsilon and more vertices in the resulting polygon."""
    complexity_pct = max(1, min(100, complexity_pct))
    t = (complexity_pct - 1) / 99.0
    return 0.02 * (1 - t) + 0.0003 * t


def mask_to_polygons(mask, epsilon_fraction=0.0005):
    """Mask outlines simplified with approxPolyDP, in pixel coordinates."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(
            contour,
            epsilon=max(0.5, perimeter * epsilon_fraction),
            closed=True,
        ).reshape(-1, 2)
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def polygon_to_mask(polygon, frame_shape):
    """The inverse of mask_to_polygons: fill a polygon into a binary mask.

    Used so a hand-corrected polygon becomes the mask the automatic
    measurement is computed from.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    return mask


def mask_to_yolo_segmentation_lines(mask, epsilon_fraction=0.0005):
    """Convert a binary mask into YOLO-Seg polygon format."""
    frame_h, frame_w = mask.shape[:2]
    lines = []
    for polygon in mask_to_polygons(mask, epsilon_fraction):
        normalized = []
        for point_x, point_y in polygon:
            normalized.extend(
                (
                    np.clip(float(point_x) / frame_w, 0.0, 1.0),
                    np.clip(float(point_y) / frame_h, 0.0, 1.0),
                )
            )
        lines.append(
            "0 " + " ".join(f"{coordinate:.6f}" for coordinate in normalized)
        )
    return lines


def extract_detection_mask(result, detection_index, frame_shape):
    """Take YOLO's segmentation mask, if there is one, at the original size."""
    if result.masks is None or len(result.masks.data) <= detection_index:
        return None
    mask_data = result.masks.data[detection_index].cpu().numpy()
    if mask_data.size:
        mask = (mask_data > 0.5).astype(np.uint8) * 255
        if mask.shape != frame_shape[:2]:
            mask = cv2.resize(
                mask,
                (frame_shape[1], frame_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask

    if len(result.masks.xy) <= detection_index:
        return None
    polygon = np.asarray(result.masks.xy[detection_index], dtype=np.int32)
    if polygon.ndim != 2 or len(polygon) < 3:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def deduplicate_detections_by_mask_iou(result, frame_shape, iou_threshold=0.5):
    """Indices of one frame's detections to keep, dropping duplicate masks.

    YOLO's NMS compares bounding boxes, not masks. For a long body like a sperm
    whale, two boxes covering different parts (one the front, one nearly the
    whole animal) can have a low box IoU, below the NMS threshold, and still be
    the same animal, because their MASKS overlap heavily. Comparing the actual
    masks is more reliable for shapes that do not fill their box.

    This is also the one kind of duplicate the mean-shape filter cannot catch:
    each mask on its own looks like a plausible whale; the problem is that they
    are the same whale.

    Kept greedily by decreasing confidence: the most confident detection in
    each group survives, and any other overlapping it too much is dropped.
    """
    num_detections = len(result.boxes)
    if num_detections <= 1:
        return list(range(num_detections))

    confidences = result.boxes.conf.detach().cpu().numpy()
    order = np.argsort(-confidences)
    masks = [extract_detection_mask(result, i, frame_shape) for i in range(num_detections)]

    kept_indices = []
    kept_masks = []
    for idx in order:
        mask = masks[idx]
        mask_bool = mask > 0 if mask is not None else None
        is_duplicate = False
        if mask_bool is not None:
            for kept_mask in kept_masks:
                if kept_mask is None:
                    continue
                intersection = np.logical_and(mask_bool, kept_mask).sum()
                union = np.logical_or(mask_bool, kept_mask).sum()
                if union > 0 and intersection / union >= iou_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            kept_indices.append(int(idx))
            kept_masks.append(mask_bool)
    return sorted(kept_indices)


def select_detection_near_reference(result, reference_bbox):
    """Pick the box closest to the previous frame's, falling back to the
    largest one."""
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(
        np.float32,
        copy=True,
    )
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    if reference_bbox is None:
        return int(np.argmax(areas))

    reference = np.asarray(reference_bbox, dtype=np.float32).reshape(-1, 4)[0]
    intersection_x1 = np.maximum(boxes[:, 0], reference[0])
    intersection_y1 = np.maximum(boxes[:, 1], reference[1])
    intersection_x2 = np.minimum(boxes[:, 2], reference[2])
    intersection_y2 = np.minimum(boxes[:, 3], reference[3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0,
        intersection_y2 - intersection_y1,
    )
    reference_area = max(
        0.0,
        float(reference[2] - reference[0]),
    ) * max(
        0.0,
        float(reference[3] - reference[1]),
    )
    union = areas + reference_area - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    best_overlap = int(np.argmax(iou))
    if iou[best_overlap] > 0.01:
        return best_overlap
    return int(np.argmax(areas))


def expanded_bbox_crop(frame, bbox_xyxy, margin_fraction):
    """Ritaglia il bbox con margine e restituisce crop e coordinate originali."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox_xyxy).reshape(-1)[:4]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_x1 = max(0, int(math.floor(x1 - box_w * margin_fraction)))
    crop_y1 = max(0, int(math.floor(y1 - box_h * margin_fraction)))
    crop_x2 = min(frame_w, int(math.ceil(x2 + box_w * margin_fraction)))
    crop_y2 = min(frame_h, int(math.ceil(y2 + box_h * margin_fraction)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2], (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )


def expand_bbox(bbox_xyxy, frame_shape, margin_fraction):
    """Grow an xyxy box by a fraction on each side, clipped to the image.

    Used to give SAM some margin around YOLO's box, which sometimes cuts thin
    parts short, such as the tip of the head.
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)[:4]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    x1 = max(0.0, x1 - box_w * margin_fraction)
    y1 = max(0.0, y1 - box_h * margin_fraction)
    x2 = min(float(frame_w), x2 + box_w * margin_fraction)
    y2 = min(float(frame_h), y2 + box_h * margin_fraction)
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def normalize_mask_shape(mask, output_size=(256, 128)):
    """Align and rescale a shape so it can be compared with the mean shape."""
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) < 25:
        return None
    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    angle = math.degrees(
        math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    )
    rotation = cv2.getRotationMatrix2D(
        (float(mean[0, 0]), float(mean[0, 1])),
        angle,
        1.0,
    )
    rotated = cv2.warpAffine(
        mask,
        rotation,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    rotated_y, rotated_x = np.nonzero(rotated)
    if len(rotated_x) < 25:
        return None
    crop = rotated[
        rotated_y.min():rotated_y.max() + 1,
        rotated_x.min():rotated_x.max() + 1,
    ]

    # Which end is the head: when the automatic measurement confidently finds
    # head and notch (the same tested logic as measure_length_pca, not just a
    # density comparison at the two ends), use that to decide. It is more
    # reliable and follows the same convention (head on the left). Only when no
    # trustworthy notch is found does this fall back on comparing densities.
    need_flip = None
    pca_measure = measure_length_pca(mask, gsd=1.0)
    if pca_measure is not None:
        _, head_orig, notch_orig, _ = pca_measure
        head_rot_x = rotation[0, 0] * head_orig[0] + rotation[0, 1] * head_orig[1] + rotation[0, 2]
        notch_rot_x = rotation[0, 0] * notch_orig[0] + rotation[0, 1] * notch_orig[1] + rotation[0, 2]
        need_flip = head_rot_x > notch_rot_x

    if need_flip is None:
        edge_width = max(1, crop.shape[1] // 6)
        need_flip = np.count_nonzero(crop[:, :edge_width]) > np.count_nonzero(
            crop[:, -edge_width:]
        )

    if need_flip:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    normalized = cv2.resize(
        crop,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    return (normalized > 0).astype(np.uint8)


def candidate_navigation_order(candidates, mode):
    """Return the candidate indices in the order chosen for preselection."""
    if mode == "Size (largest first)":
        ordered = candidates.sort_values(
            ["auto_length", "frame_pos"],
            ascending=[False, True],
            kind="stable",
        )
    else:
        ordered = candidates.sort_values(
            ["frame_pos", "auto_length"],
            ascending=[True, False],
            kind="stable",
        )
    return [int(position) for position in ordered.index]
