import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _draw_epilines(
    img_left: np.ndarray,
    img_right: np.ndarray,
    lines_in_left: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    max_lines: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw epipolar lines in left image and corresponding points in both images."""
    h, w = img_left.shape[:2]

    vis_left = _ensure_color(img_left)
    vis_right = _ensure_color(img_right)

    n = min(max_lines, len(lines_in_left), len(pts_left), len(pts_right))
    for i in range(n):
        r = lines_in_left[i]
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))

        a, b, c = r
        if abs(b) > 1e-9:
            x0, y0 = 0, int(round(-c / b))
            x1, y1 = w, int(round((-(c + a * w)) / b))
        else:
            x = int(round(-c / (a + 1e-9)))
            x0, y0 = x, 0
            x1, y1 = x, h

        cv2.line(vis_left, (x0, y0), (x1, y1), color, 1)

        pl = tuple(np.round(pts_left[i]).astype(int))
        pr = tuple(np.round(pts_right[i]).astype(int))

        cv2.circle(vis_left, pl, 5, color, -1)
        cv2.circle(vis_right, pr, 5, color, -1)

    return vis_left, vis_right


def _default_calibration(image_shape: Tuple[int, int]) -> np.ndarray:
    """Fallback intrinsic matrix when calibration is not provided."""
    h, w = image_shape[:2]
    f = float(max(h, w))
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_epipolar_geometry(
    left_image_path: str,
    right_image_path: str,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    output_dir: str = "outputs",
    K: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes fundamental matrix, epipolar line visualizations, and relative orientation.

    Returns a dictionary with keys: F, inlier_mask, E, R, t
    """
    if pts_left is None or pts_right is None:
        raise ValueError("Point correspondences are required.")

    if len(pts_left) < 8 or len(pts_right) < 8:
        raise ValueError("At least 8 matched points are required to estimate F.")

    img_left_gray = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right_gray = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if img_left_gray is None or img_right_gray is None:
        raise FileNotFoundError("Could not load one or both images for epipolar geometry.")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1/2: Fundamental matrix
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        raise RuntimeError("Fundamental matrix estimation failed.")

    inlier_mask = mask.ravel().astype(bool)
    inlier_pts_left = pts_left[inlier_mask]
    inlier_pts_right = pts_right[inlier_mask]

    if len(inlier_pts_left) < 8:
        raise RuntimeError("Not enough inlier correspondences after RANSAC.")

    # Step 3: Epipolar line visualization
    lines_left = cv2.computeCorrespondEpilines(inlier_pts_right.reshape(-1, 1, 2), 2, F)
    lines_left = lines_left.reshape(-1, 3)

    lines_right = cv2.computeCorrespondEpilines(inlier_pts_left.reshape(-1, 1, 2), 1, F)
    lines_right = lines_right.reshape(-1, 3)

    epi_left, points_on_right = _draw_epilines(
        img_left_gray, img_right_gray, lines_left, inlier_pts_left, inlier_pts_right
    )
    epi_right, points_on_left = _draw_epilines(
        img_right_gray, img_left_gray, lines_right, inlier_pts_right, inlier_pts_left
    )

    cv2.imwrite(os.path.join(output_dir, "epilines_left.png"), epi_left)
    cv2.imwrite(os.path.join(output_dir, "epilines_right_points.png"), points_on_right)
    cv2.imwrite(os.path.join(output_dir, "epilines_right.png"), epi_right)
    cv2.imwrite(os.path.join(output_dir, "epilines_left_points.png"), points_on_left)

    # Step 4: Relative orientation via Essential matrix
    K_use = K if K is not None else _default_calibration(img_left_gray.shape)
    E = K_use.T @ F @ K_use

    _, R, t, pose_mask = cv2.recoverPose(E, inlier_pts_left, inlier_pts_right, K_use)

    # Step 5: Save outputs for team
    np.savetxt(os.path.join(output_dir, "fundamental_matrix_F.txt"), F, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "essential_matrix_E.txt"), E, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "rotation_R.txt"), R, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "translation_direction_t.txt"), t, fmt="%.8f")

    np.save(os.path.join(output_dir, "inlier_mask.npy"), inlier_mask)
    np.save(os.path.join(output_dir, "pose_inlier_mask.npy"), pose_mask)

    return {
        "F": F,
        "inlier_mask": inlier_mask,
        "E": E,
        "R": R,
        "t": t,
    }
