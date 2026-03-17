import os
from typing import Dict, Tuple

import cv2
import numpy as np


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
    return transformed.reshape(-1, 2)


def _draw_horizontal_epilines(
    left_image: np.ndarray,
    right_image: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    max_lines: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    vis_left = _ensure_color(left_image)
    vis_right = _ensure_color(right_image)
    width_left = vis_left.shape[1]
    width_right = vis_right.shape[1]

    n = min(max_lines, len(pts_left), len(pts_right))
    for i in range(n):
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        left_pt = tuple(np.round(pts_left[i]).astype(int))
        right_pt = tuple(np.round(pts_right[i]).astype(int))
        y_left = left_pt[1]
        y_right = right_pt[1]

        cv2.line(vis_left, (0, y_left), (width_left - 1, y_left), color, 1)
        cv2.line(vis_right, (0, y_right), (width_right - 1, y_right), color, 1)
        cv2.circle(vis_left, left_pt, 5, color, -1)
        cv2.circle(vis_right, right_pt, 5, color, -1)

    return vis_left, vis_right


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(
        panel,
        title,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def rectify_stereo_images(
    left_image_path: str,
    right_image_path: str,
    F: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    output_dir: str = "images",
) -> Dict[str, np.ndarray]:
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)

    if left_image is None or right_image is None:
        raise FileNotFoundError("Could not load one or both stereo images for rectification.")

    if len(pts_left) < 8 or len(pts_right) < 8:
        raise ValueError("At least 8 inlier correspondences are required for rectification.")

    os.makedirs(output_dir, exist_ok=True)
    image_size = (left_image.shape[1], left_image.shape[0])

    success, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts_left),
        np.float32(pts_right),
        F,
        imgSize=image_size,
    )
    if not success:
        raise RuntimeError("stereoRectifyUncalibrated failed to compute homographies.")

    rectified_left = cv2.warpPerspective(left_image, H1, image_size)
    rectified_right = cv2.warpPerspective(right_image, H2, image_size)

    rectified_pts_left = _transform_points(np.float32(pts_left), H1)
    rectified_pts_right = _transform_points(np.float32(pts_right), H2)

    rectified_epilines_left, rectified_epilines_right = _draw_horizontal_epilines(
        rectified_left,
        rectified_right,
        rectified_pts_left,
        rectified_pts_right,
    )

    cv2.imwrite(os.path.join(output_dir, "rectified_left.png"), rectified_left)
    cv2.imwrite(os.path.join(output_dir, "rectified_right.png"), rectified_right)
    cv2.imwrite(
        os.path.join(output_dir, "epilines_rectified_left.png"),
        rectified_epilines_left,
    )
    cv2.imwrite(
        os.path.join(output_dir, "epilines_rectified_right.png"),
        rectified_epilines_right,
    )

    before_left = cv2.imread(os.path.join(output_dir, "epilines_left.png"), cv2.IMREAD_COLOR)
    before_right = cv2.imread(os.path.join(output_dir, "epilines_right.png"), cv2.IMREAD_COLOR)
    if before_left is not None and before_right is not None:
        comparison_top = cv2.hconcat(
            [
                _add_title(before_left, "Before Rectification"),
                _add_title(rectified_epilines_left, "After Rectification"),
            ]
        )
        comparison_bottom = cv2.hconcat(
            [
                _add_title(before_right, "Before Rectification"),
                _add_title(rectified_epilines_right, "After Rectification"),
            ]
        )
        comparison = cv2.vconcat([comparison_top, comparison_bottom])
        cv2.imwrite(
            os.path.join(output_dir, "epilines_before_after_comparison.png"),
            comparison,
        )

    np.savetxt(os.path.join(output_dir, "rectification_H1.txt"), H1, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "rectification_H2.txt"), H2, fmt="%.8f")

    return {
        "H1": H1,
        "H2": H2,
        "rectified_pts_left": rectified_pts_left,
        "rectified_pts_right": rectified_pts_right,
    }
