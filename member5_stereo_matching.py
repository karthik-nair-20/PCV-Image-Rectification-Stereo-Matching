import os
import shutil
from typing import Dict, Optional

import cv2
import numpy as np

from stereo_matching import compute_disparity_map


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(
        panel,
        title,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _resize_to_match(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    target_h, target_w = target_shape
    if image.shape[:2] == target_shape:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _find_valid_overlap_bbox(
    left_image: np.ndarray,
    right_image: np.ndarray,
) -> tuple:
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    valid_mask = (left_gray > 2) & (right_gray > 2)

    ys, xs = np.where(valid_mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = left_image.shape[:2]
        return 0, 0, w, h

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    return x_min, y_min, x_max, y_max


def run_member5_stereo_matching(
    rectified_left_path: str,
    rectified_right_path: str,
    original_output_dir: str = "images",
    output_dir: str = "output",
    window_size: int = 11,
    min_disparity: int = 0,
    max_disparity: int = 48,
    median_filter_size: int = 5,
    before_valid_ratio: Optional[float] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    rectified_left = cv2.imread(rectified_left_path, cv2.IMREAD_COLOR)
    rectified_right = cv2.imread(rectified_right_path, cv2.IMREAD_COLOR)
    if rectified_left is None or rectified_right is None:
        raise FileNotFoundError("Could not load rectified images for Member 5.")

    x_min, y_min, x_max, y_max = _find_valid_overlap_bbox(rectified_left, rectified_right)
    cropped_left = rectified_left[y_min:y_max, x_min:x_max]
    cropped_right = rectified_right[y_min:y_max, x_min:x_max]

    cropped_left_path = os.path.join(output_dir, "rectified_left_cropped.png")
    cropped_right_path = os.path.join(output_dir, "rectified_right_cropped.png")
    cv2.imwrite(cropped_left_path, cropped_left)
    cv2.imwrite(cropped_right_path, cropped_right)

    _, after_mask = compute_disparity_map(
        cropped_left_path,
        cropped_right_path,
        output_dir=output_dir,
        window_size=window_size,
        min_disparity=min_disparity,
        max_disparity=max_disparity,
        median_filter_size=median_filter_size,
        ignore_black_borders=True,
    )

    cropped_left_gray = cv2.cvtColor(cropped_left, cv2.COLOR_BGR2GRAY)
    cropped_right_gray = cv2.cvtColor(cropped_right, cv2.COLOR_BGR2GRAY)
    valid_overlap_mask = (cropped_left_gray > 2) & (cropped_right_gray > 2)

    after_raw_path = os.path.join(output_dir, "disparity_raw.png")
    after_color_path = os.path.join(output_dir, "disparity_color.png")
    before_raw_path = os.path.join(original_output_dir, "disparity_raw.png")
    before_color_path = os.path.join(original_output_dir, "disparity_color.png")

    after_valid_ratio = 0.0
    if valid_overlap_mask.any():
        after_valid_ratio = float(after_mask.sum()) / float(valid_overlap_mask.sum())

    if os.path.exists(before_color_path):
        shutil.copy2(
            before_color_path,
            os.path.join(output_dir, "before_rectification_disparity_color.png"),
        )
    if os.path.exists(before_raw_path):
        shutil.copy2(
            before_raw_path,
            os.path.join(output_dir, "before_rectification_disparity_raw.png"),
        )
    if os.path.exists(after_color_path):
        shutil.copy2(
            after_color_path,
            os.path.join(output_dir, "after_rectification_disparity_color.png"),
        )
    if os.path.exists(after_raw_path):
        shutil.copy2(
            after_raw_path,
            os.path.join(output_dir, "after_rectification_disparity_raw.png"),
        )

    before_color = cv2.imread(
        before_color_path,
        cv2.IMREAD_COLOR,
    )
    after_color = cv2.imread(
        after_color_path,
        cv2.IMREAD_COLOR,
    )
    before_raw = cv2.imread(
        before_raw_path,
        cv2.IMREAD_COLOR,
    )
    after_raw = cv2.imread(
        after_raw_path,
        cv2.IMREAD_COLOR,
    )

    if (
        before_color is not None
        and after_color is not None
        and before_raw is not None
        and after_raw is not None
    ):
        target_shape = before_color.shape[:2]
        after_color = _resize_to_match(after_color, target_shape)
        before_raw = _resize_to_match(before_raw, target_shape)
        after_raw = _resize_to_match(after_raw, target_shape)

        comparison_top = cv2.hconcat(
            [
                _add_title(before_color, "Before Rectification"),
                _add_title(after_color, "After Rectification"),
            ]
        )
        comparison_bottom = cv2.hconcat(
            [
                _add_title(before_raw, "Before Rectification | Raw"),
                _add_title(after_raw, "After Rectification | Raw"),
            ]
        )
        comparison = cv2.vconcat([comparison_top, comparison_bottom])
        cv2.imwrite(
            os.path.join(output_dir, "disparity_before_after_comparison.png"),
            comparison,
        )

    with open(os.path.join(output_dir, "member5_summary.txt"), "w") as f:
        f.write("Member 5 reused the Member 3 stereo matching code.\n")
        f.write(f"Stereo matcher: NCC block matching\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"Disparity range: [{min_disparity}, {max_disparity})\n")
        f.write(f"Median filter size: {median_filter_size}\n")
        f.write(f"Original disparity outputs: {original_output_dir}\n")
        f.write(f"Rectified disparity outputs: {output_dir}\n")
        f.write(f"Rectified left image: {rectified_left_path}\n")
        f.write(f"Rectified right image: {rectified_right_path}\n")
        f.write(f"Cropped rectified left image: {cropped_left_path}\n")
        f.write(f"Cropped rectified right image: {cropped_right_path}\n")
        f.write(f"Overlap crop box: x={x_min}:{x_max}, y={y_min}:{y_max}\n")
        if before_valid_ratio is not None:
            f.write(f"Before rectification valid ratio: {before_valid_ratio * 100:.2f}%\n")
        f.write(f"After rectification valid ratio: {after_valid_ratio * 100:.2f}%\n")
        if before_valid_ratio is not None:
            improvement = after_valid_ratio - before_valid_ratio
            factor = after_valid_ratio / before_valid_ratio if before_valid_ratio > 0 else 0.0
            f.write(f"Absolute improvement: {improvement * 100:.2f} percentage points\n")
            f.write(f"Relative improvement factor: {factor:.2f}x\n")

    return {
        "output_dir": output_dir,
        "comparison_path": os.path.join(output_dir, "disparity_before_after_comparison.png"),
        "summary_path": os.path.join(output_dir, "member5_summary.txt"),
        "after_valid_ratio": after_valid_ratio,
    }
