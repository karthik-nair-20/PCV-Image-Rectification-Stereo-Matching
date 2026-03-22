import os
from typing import Optional, Tuple

import cv2
import numpy as np


def _box_mean(img: np.ndarray, ksize: int) -> np.ndarray:
    """Returns the local mean of 'img' over a (ksize x ksize) window at every pixel."""
    return cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize),
                         borderType=cv2.BORDER_REFLECT)


def _ncc_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    window_size: int,
    min_disparity: int,
    max_disparity: int,
) -> np.ndarray:
    """
    Build a cost volume of NCC scores.

    Returns
    -------
    volume : np.ndarray, shape (H, W, num_disparities), dtype float32
        volume[y, x, d_idx] = NCC score when left pixel (y,x) is matched
        to right pixel (y, x - (min_disparity + d_idx)).
        Value is in [-1, 1].  -1 is used as a sentinel for invalid windows.
    """
    h, w = left.shape
    num_disp = max_disparity - min_disparity
    ks = window_size

    left_f  = left.astype(np.float32)
    right_f = right.astype(np.float32)


    left_mean    = _box_mean(left_f,    ks)
    left_sq_mean = _box_mean(left_f**2, ks)
    left_var     = np.maximum(left_sq_mean - left_mean**2, 0.0)
    left_std     = np.sqrt(left_var)

    volume = np.full((h, w, num_disp), -1.0, dtype=np.float32)

    for d_idx, d in enumerate(range(min_disparity, max_disparity)):
        # Shift the right image to the RIGHT by d pixels.
        # After the shift: right_shifted[y, x] == right_f[y, x - d]  (for x >= d).
        # This aligns the right image with the left image at disparity d.
        right_shifted = np.zeros_like(right_f)
        if d > 0:
            right_shifted[:, d:] = right_f[:, :w - d]
        else:
            right_shifted[:, :] = right_f

        right_mean    = _box_mean(right_shifted,    ks)
        right_sq_mean = _box_mean(right_shifted**2, ks)
        right_var     = np.maximum(right_sq_mean - right_mean**2, 0.0)
        right_std     = np.sqrt(right_var)

        # Cross-covariance of left and (shifted) right.
        cross_mean  = _box_mean(left_f * right_shifted, ks)
        covariance  = cross_mean - left_mean * right_mean

        # NCC = covariance / (std_left * std_right).
        # Where either std is near-zero the window is homogeneous → set NCC = -1
        # (the lowest possible score, so it will never be the winner unless
        # everything is homogeneous).
        denom = left_std * right_std
        # Suppress divide-by-zero warning: the np.where mask already handles it,
        # but numpy still evaluates both branches before selecting.
        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = np.where(denom > 1e-4, covariance / denom, -1.0)

        volume[:, :, d_idx] = ncc.astype(np.float32)

    return volume


def _winner_takes_all(volume: np.ndarray, min_disparity: int) -> np.ndarray:
    """
    For each pixel, return the disparity index with the highest NCC score.

    Returns uint16 disparity map (absolute disparity values).
    """
    best_d_idx   = np.argmax(volume, axis=2)            # index into [0, num_disp)
    disparity_map = (best_d_idx + min_disparity).astype(np.uint16)
    return disparity_map


def _apply_median_filter(disparity: np.ndarray, filter_size: int) -> np.ndarray:
    if filter_size <= 1:
        return disparity
    if filter_size % 2 == 0:
        raise ValueError("median_filter_size must be odd.")
    return cv2.medianBlur(disparity, filter_size)


def _left_right_consistency_check(
    disp_left: np.ndarray,
    disp_right: np.ndarray,
    threshold: int = 1,
) -> np.ndarray:
    """
    Detect occluded/unreliable pixels using the left-right consistency check
    (Lecture 12, slide 40).

    For a pixel (y, x) in the left disparity map with disparity d:
    The same 3D point should appear in the right image at column (x - d).
    The right disparity map at (y, x - d) should give back disparity d.
    If |disp_left[y, x] - disp_right[y, x - d]| > threshold, the match is
    considered inconsistent (likely an occlusion or mismatched region).

    Returns
    -------
    mask : np.ndarray of bool, same shape as disp_left
        True  = consistent (reliable) match
        False = inconsistent (occluded or erroneous)
    """
    h, w      = disp_left.shape
    mask      = np.zeros((h, w), dtype=bool)
    xs        = np.arange(w)

    for y in range(h):
        d_left    = disp_left[y].astype(np.int32)
        x_right   = xs - d_left                          # column in right image
        valid     = (x_right >= 0) & (x_right < w)
        d_right   = np.where(valid, disp_right[y, np.clip(x_right, 0, w - 1)], -9999)
        consistent = np.abs(d_left - d_right) <= threshold
        mask[y]   = valid & consistent

    return mask


def _colorize_disparity(
    disparity: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_disp: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a uint16 disparity map to a colour visualization (JET colormap).

    Parameters
    ----------
    disparity : uint16 map of disparity values
    mask      : bool array — pixels where mask=False are rendered dark grey
    max_disp  : clip to this value before normalizing (defaults to array max)
    """
    disp_f = disparity.astype(np.float32)
    if max_disp is None:
        max_disp = int(disp_f.max()) if disp_f.max() > 0 else 1
    disp_norm = np.clip(disp_f / max_disp, 0.0, 1.0)
    disp_uint8 = (disp_norm * 255).astype(np.uint8)
    colored    = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)

    if mask is not None:
        colored[~mask] = (40, 40, 40)   # dark grey for invalid pixels

    return colored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_disparity_map(
    left_image_path: str,
    right_image_path: str,
    window_size: int = 11,
    min_disparity: int = 0,
    max_disparity: int = 80,
    output_dir: str = "images",
    do_lr_check: bool = True,
    median_filter_size: int = 5,
    ignore_black_borders: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a dense disparity map using NCC block matching.

    Parameters
    ----------
    left_image_path  : path to left stereo image
    right_image_path : path to right stereo image
    window_size      : correlation window side length (must be odd, e.g. 11)
    min_disparity    : smallest disparity to search (pixels)
    max_disparity    : largest  disparity to search (pixels)
    output_dir       : folder where output images are saved
    do_lr_check      : if True, apply left-right consistency filtering

    Returns
    -------
    disparity_map : np.ndarray uint16, shape (H, W)
    reliability_mask : np.ndarray bool, shape (H, W)
                       True  = pixel passed left-right consistency check
                       False = occluded / unreliable
    """
    # ---- Load & convert to grayscale ----
    left_bgr  = cv2.imread(left_image_path)
    right_bgr = cv2.imread(right_image_path)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not load stereo images. "
                                f"Checked: {left_image_path}, {right_image_path}")

    left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    if max_disparity <= min_disparity:
        raise ValueError("max_disparity must be greater than min_disparity.")

    valid_input_mask = np.ones(left_gray.shape, dtype=bool)
    if ignore_black_borders:
        valid_input_mask = (left_gray > 2) & (right_gray > 2)

    print(f"[stereo_matching] Image size: {left_gray.shape[1]}x{left_gray.shape[0]}")
    print(f"[stereo_matching] Window size: {window_size}x{window_size}")
    print(f"[stereo_matching] Disparity range: [{min_disparity}, {max_disparity})")
    print(f"[stereo_matching] Building NCC cost volume ...")

    # ---- Forward pass: left-to-right ----
    vol_l2r    = _ncc_cost_volume(left_gray, right_gray,
                                  window_size, min_disparity, max_disparity)
    disp_left  = _winner_takes_all(vol_l2r, min_disparity)
    disp_left = _apply_median_filter(disp_left, median_filter_size)

    reliability_mask = np.ones(left_gray.shape, dtype=bool)  # default: all valid

    if do_lr_check:
        print(f"[stereo_matching] Building reverse NCC cost volume (for L-R check) ...")
        # Reverse pass: right-to-left (swap images and reverse shift direction)
        vol_r2l    = _ncc_cost_volume(right_gray, left_gray,
                                      window_size, min_disparity, max_disparity)
        disp_right = _winner_takes_all(vol_r2l, min_disparity)
        disp_right = _apply_median_filter(disp_right, median_filter_size)
        reliability_mask = _left_right_consistency_check(disp_left, disp_right, threshold=1)
        reliability_mask &= valid_input_mask
        n_valid    = reliability_mask.sum()
        n_total    = valid_input_mask.sum() if ignore_black_borders else reliability_mask.size
        print(f"[stereo_matching] L-R consistent pixels: "
              f"{n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    else:
        reliability_mask = valid_input_mask.copy()

    disp_left[~valid_input_mask] = 0

    # ---- Save outputs ----
    os.makedirs(output_dir, exist_ok=True)

    # Raw disparity (grayscale, scaled to [0, 255])
    disp_scaled = np.clip(
        (disp_left.astype(np.float32) - min_disparity) /
        max(max_disparity - min_disparity, 1) * 255, 0, 255
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "disparity_raw.png"), disp_scaled)

    # Colour disparity map
    disp_color = _colorize_disparity(disp_left,
                                     mask=reliability_mask if do_lr_check else None,
                                     max_disp=max_disparity)
    cv2.imwrite(os.path.join(output_dir, "disparity_color.png"), disp_color)

    # Colour disparity map without occlusion mask (for comparison)
    disp_color_nocheck = _colorize_disparity(disp_left, mask=None,
                                             max_disp=max_disparity)
    cv2.imwrite(os.path.join(output_dir, "disparity_color_nocheck.png"), disp_color_nocheck)

    print(f"[stereo_matching] Outputs saved to: {output_dir}/")
    print(f"  disparity_raw.png         (grayscale, brighter = closer)")
    print(f"  disparity_color.png       (JET colormap, grey = occluded/unreliable)")
    print(f"  disparity_color_nocheck.png (JET colormap, no masking)")

    return disp_left, reliability_mask
