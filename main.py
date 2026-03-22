import os

from feature_matching import extract_and_match_features
from epipolar_geometry import compute_epipolar_geometry
from stereo_matching import compute_disparity_map
from rectification import rectify_stereo_images
from member5_stereo_matching import run_member5_stereo_matching

if __name__ == "__main__":
    left = "images/left.png"
    right = "images/right.png"
    images_dir = "images"
    output_dir = "output"

    window_size = 11
    min_disparity = 0
    max_disparity = 48
    median_filter_size = 5

    print("=== Member 3: Stereo Matching Before Rectification ===")
    _, before_mask = compute_disparity_map(
        left,
        right,
        window_size=window_size,
        min_disparity=min_disparity,
        max_disparity=max_disparity,
        output_dir=images_dir,
        median_filter_size=median_filter_size,
    )
    before_valid_ratio = before_mask.sum() / before_mask.size

    print("\n=== Members 1 and 2: Feature Matching and Epipolar Geometry ===")
    pts_left, pts_right = extract_and_match_features(left, right)

    print("Left points shape:", pts_left.shape)
    print("Right points shape:", pts_right.shape)

    results = compute_epipolar_geometry(
        left_image_path=left,
        right_image_path=right,
        pts_left=pts_left,
        pts_right=pts_right,
        output_dir=images_dir,
    )

    print("Fundamental matrix F:\n", results["F"])
    print("Essential matrix E:\n", results["E"])
    print("Rotation R:\n", results["R"])
    print("Translation direction t:\n", results["t"].ravel())

    inlier_pts_left = pts_left[results["inlier_mask"]]
    inlier_pts_right = pts_right[results["inlier_mask"]]

    print("\n=== Member 4: Rectification ===")
    rectification = rectify_stereo_images(
        left_image_path=left,
        right_image_path=right,
        F=results["F"],
        pts_left=inlier_pts_left,
        pts_right=inlier_pts_right,
        output_dir=images_dir,
    )

    print("Rectification homography H1:\n", rectification["H1"])
    print("Rectification homography H2:\n", rectification["H2"])

    print("\n=== Member 5: Stereo Matching After Rectification ===")
    member5_outputs = run_member5_stereo_matching(
        rectified_left_path=os.path.join(images_dir, "rectified_left.png"),
        rectified_right_path=os.path.join(images_dir, "rectified_right.png"),
        original_output_dir=images_dir,
        output_dir=output_dir,
        window_size=window_size,
        min_disparity=min_disparity,
        max_disparity=max_disparity,
        median_filter_size=median_filter_size,
        before_valid_ratio=before_valid_ratio,
    )

    print("Member 5 outputs saved to:", member5_outputs["output_dir"])
    print(
        "Comparison summary: "
        f"{before_valid_ratio * 100:.1f}% before rectification -> "
        f"{member5_outputs['after_valid_ratio'] * 100:.1f}% after rectification"
    )
