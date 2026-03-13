from feature_matching import extract_and_match_features
from epipolar_geometry import compute_epipolar_geometry
from stereo_matching import compute_disparity_map

if __name__ == "__main__":
    left = "images/left.png"
    right = "images/right.png"

    compute_disparity_map(left, right, output_dir="images")

    pts_left, pts_right = extract_and_match_features(left, right)

    print("Left points shape:", pts_left.shape)
    print("Right points shape:", pts_right.shape)

    results = compute_epipolar_geometry(
        left_image_path=left,
        right_image_path=right,
        pts_left=pts_left,
        pts_right=pts_right,
        output_dir="images",
    )

    print("Fundamental matrix F:\n", results["F"])
    print("Essential matrix E:\n", results["E"])
    print("Rotation R:\n", results["R"])
    print("Translation direction t:\n", results["t"].ravel())
