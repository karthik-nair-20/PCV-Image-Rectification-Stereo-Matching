# Image Rectification & Stereo Matching

This project computes stereo correspondences and uses them to estimate epipolar geometry and camera relative orientation.

The pipeline currently has two parts:
- Part 1: Feature extraction and point matching (`feature_matching.py`)
- Part 2: Fundamental matrix, epipolar lines, and pose estimation (`epipolar_geometry.py`)

`main.py` orchestrates both parts end-to-end.

## Project Structure

- `main.py`: Entry point; runs Part 1 then Part 2.
- `feature_matching.py`: Detects and matches keypoints between left/right images.
- `epipolar_geometry.py`: Estimates `F`, visualizes epipolar lines, computes `E`, `R`, `t` direction, and saves results.
- `images/left.png`, `images/right.png`: Input stereo pair.
- `images/`: Also used as output directory for generated results.
- `test_opencv.py`: Optional environment check for OpenCV/NumPy.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`

## Setup (Step-by-Step)

1. Open terminal and go to project folder:

```bash
cd /Users/karthiknair/Downloads/stereo_project
```

2. (Recommended) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Upgrade pip:

```bash
python3 -m pip install --upgrade pip
```

4. Install dependencies:

```bash
python3 -m pip install numpy opencv-python
```

5. (Optional) Verify installation:

```bash
python3 test_opencv.py
```

Expected output includes OpenCV and NumPy versions.

## Input Images

Place your stereo images at:
- `images/left.png`
- `images/right.png`

If either image is missing or unreadable, the pipeline will fail early.

## How to Run

Run the full pipeline:

```bash
python3 main.py
```

The script will:
1. Load images
2. Extract/match features
3. Compute `F`
4. Draw epipolar lines
5. Estimate `R` and `t` direction
6. Save all outputs inside `images/`

## Part-Wise Documentation

## Part 1: Feature Matching (`feature_matching.py`)

Purpose: Generate point correspondences between the left and right images.

### Workflow
1. Read both images in grayscale (`cv2.imread(..., cv2.IMREAD_GRAYSCALE)`).
2. Detect ORB keypoints and descriptors (`cv2.ORB_create(nfeatures=2000)`).
3. Match descriptors using BFMatcher with Hamming distance (`knnMatch`, `k=2`).
4. Apply Lowe ratio test (`m.distance < 0.75 * n.distance`) to keep reliable matches.
5. Convert surviving matches into coordinate arrays:
   - `pts_left` (Nx2)
   - `pts_right` (Nx2)

### Output of Part 1
- `pts_left`, `pts_right` as `np.float32` arrays used directly in Part 2.

## Part 2: Epipolar Geometry and Relative Orientation (`epipolar_geometry.py`)

Purpose: Use correspondences from Part 1 to recover stereo geometry.

### Workflow
1. Validate correspondences (minimum 8 points).
2. Compute Fundamental matrix:
   - `F, mask = cv2.findFundamentalMat(..., cv2.FM_RANSAC, 1.0, 0.99)`
   - Keeps robust inliers via RANSAC mask.
3. Compute epipolar lines:
   - `cv2.computeCorrespondEpilines(...)` in both image directions.
4. Visualize epipolar constraints:
   - Draw lines and corresponding points.
   - Save visualization images.
5. Estimate camera relative orientation:
   - Build Essential matrix `E = K^T F K`
   - Recover pose via `cv2.recoverPose(E, ...)`.
   - Returns rotation `R` and translation direction `t` (scale is unknown).
6. Save numeric outputs for downstream team usage.

### Camera Intrinsics Note
- If `K` is passed to `compute_epipolar_geometry(...)`, that calibrated intrinsic matrix is used.
- If `K` is not passed, a fallback intrinsic matrix is generated from image size.
- For accurate metric geometry, provide real camera calibration.

## Output Files (Saved in `images/`)

Visual outputs:
- `epilines_left.png`
- `epilines_right.png`
- `epilines_left_points.png`
- `epilines_right_points.png`

Matrix/vector outputs:
- `fundamental_matrix_F.txt`
- `essential_matrix_E.txt`
- `rotation_R.txt`
- `translation_direction_t.txt`

Masks:
- `inlier_mask.npy`
- `pose_inlier_mask.npy`

## Console Output You Should See

`main.py` prints:
- Left/right point array shapes
- Fundamental matrix `F`
- Essential matrix `E`
- Rotation matrix `R`
- Translation direction vector `t`

## Troubleshooting

1. `ModuleNotFoundError: No module named 'cv2'`
- Install OpenCV:

```bash
python3 -m pip install opencv-python
```

2. `At least 8 matched points are required to estimate F`
- Use higher-quality stereo images with more overlap/texture.
- Ensure both input files are correct.

3. Poor epipolar lines or unstable pose
- Use calibrated camera intrinsics (`K`) instead of fallback values.
- Improve match quality (sharper images, less blur).

## Reproducibility Notes

- The epipolar line colors are randomly generated each run, so visualization colors may differ.
- `t` from essential matrix decomposition is a direction only, not absolute translation magnitude.

## How to Create a Pull Request (PR) to This Repository

Use these steps when you want to contribute changes safely through a PR instead of pushing directly to `main`.

1. Go to the project folder:

```bash
cd /Users/karthiknair/Downloads/stereo_project
```

2. Make sure your local `main` is up to date:

```bash
git checkout main
git pull origin main
```

3. Create a new feature branch:

```bash
git checkout -b feature/<short-description>
```

Example:

```bash
git checkout -b feature/epipolar-geometry-docs
```

4. Make your code/documentation changes.

5. Stage files:

```bash
git add .
```

6. Commit with a clear message:

```bash
git commit -m "Add epipolar geometry pipeline documentation"
```

7. Push the branch to GitHub:

```bash
git push -u origin feature/<short-description>
```

8. Open GitHub and create the PR:
- Open: `https://github.com/karthik-nair-20/PCV-Image-Rectification-Stereo-Matching`
- GitHub usually shows a **Compare & pull request** button after push.
- If not, go to **Pull requests** -> **New pull request**.
- Set:
  - `base`: `main`
  - `compare`: `feature/<short-description>`
- Add PR title and description (what changed, why, any test output).
- Click **Create pull request**.

9. Address review comments:
- Make requested edits locally on the same branch.
- Commit and push again:

```bash
git add .
git commit -m "Address PR review comments"
git push
```

10. Merge after approval:
- Use **Squash and merge** (recommended for clean history), or project-preferred merge strategy.
- Delete the branch on GitHub after merge.

11. Sync local `main` after merge:

```bash
git checkout main
git pull origin main
git branch -d feature/<short-description>
```

### If your push is rejected (`fetch first`)

Rebase your branch on latest remote `main` and push again:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Use `--force-with-lease` only on your feature branch, not on `main`.
