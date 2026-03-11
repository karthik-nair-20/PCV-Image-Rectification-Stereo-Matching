import cv2
import numpy as np

def extract_and_match_features(img_left_path, img_right_path):

    # 1. Load images in grayscale
    img_left = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        print("Error: Could not load one or both images.")
        return None

    print("Images loaded successfully.")

    # 2. Create ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img_left, None)
    kp2, des2 = orb.detectAndCompute(img_right, None)

    print(f"Detected {len(kp1)} keypoints in left image.")
    print(f"Detected {len(kp2)} keypoints in right image.")

    # 3. Match features using Brute Force + Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch gives 2 nearest matches for Lowe's ratio test
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches after filtering: {len(good_matches)}")

    # 5. Extract matched point coordinates
    pts_left = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    print("Returning matched point coordinates.")

    return pts_left, pts_right