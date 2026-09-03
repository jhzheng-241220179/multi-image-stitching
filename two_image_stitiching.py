import cv2
import numpy as np
import glob
import os
import networkx as nx

def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None, None
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return None, None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    return src_pts, dst_pts, good_matches

def compute_homography(src_pts, dst_pts):
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def warp_image(image, H, reference_shape):
    h_ref, w_ref = reference_shape[:2]
    h_img, w_img = image.shape[:2]
    
    corners = np.float32([[0, 0], [0, h_img], [w_img, h_img], [w_img, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    all_corners = np.vstack((transformed_corners.reshape(-1, 2), 
                             np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]])))
    x_min, y_min = np.int32(np.floor(all_corners.min(axis=0)))
    x_max, y_max = np.int32(np.ceil(all_corners.max(axis=0)))
    
    if x_max <= x_min or y_max <= y_min:
        return None, None
    
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_adjusted = translation @ H
    
    output_shape = (y_max - y_min, x_max - x_min)
    warped = cv2.warpPerspective(image, H_adjusted, (output_shape[1], output_shape[0]))
    
    return warped, (x_min, y_min)

def blend_images(img1, img2, warped_img2, offset):
    x_min, y_min = offset
    
    h1, w1 = img1.shape[:2]
    h_warped, w_warped = warped_img2.shape[:2]
    
    canvas_h = max(h1, y_min + h_warped)
    canvas_w = max(w1, x_min + w_warped)
    
    if canvas_h <= 0 or canvas_w <= 0:
        return None
    
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    
    for i in range(h_warped):
        for j in range(w_warped):
            y = y_min + i
            x = x_min + j
            
            if y < 0 or y >= canvas_h or x < 0 or x >= canvas_w:
                continue
            
            pixel = warped_img2[i, j]
            if np.any(pixel != 0):
                if np.any(result[y, x] != 0):
                    result[y, x] = (result[y, x].astype(np.float32) + pixel.astype(np.float32)) / 2
                else:
                    result[y, x] = pixel
    
    return result

def stitch_two_images(img1, img2):
    src_pts, dst_pts, matches = get_matches(img1, img2)
    
    if src_pts is None or dst_pts is None:
        return None
    
    H, mask = compute_homography(src_pts, dst_pts)
    
    if H is None:
        return None
    
    warped_img2, offset = warp_image(img2, H, img1.shape)
    
    if warped_img2 is None:
        return None
    
    result = blend_images(img1, img2, warped_img2, offset)
    
    return result

def stitch_case(case_name, input_dir, output_dir):
    case_output_dir = os.path.join(output_dir, case_name)
    output_path = os.path.join(case_output_dir, f"{case_name}.JPG")
    
    if os.path.exists(output_path):
        print(f"[{case_name}] Already exists, skipping...")
        return
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, case_name, "*")))
    images = [cv2.imread(p) for p in image_paths]
    
    if any(img is None for img in images) or len(images) != 2:
        print(f"[{case_name}] Skipped: insufficient or unreadable images.")
        return

    print(f"[{case_name}] Processing...")
    
    stitched_image = stitch_two_images(images[0], images[1])

    if stitched_image is None:
        print(f"[{case_name}] Failed: stitching failed.")
        return

    os.makedirs(case_output_dir, exist_ok=True)
    cv2.imwrite(output_path, stitched_image)
    print(f"[{case_name}] Done: saved to {output_path}")

def main():
    input_root = "data/task1_pairwise"
    output_root = "output/task1_pairwise"
    os.makedirs(output_root, exist_ok=True)
    
    if not os.path.exists(input_root):
        print(f"Error: Input directory '{input_root}' not found!")
        return
        
    cases = [name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name))]
    if not cases:
        print(f"No cases found in '{input_root}' directory.")
        return

    print(f"Found {len(cases)} cases to process")
    
    for case in sorted(cases):
        stitch_case(case, input_root, output_root)
    
    print("\nAll cases processed!")

if __name__ == "__main__":
    main()