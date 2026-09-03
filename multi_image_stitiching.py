import cv2
import numpy as np
import glob
import os
import networkx as nx
from collections import deque

def get_matches_pairwise(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None, None, 0
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return None, None, None, 0
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return None, None, None, 0
    
    inlier_count = np.sum(mask) if mask is not None else 0
    
    if not is_homography_valid(H):
        return None, None, None, 0
    
    return H, src_pts, dst_pts, inlier_count

def is_homography_valid(H):
    det = np.linalg.det(H)
    if abs(det) < 0.1 or abs(det) > 10:
        return False
    
    scale = np.sqrt(H[0,0]**2 + H[1,0]**2 + H[0,1]**2 + H[1,1]**2) / 2
    if scale < 0.2 or scale > 5:
        return False
    
    if abs(H[2,0]) > 0.1 or abs(H[2,1]) > 0.1:
        return False
    
    return True

def build_matching_graph(images):
    n = len(images)
    graph = nx.Graph()
    homographies = {}
    
    for i in range(n):
        graph.add_node(i)
    
    for i in range(n):
        for j in range(i + 1, n):
            H, src_pts, dst_pts, inlier_count = get_matches_pairwise(images[i], images[j])
            
            if H is not None and inlier_count >= 15:
                graph.add_edge(i, j, weight=inlier_count)
                homographies[(i, j)] = H
                homographies[(j, i)] = np.linalg.inv(H)
                print(f"  匹配图像 {i} 和 {j}: {inlier_count} 个内点")
    
    return graph, homographies

def get_global_homographies(graph, homographies, ref_idx):
    global_H = {ref_idx: np.eye(3)}
    
    visited = set([ref_idx])
    queue = deque([ref_idx])
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                H_current_to_neighbor = homographies.get((current, neighbor))
                if H_current_to_neighbor is not None:
                    H_ref_to_neighbor = global_H[current] @ H_current_to_neighbor
                    H_ref_to_neighbor = H_ref_to_neighbor / H_ref_to_neighbor[2, 2]
                    
                    if is_homography_valid(H_ref_to_neighbor):
                        global_H[neighbor] = H_ref_to_neighbor
                        visited.add(neighbor)
                        queue.append(neighbor)
                    else:
                        print(f"  警告: 图像 {neighbor} 的累积单应性无效，跳过")
    
    return global_H

def warp_image_global(image, H, canvas_bounds=None):
    h_img, w_img = image.shape[:2]
    
    corners = np.float32([[0, 0], [0, h_img], [w_img, h_img], [w_img, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    if canvas_bounds is None:
        x_min = int(np.floor(transformed_corners.min(axis=0)[0][0]))
        y_min = int(np.floor(transformed_corners.min(axis=0)[0][1]))
        x_max = int(np.ceil(transformed_corners.max(axis=0)[0][0]))
        y_max = int(np.ceil(transformed_corners.max(axis=0)[0][1]))
    else:
        x_min, y_min, x_max, y_max = canvas_bounds
    
    max_dim = 20000
    if x_max - x_min > max_dim or y_max - y_min > max_dim:
        return None, None
    
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1
    
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_adjusted = translation @ H
    
    output_shape = (y_max - y_min, x_max - x_min)
    
    if output_shape[0] <= 0 or output_shape[1] <= 0:
        return None, None
    if output_shape[0] > max_dim or output_shape[1] > max_dim:
        return None, None
    
    try:
        warped = cv2.warpPerspective(image, H_adjusted, (output_shape[1], output_shape[0]))
        return warped, (x_min, y_min)
    except cv2.error:
        return None, None

def blend_multi_images(images, warped_images, offsets):
    if not warped_images:
        return None
    
    max_h = 0
    max_w = 0
    
    for img, (x_min, y_min) in zip(warped_images, offsets):
        if img is None:
            continue
        h, w = img.shape[:2]
        max_h = max(max_h, y_min + h)
        max_w = max(max_w, x_min + w)
    
    max_canvas_dim = 15000
    if max_h > max_canvas_dim or max_w > max_canvas_dim:
        scale = min(max_canvas_dim / max_w, max_canvas_dim / max_h, 1.0)
        max_w = int(max_w * scale)
        max_h = int(max_h * scale)
    
    if max_h <= 0 or max_w <= 0:
        return None
    
    print(f"画布尺寸: {max_w} x {max_h}")
    
    result = np.zeros((max_h, max_w, 3), dtype=np.float32)
    weight_sum = np.zeros((max_h, max_w), dtype=np.float32)
    
    for idx, (img, (x_min, y_min)) in enumerate(zip(warped_images, offsets)):
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        weight = np.ones((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                if np.all(img[y, x] == 0):
                    weight[y, x] = 0
                else:
                    center_dist = np.sqrt(((x - w/2) / w)**2 + ((y - h/2) / h)**2)
                    weight[y, x] = max(0, 1 - center_dist * 0.5)
        
        valid_pixels = 0
        for y in range(h):
            for x in range(w):
                if weight[y, x] > 0:
                    canvas_y = y_min + y
                    canvas_x = x_min + x
                    
                    if 0 <= canvas_y < max_h and 0 <= canvas_x < max_w:
                        result[canvas_y, canvas_x] += img[y, x].astype(np.float32) * weight[y, x]
                        weight_sum[canvas_y, canvas_x] += weight[y, x]
                        valid_pixels += 1
        
        print(f"  图像 {idx}: 有效像素 {valid_pixels}/{h*w}")
    
    for y in range(max_h):
        for x in range(max_w):
            if weight_sum[y, x] > 0:
                result[y, x] /= weight_sum[y, x]
    
    return result.astype(np.uint8)

def stitch_multi_images(images):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    
    print(f"拼接 {len(images)} 张图像...")
    
    graph, homographies = build_matching_graph(images)
    
    if graph.number_of_edges() == 0:
        print("未找到任何图像对之间的匹配")
        return images[0]
    
    print(f"匹配图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    
    degrees = dict(graph.degree())
    ref_idx = max(degrees, key=degrees.get)
    print(f"选择图像 {ref_idx} 作为参考 (度数: {degrees[ref_idx]})")
    
    global_H = get_global_homographies(graph, homographies, ref_idx)
    
    if len(global_H) < 2:
        print("无法建立足够的连接")
        return None
    
    all_corners = []
    for idx, H in global_H.items():
        h_img, w_img = images[idx].shape[:2]
        corners = np.float32([[0, 0], [0, h_img], [w_img, h_img], [w_img, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(transformed_corners.reshape(-1, 2))
    
    if not all_corners:
        print("没有有效的角点")
        return None
    
    all_corners = np.vstack(all_corners)
    x_min = int(np.floor(all_corners.min(axis=0)[0]))
    y_min = int(np.floor(all_corners.min(axis=0)[1]))
    x_max = int(np.ceil(all_corners.max(axis=0)[0]))
    y_max = int(np.ceil(all_corners.max(axis=0)[1]))
    
    if abs(x_max - x_min) > 30000 or abs(y_max - y_min) > 30000:
        print(f"边界框过大 ({x_max-x_min} x {y_max-y_min})，使用保守边界")
        x_min = max(x_min, -5000)
        y_min = max(y_min, -5000)
        x_max = min(x_max, 15000)
        y_max = min(y_max, 10000)
    
    margin = 200
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin
    
    canvas_bounds = (x_min, y_min, x_max, y_max)
    print(f"画布边界: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    warped_images = []
    offsets = []
    for idx, H in global_H.items():
        warped, offset = warp_image_global(images[idx], H, canvas_bounds)
        if warped is not None:
            warped_images.append(warped)
            offsets.append(offset)
            print(f"  图像 {idx} 变形完成: {warped.shape}")
    
    if not warped_images:
        print("没有成功变形的图像")
        return None
    
    result = blend_multi_images(images, warped_images, offsets)
    
    return result

def stitch_case(case_name, input_dir, output_dir):
    case_output_dir = os.path.join(output_dir, case_name)
    output_path = os.path.join(case_output_dir, f"{case_name}_stitched.JPG")
    
    if os.path.exists(output_path):
        print(f"[{case_name}] Already exists, skipping...")
        return
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, case_name, "*")))
    images = [cv2.imread(p) for p in image_paths]
    
    images = [img for img in images if img is not None]
    
    if len(images) < 2:
        print(f"[{case_name}] Skipped: insufficient or unreadable images.")
        return

    print(f"\n[{case_name}] Processing {len(images)} images...")
    
    try:
        stitched_image = stitch_multi_images(images)
    except Exception as e:
        print(f"[{case_name}] Error: {e}")
        return

    if stitched_image is None:
        print(f"[{case_name}] Failed: stitching failed.")
        return

    os.makedirs(case_output_dir, exist_ok=True)
    cv2.imwrite(output_path, stitched_image)
    print(f"[{case_name}] Done: saved to {output_path}")

def main():
    input_root = "data/task2_multiview"
    output_root = "output/task2_multiview"
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