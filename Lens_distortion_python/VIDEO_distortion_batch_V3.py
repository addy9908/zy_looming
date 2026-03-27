# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:11:09 2026

Lens distortation for wide-view camera used for behavior test. Help with Gemini

update 20260326: output video should has the l/w ratio defined by user

@author: yez4
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.optimize import minimize
import os
import time

points = []

def click_event(event, x, y, flags, params):
    """Callback to capture eight user-clicked points. It ONLY appends points."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x, y))
        print(f"Point {len(points)} selected: ({x}, {y})")

def to_3d(v):
    """Converts a 2D vector or array of vectors to 3D by adding a zero Z-component."""
    return np.hstack((v, np.zeros((v.shape[0], 1)))) if len(v.shape) > 1 else np.append(v, 0)

def distance_to_line(p, a, b):
    """Calculates the perpendicular distance of a point from a line using 3D vectors."""
    p_3d, a_3d, b_3d = to_3d(p), to_3d(a), to_3d(b)
    return np.linalg.norm(np.cross(b_3d - a_3d, a_3d - p_3d)) / np.linalg.norm(b_3d - a_3d)

def get_straightness_error(params, src_points, h, w):
    """Error function that measures the straightness of the four sides."""
    k1, cx, cy = params
    camera_matrix = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(np.array([src_points], dtype=np.float32),
                                          camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 8, 2)
    p1, p2, p3, p4, p5, p6, p7, p8 = undistorted_pts[0]
    return sum([distance_to_line(p5, p1, p2), distance_to_line(p6, p2, p3),
                distance_to_line(p7, p3, p4), distance_to_line(p8, p4, p1)])

def get_aspect_ratio():
    """Shows a dialog to get the desired output aspect ratio."""
    root = tk.Tk(); root.withdraw()
    ratio_str = simpledialog.askstring("Aspect Ratio", "Enter desired aspect ratio (e.g., '16:9' or '1.777').\nDefault is 16:9.", initialvalue="16:9")
    if ratio_str:
        try:
            if ':' in ratio_str: w_r, h_r = map(float, ratio_str.split(':')); return w_r / h_r if h_r > 0 else 16/9
            else: return float(ratio_str)
        except (ValueError, ZeroDivisionError): print("Invalid format. Defaulting to 16:9."); return 16/9
    else: print("No input. Defaulting to 16:9."); return 16/9

def fine_tune_and_preview(frame, initial_params, original_corner_points, aspect_ratio):
    h, w = frame.shape[:2]; window_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    k1_auto, cx_auto, cy_auto = initial_params
    initial_k1_slider_pos, initial_cx_slider_pos, initial_cy_slider_pos = int((k1_auto * 10000) + 4000), int(cx_auto), int(cy_auto)
    cv2.createTrackbar("Correction", window_name, initial_k1_slider_pos, 8000, lambda x:None)
    cv2.createTrackbar("Center X", window_name, initial_cx_slider_pos, w, lambda x: None)
    cv2.createTrackbar("Center Y", window_name, initial_cy_slider_pos, h, lambda x: None)
    print(f"\n--- Fine-Tuning to {aspect_ratio:.2f}:1 Ratio ---\nAdjust sliders. Press 'A' to Apply, 'R' to Reset, 'Q' to Quit.")
    last_params = ()
    while True:
        k1_pos, cx_pos, cy_pos = cv2.getTrackbarPos("Correction", window_name), cv2.getTrackbarPos("Center X", window_name), cv2.getTrackbarPos("Center Y", window_name)
        if (k1_pos, cx_pos, cy_pos) != last_params:
            k1_fine = (k1_pos - 4000) / 10000.0
            temp_cam_matrix = np.array([[w, 0, cx_pos], [0, w, cy_pos], [0,0,1]], dtype=np.float32)
            temp_dist_coeffs = np.array([k1_fine, 0, 0, 0, 0], dtype=np.float32)
            new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(temp_cam_matrix, temp_dist_coeffs, (w,h), 1, (w,h))
            lens_corrected_frame = cv2.undistort(frame, temp_cam_matrix, temp_dist_coeffs, None, new_cam_matrix)
            corrected_corners = cv2.undistortPoints(original_corner_points, temp_cam_matrix, temp_dist_coeffs, P=new_cam_matrix).reshape(-1, 2)
            max_width = int(max(np.linalg.norm(corrected_corners[0] - corrected_corners[1]), np.linalg.norm(corrected_corners[2] - corrected_corners[3])))
            max_height = int(max_width / aspect_ratio)
            dst_points = np.array([[0,0], [max_width-1,0], [max_width-1,max_height-1], [0,max_height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(corrected_corners, dst_points)
            preview_frame = cv2.warpPerspective(lens_corrected_frame, M, (max_width, max_height))
            cv2.imshow(window_name, preview_frame); last_params = (k1_pos, cx_pos, cy_pos)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27: return 'quit'
        if key == ord('a'): return (k1_fine, cx_pos, cy_pos, M, (max_width, max_height), new_cam_matrix)
        if key == ord('r'):
            cv2.setTrackbarPos("Correction", window_name, initial_k1_slider_pos); cv2.setTrackbarPos("Center X", window_name, initial_cx_slider_pos); cv2.setTrackbarPos("Center Y", window_name, initial_cy_slider_pos)

def generate_output_path(input_path):
    directory, filename = os.path.split(input_path); output_folder = os.path.join(directory, "Distortion_correction"); os.makedirs(output_folder, exist_ok=True)
    filename_without_ext, _ = os.path.splitext(filename); timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_folder, f"{filename_without_ext}_{timestamp}.mp4")

def print_bonsai_parameters(camera_matrix, dist_coeffs, perspective_matrix, output_dims):
    """Prints the final parameters in a format ready for Bonsai."""
    print("\n\n--- Bonsai Workflow Parameters ---")
    print("Copy and paste these values into the corresponding nodes in your Bonsai workflow.")
    
    print("\n1. For the 'Undistort' node:")
    print("   - CameraMatrix (3x3 Matrix):")
    print(f"     [[{camera_matrix[0,0]:.4f}, {camera_matrix[0,1]:.4f}, {camera_matrix[0,2]:.2f}],")
    print(f"      [{camera_matrix[1,0]:.4f}, {camera_matrix[1,1]:.4f}, {camera_matrix[1,2]:.2f}],")
    print(f"      [{camera_matrix[2,0]:.4f}, {camera_matrix[2,1]:.4f}, {camera_matrix[2,2]:.2f}]]")
    
    print("\n   - DistortionCoefficients (1x5 Vector):")
    print(f"     [[{dist_coeffs[0]:.6f}, {dist_coeffs[1]:.6f}, {dist_coeffs[2]:.6f}, {dist_coeffs[3]:.6f}, {dist_coeffs[4]:.6f}]]")

    print("\n2. For the 'WarpPerspective' node:")
    print("   - Transform (3x3 Homography Matrix M):")
    print(f"     [[{perspective_matrix[0,0]:.6f}, {perspective_matrix[0,1]:.6f}, {perspective_matrix[0,2]:.2f}],")
    print(f"      [{perspective_matrix[1,0]:.6f}, {perspective_matrix[1,1]:.6f}, {perspective_matrix[1,2]:.2f}],")
    print(f"      [{perspective_matrix[2,0]:.6f}, {perspective_matrix[2,1]:.6f}, {perspective_matrix[2,2]:.2f}]]")
    
    print("\n   - Size (Width, Height):")
    print(f"     ({output_dims[0]}, {output_dims[1]})")
    print("\n------------------------------------")

def main():
    global points
    root = tk.Tk(); root.withdraw()
    video_paths = filedialog.askopenfilenames(title="Select one or more video files for batch processing",
                                              filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
                                              )
    if not video_paths: print("No files selected."); return
    
    cap = cv2.VideoCapture(video_paths[0]); ret, frame = cap.read()
    if not ret: print("Error reading frame."); cap.release(); return
    h, w = frame.shape[:2]
    
    # --- UNIFIED POINT SELECTION LOOP ---
    points = []
    window_title = "Select 8 Points clockwise from top-left (4 corners then 4 mid-lines) | ENTER=Done | C=Clear | Q/ESC=Quit"
    cv2.namedWindow(window_title); cv2.setMouseCallback(window_title, click_event)
    print(f"\n--- Point Selection on {os.path.basename(video_paths[0])} ---\n1-4: Corners (GREEN) | 5-8: Midpoints (YELLOW)\nPress ENTER after 8 points.")
    
    while True:
        # Create a fresh copy to draw on in every loop iteration
        display_frame = frame.copy()
        # Draw all current points on the fresh frame
        for i, p in enumerate(points):
            color = (0, 255, 0) if i < 4 else (0, 255, 255)
            cv2.circle(display_frame, p, 7, color, -1)
            cv2.putText(display_frame, str(i + 1), (p[0] + 15, p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow(window_title, display_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27: # Q or ESC to quit
            cv2.destroyAllWindows(); cap.release(); print("Quit command received. Exiting."); return
        if key == ord('c'): # C to clear points
            print("Clearing points. Please select 8 points again.")
            points = [] # Reset the list, the loop will redraw a clean frame
            continue
        if key == 13 and len(points) == 8: # ENTER to confirm
            break
    
    cv2.destroyAllWindows()
    
    print("\nOptimizing lens parameters..."); result = minimize(get_straightness_error, [0.0, w/2, h/2], args=(np.array(points, dtype=np.float32), h, w), method='L-BFGS-B', bounds=[(-0.5, 0.5), (0, w), (0, h)])
    k1_auto, cx_auto, cy_auto = result.x; print(f"Auto-detection Complete! k1: {k1_auto:.4f}, Center: ({cx_auto:.1f}, {cy_auto:.1f})")

    aspect_ratio = get_aspect_ratio()

    final_params_tuple = fine_tune_and_preview(frame, (k1_auto, cx_auto, cy_auto), np.array([points[:4]], dtype=np.float32), aspect_ratio)
    cv2.destroyAllWindows(); cap.release()
    if final_params_tuple == 'quit': print("Quit command received. Exiting."); return
    
    k1_final, cx_final, cy_final, M_final, output_dims, new_cam_matrix_final = final_params_tuple
    final_camera_matrix = np.array([[w, 0, cx_final], [0, w, cy_final], [0, 0, 1]], dtype=np.float32)
    final_dist_coeffs = np.array([k1_final, 0, 0, 0, 0], dtype=np.float32)
    print(f"\nFinal parameters locked! Applying to all {len(video_paths)} files.")
    print_bonsai_parameters(final_camera_matrix, final_dist_coeffs, M_final, output_dims)
    
    # --- ALGORITHMIC IMPROVEMENT STARTS HERE ---

    # 1. Pre-compute the lens correction maps (map1, map2)
    # These maps tell `remap` where to find each pixel for the undistort operation.
    map1, map2 = cv2.initUndistortRectifyMap(final_camera_matrix, final_dist_coeffs, None, new_cam_matrix_final, (w,h), cv2.CV_32FC1)
    
    # 2. Pre-compute the perspective transform on the lens correction maps.
    # This creates a NEW, combined map that does both operations at once.
    # We apply the perspective matrix M to the (x,y) coordinates stored in map1 and map2.
    # This is an advanced operation that essentially combines the two transforms.
    # (A simpler, though slightly less efficient way, not shown here, is to remap, then warp).
    # For simplicity and correctness in this context, we will actually apply the two steps,
    # but the principle is what matters. A true map composition is more complex.
    # Let's stick to the most direct optimization: GPU is better. For CPU, let's optimize video writing.

    start_time = time.time()
    for i, video_path in enumerate(video_paths):
        output_path = generate_output_path(video_path)
        print(f"\n--- Processing file {i+1}/{len(video_paths)} ---\nSource: {video_path}\nOutput: {output_path}")
        batch_cap = cv2.VideoCapture(video_path)
        if not batch_cap.isOpened(): print(f"ERROR: Could not open {video_path}. Skipping."); continue
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), batch_cap.get(cv2.CAP_PROP_FPS), output_dims)
        total_frames = int(batch_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(total_frames):
            ret_batch, f = batch_cap.read()
            if not ret_batch: break
            # lens_corrected = cv2.undistort(f, final_camera_matrix, final_dist_coeffs, None, new_cam_matrix_final)
            # This is slightly faster than separate undistort/warp
            lens_corrected = cv2.remap(f, map1, map2, cv2.INTER_LINEAR)
            final_frame = cv2.warpPerspective(lens_corrected, M_final, output_dims)
            out.write(final_frame); print(f"Processing frame {frame_num+1}/{total_frames}", end='\r')
        out.release(); batch_cap.release(); print(f"\nFinished file {i+1}/{len(video_paths)}.")
    
    end_time = time.time(); total_duration = end_time - start_time
    minutes, seconds = int(total_duration // 60), total_duration % 60
    print("\n\n-----------------------------------------")
    print("All files processed successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds:.2f} seconds.")
    print("-----------------------------------------")

if __name__ == '__main__':
    main()
