# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:12:58 2026

max_width = h # Set the output width to the original video's height

In case of hard to tune, just save the video from the setted frame

@author: yez4
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from scipy.optimize import minimize
import os
import time
import pandas as pd
import re
import datetime  # <--- ADD THIS LINE


# --- Global variable for mouse clicks ---
points = []

# --- All the correction logic functions from before remain the same ---
def click_event(event, x, y, flags, params):
    global points; frame_for_clicks = params['frame']
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x, y))

def to_3d(v):
    return np.hstack((v, np.zeros((v.shape[0], 1)))) if len(v.shape) > 1 else np.append(v, 0)

def distance_to_line(p, a, b):
    p_3d, a_3d, b_3d = to_3d(p), to_3d(a), to_3d(b)
    return np.linalg.norm(np.cross(b_3d - a_3d, a_3d - p_3d)) / np.linalg.norm(b_3d - a_3d)

def get_straightness_error(params, src_points, h, w):
    k1, cx, cy = params; camera_matrix = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(np.array([src_points], dtype=np.float32), camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 8, 2)
    p1, p2, p3, p4, p5, p6, p7, p8 = undistorted_pts[0]
    return sum([distance_to_line(p5, p1, p2), distance_to_line(p6, p2, p3), distance_to_line(p7, p3, p4), distance_to_line(p8, p4, p1)])

class VideoCorrectorApp:
    def __init__(self, root):
        self.root = root; 
        self.root.title("Video Analysis & Correction Station"); 
        self.root.geometry("1100x800")
        self.cam_df, self.looming_df = None, None; 
        self.cam_file_path, self.looming_file_path, self.video_path = "", "", ""
        self.correction_params = None
        self.log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_log.csv")
        style = ttk.Style(self.root); style.theme_use("clam")
        self.create_widgets()
        self.load_last_parameters_from_log()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Style Configuration (The New Part) ---
        style = ttk.Style(self.root)
        # Define a new custom style for red text
        style.configure("Red.TLabel", foreground="red")

        # --- Frame 1: File Loading ---
        load_frame = ttk.LabelFrame(main_frame, text="1. Load All Files", padding="10")
        load_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Video File", command=self.load_video_file).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.lbl_video_file = ttk.Label(load_frame, text="No video loaded.", width=60, relief=tk.SUNKEN)
        self.lbl_video_file.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_auto_find = ttk.Button(load_frame, text="Auto-Find Companion Files", command=self.auto_find_files, state=tk.DISABLED)
        self.btn_auto_find.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ttk.Button(load_frame, text="Load Cam CSV", command=self.load_cam_file).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.lbl_cam_file = ttk.Label(load_frame, text="No file loaded.", relief=tk.SUNKEN)
        self.lbl_cam_file.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # ADD THIS NEW LINE right below it:
        ttk.Button(load_frame, text="Clear / Reset Sync Data", command=self.reset_data).grid(row=1, column=2, padx=5, pady=5, sticky="ew")


        ttk.Button(load_frame, text="Load Looming CSV", command=self.load_looming_file).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.lbl_looming_file = ttk.Label(load_frame, text="No file loaded.", relief=tk.SUNKEN)
        self.lbl_looming_file.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # --- Frame 2: Looming Data Table ---
        table_frame = ttk.LabelFrame(main_frame, text="Looming Data (Click a row to select event)", padding="10")
        table_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        main_frame.grid_rowconfigure(1, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=("TimeStamp", "Millis", "TTL"), show='headings')
        for col in ["TimeStamp", "Millis", "TTL"]:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_treeview(c, False))
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # --- Frame 3: Synchronization ---
        sync_frame = ttk.LabelFrame(main_frame, text="2. Find Synchronization Frames", padding="10")
        sync_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(sync_frame, text="Pre-event Time (minutes):").grid(row=0, column=0, sticky="w", padx=5)
        self.pre_event_minutes = tk.StringVar(value="1")
        ttk.Entry(sync_frame, textvariable=self.pre_event_minutes, width=10).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(sync_frame, text="Find Sync Frames", command=self.find_sync_frames).grid(row=0, column=2, padx=10)
        self.lbl_sync_result = ttk.Label(sync_frame, text="Results will appear here.", foreground="blue")
        self.lbl_sync_result.grid(row=1, column=0, columnspan=3, pady=5)

        # --- Frame 4: Correction ---
        corr_frame = ttk.LabelFrame(main_frame, text="3. Process Video", padding="10")
        corr_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(corr_frame, text="Start correction from frame:").grid(row=0, column=0, sticky="w", padx=5)
        self.start_frame = tk.StringVar(value="0")
        ttk.Entry(corr_frame, textvariable=self.start_frame, width=10).grid(row=0, column=1, sticky="w", padx=5)
        self.lbl_params_status = ttk.Label(corr_frame, text="Correction Parameters: Not Set", foreground="red")
        self.lbl_params_status.grid(row=0, column=2, padx=10, sticky="w")
        
        # ADDED BUTTON FOR JUST TRIM
        ttk.Button(corr_frame, text="Start Correction & Logging", command=self.start_correction_process).grid(row=1, column=0, columnspan=2, pady=10, ipady=5, sticky="ew")
        ttk.Button(corr_frame, text="Just Trim & Save (No Correction)", command=self.just_trim_video).grid(row=1, column=2, pady=10, ipady=5, sticky="ew", padx=5)

        # --- Status Bar (The Modified Line) ---
        self.status_var = tk.StringVar(value="Ready.")
        # Apply the new "Red.TLabel" style here
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, style="Red.TLabel")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        # --- Progress Bar for Loading and Processing ---
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 0))


    def load_video_file(self): 
        # path = self._load_file("video", self.lbl_video_file,)
        self.reset_data()  # <--- ADD THIS LINE HERE
        path = filedialog.askopenfilename(title="Select video file",
                                          filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
                                          )
        
        if path: 
            self.lbl_video_file.config(text=os.path.basename(path))
            self.video_path = path
            self.btn_auto_find.config(state=tk.NORMAL)

            self.status_var.set("video file loaded.")
            
            # more automatically
            try:
                self.auto_find_files()
                self.find_sync_frames()
                self.status_var.set("All files loaded and frame found.")
            except:
                self.status_var.set("auto_find or find sync fail.")
                
        else: 
            self.status_var.set("Ready.")
            self.btn_auto_find.config(state=tk.DISABLED)
            
        self.root.update()    

        
    def load_cam_file(self): self.cam_file_path, self.cam_df = self._load_csv_file("Cam", self.lbl_cam_file, ["Millis", "Cam_Frame"])
    def load_looming_file(self):
        self.looming_file_path, self.looming_df = self._load_csv_file("Looming", self.lbl_looming_file, ["TimeStamp", "Millis", "TTL_looming"])
        if self.looming_df is not None: self.populate_looming_table()

    def reset_data(self):
        """Clears companion files and sync data when a new video is loaded."""
        self.cam_df, self.looming_df = None, None
        self.cam_file_path, self.looming_file_path = "", ""
        self.lbl_video_file.config(text="No file loaded.")
        self.lbl_cam_file.config(text="No file loaded.")
        self.lbl_looming_file.config(text="No file loaded.")
        self.lbl_sync_result.config(text="Results will appear here.")
        self.start_frame.set("0")
        for i in self.tree.get_children(): 
            self.tree.delete(i)
        self.status_var.set("Workspace reset for new video.")


    def _load_file(self, file_type, label_widget):
        path = filedialog.askopenfilename(title=f"Select {file_type} file")
        if path: label_widget.config(text=os.path.basename(path)); self.status_var.set(f"{file_type.capitalize()} file loaded."); return path
        return ""
    
    def _load_csv_file(self, file_type, label_widget, required_cols, path=None):
        if not path: path = self._load_file(file_type, label_widget)
        if path:
            try:
                df = pd.read_csv(path); assert all(col in df.columns for col in required_cols)
                label_widget.config(text=os.path.basename(path)) # Update label even on auto-find
                return path, df
            except Exception as e: messagebox.showerror("Error", f"Failed to load or validate {file_type} file:\n{e}"); return "", None
        return "", None
    
    def auto_find_files(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Load a video file first.")
            return
    
        directory = os.path.dirname(self.video_path)
        video_filename = os.path.basename(self.video_path)
        
        # Extract base name and the FULL timestamp (e.g., 2026-03-05T14_08_57)
        match = re.match(r'^(.*?)_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', video_filename)
        
        if not match:
            messagebox.showerror("Error", "Could not determine base name and timestamp from filename.\nExpected pattern: [BaseName]_[YYYY-MM-DD]T[HH_MM_SS].avi")
            return
            
        base_name = match.group(1)
        video_ts_str = match.group(2)
        date_string = video_ts_str.split('T')[0] # "2026-03-05"
        
        # Parse the video's exact time
        try:
            video_dt = datetime.datetime.strptime(video_ts_str, "%Y-%m-%dT%H_%M_%S")
        except ValueError:
            messagebox.showerror("Error", "Timestamp format error in video filename.")
            return
        
        self.status_var.set(f"Searching for files closest AFTER: {video_ts_str}")
        
        found_cam, found_looming = [], []
        for f in os.listdir(directory):
            if f.startswith(f"{base_name}_cam_") and date_string in f and f.endswith(".csv"):
                found_cam.append(f)
            if f.startswith(f"{base_name}_Looming_") and date_string in f and f.endswith(".csv"):
                found_looming.append(f)

        # --- Helper function: closest file AFTER (or exactly at) video timestamp ---
        def get_closest_after_file(file_list):
            if not file_list: return None
            
            closest_file = None
            min_diff = float('inf') # Start with an infinitely large difference
            
            for f in file_list:
                m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', f)
                if m:
                    cand_ts_str = m.group(1)
                    try:
                        cand_dt = datetime.datetime.strptime(cand_ts_str, "%Y-%m-%dT%H_%M_%S")
                        
                        # Calculate difference. Positive means candidate is AFTER video
                        diff = (cand_dt - video_dt).total_seconds()
                        
                        # MUST be >= 0 (after or same time) AND smaller than previous best
                        if diff >= -5 and diff < min_diff:
                            min_diff = diff
                            closest_file = f
                    except ValueError: 
                        continue
            return closest_file

        best_cam = get_closest_after_file(found_cam)
        best_looming = get_closest_after_file(found_looming)
        
        cam_loaded, looming_loaded = False, False
        
        # Load best Cam file
        if best_cam:
            cam_path = os.path.join(directory, best_cam)
            self.cam_file_path, self.cam_df = self._load_csv_file("Cam", self.lbl_cam_file, ["Millis", "Cam_Frame"], path=cam_path)
            if self.cam_df is not None: cam_loaded = True
        else:
            messagebox.showwarning("Auto-Find", "No matching Cam file found that was created AFTER the video.")
        
        # Load best Looming file
        if best_looming:
            looming_path = os.path.join(directory, best_looming)
            self.looming_file_path, self.looming_df = self._load_csv_file("Looming", self.lbl_looming_file, ["TimeStamp", "Millis", "TTL_looming"], path=looming_path)
            if self.looming_df is not None:
                looming_loaded = True
                self.populate_looming_table()
        else:
            messagebox.showwarning("Auto-Find", "No matching Looming file found that was created AFTER the video.")
        
        if cam_loaded and looming_loaded:
            self.status_var.set("Successfully auto-loaded valid companion files.")
        else:
            self.status_var.set("Auto-find complete. Please verify files.")
    
    def populate_looming_table(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for _, row in self.looming_df.iterrows(): self.tree.insert("", "end", values=tuple(row[c] for c in ["TimeStamp", "Millis", "TTL_looming"]))

    def sort_treeview(self, col, reverse):
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        try: l.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError: l.sort(reverse=reverse)
        for index, (val, k) in enumerate(l): self.tree.move(k, '', index)
        self.tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))

    def find_sync_frames(self):
        if self.cam_df is None or self.looming_df is None: messagebox.showwarning("Warning", "Load Cam and Looming CSV files first."); return
        try:
            selected_item = self.tree.selection()
            if selected_item: looming_millis = float(self.tree.item(selected_item[0])['values'][1]); self.status_var.set(f"Using selected event at {looming_millis} ms.")
            else: first_looming_event = self.looming_df[self.looming_df['TTL_looming'] == True].iloc[0]; looming_millis = first_looming_event['Millis']; self.status_var.set("Using first 'True' looming event by default.")
            closest_cam_row = self.cam_df.loc[(self.cam_df['Millis'] - looming_millis).abs().idxmin()]
            self.looming_on_frame, self.looming_on_time = int(closest_cam_row['Cam_Frame']), int(closest_cam_row['Millis'])
            pre_event_minutes = float(self.pre_event_minutes.get()); pre_event_target_millis = looming_millis - (pre_event_minutes * 60 * 1000)
            closest_pre_event_row = self.cam_df.loc[(self.cam_df['Millis'] - pre_event_target_millis).abs().idxmin()]
            self.pre_event_frame = int(closest_pre_event_row['Cam_Frame'])
            self.lbl_sync_result.config(text=f"Looming Event Frame: {self.looming_on_frame} | Pre-Event Frame (-{pre_event_minutes} min): {self.pre_event_frame} | Total Frame: {self.cam_df['Cam_Frame'].max()}")
            self.start_frame.set(str(self.pre_event_frame))
        except Exception as e: messagebox.showerror("Error", f"Could not find sync frames:\n{e}")

    def verify_video_ready(self):
        """Waits for cloud download and returns the ready cap object."""
        if self.cam_df is None: 
            return cv2.VideoCapture(self.video_path)
        
        target_frames = int(self.cam_df['Cam_Frame'].max())
        self.status_var.set(f"Connecting to cloud file: 0 of {target_frames} frames detected...")
        self.progress.config(mode='determinate', maximum=target_frames)
        # --- 2. FORCE THE WINDOW TO REDRAW NOW ---
        self.root.update_idletasks()
        self.root.update()

        while True:
            cap = cv2.VideoCapture(self.video_path)
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if actual_frames >= target_frames:
                # SUCCESS: Return the opened cap to be used immediately
                self.progress['value'] = 0
                self.status_var.set("Video fully loaded. Starting process...")
                self.root.update()
                return cap 
                
            # STILL DOWNLOADING: Update UI and wait
            self.status_var.set(f"Cloud sync in progress: {actual_frames} of {target_frames} frames...")
            self.progress['value'] = actual_frames
            self.root.update_idletasks()
            self.root.update() # This makes the text update every 2 seconds
            
            cap.release() # Close handle before sleeping to allow cloud to write to file
            time.sleep(2) 

    def just_trim_video(self):
        if not self.video_path: messagebox.showwarning("Warning", "Please load a video file first."); return
        # 1. Get the handle (this waits for the download)
        cap = self.verify_video_ready()
        
        try: 
            start_frame_num = int(self.start_frame.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if start_frame_num >= total_frames:
                messagebox.showerror("Error", "Start frame out of bounds."); cap.release(); return
        except: 
            messagebox.showerror("Error", "Invalid start frame."); cap.release(); return

          
        w, h = int(cap.get(3)), int(cap.get(4))
        out_path = self.generate_output_path(self.video_path)
        
        self.progress.config(mode='determinate', maximum=total_frames - start_frame_num)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))
        
        processed_frames = 0
        while True:
            ret, f = cap.read()
            if not ret: break
            writer.write(f)
            processed_frames += 1
            
            # # --- THE FIX: Only update GUI every 30 frames to stop freezing ---
            # if processed_frames % 30 == 0:
            self.progress['value'] = processed_frames # <--- UPDATE BAR
            self.status_var.set(f"Trimming Frame: {start_frame_num + processed_frames}/{total_frames}")
            self.root.update()
                
        writer.release(); cap.release(); self.progress['value'] = 0
        
        if processed_frames == 0:
            messagebox.showerror("Error", "Failed to process frames. Video may be corrupted.")
            return
            
        self.log_analysis_results(out_path, None)
        self.status_var.set(f"Trimmed video saved successfully!\nSaved to: {out_path}")
        # messagebox.showinfo("Success", f"Trimmed video saved successfully!\nSaved to: {out_path}")
        # self.status_var.set("Ready.")

    def start_correction_process(self):
        if not self.video_path: messagebox.showwarning("Warning", "Please load a video file first."); return
        cap = self.verify_video_ready()
        
        try: start_frame_num = int(self.start_frame.get()); assert start_frame_num >= 0
        except: messagebox.showerror("Error", "Enter a valid, non-negative start frame number."); return
        if self.correction_params:
            user_choice = messagebox.askyesnocancel("Parameters Found", "Saved correction parameters found.\n\nYES = Reuse the saved parameters.\nNO = Create new parameters for this video.")
            if user_choice is None: self.status_var.set("Correction cancelled."); return
            if user_choice: self.process_video(self.correction_params, start_frame_num, cap); return
        self.run_calibration_workflow(start_frame_num,cap)

    def run_calibration_workflow(self, start_frame_num, cap):
        global points
        self.status_var.set("Starting calibration..."); self.root.update()
        ret, frame = cap.read()
        if not ret: cap.release(); messagebox.showerror("Error", "Could not read frame from video."); return
        h, w = frame.shape[:2]
        points = []; window_title = "Select Points | ENTER=Done | C=Clear | Q/ESC=Quit"
        cv2.namedWindow(window_title); cv2.setMouseCallback(window_title, click_event, {'frame': frame})
        
        while True: # UNIFIED POINT SELECTION LOOP
            display_frame = frame.copy()
            for i, p in enumerate(points):
                color = (0, 255, 0) if i < 4 else (0, 255, 255)
                cv2.circle(display_frame, p, 7, color, -1); cv2.putText(display_frame, str(i + 1), (p[0] + 15, p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27: cv2.destroyAllWindows(); cap.release(); self.status_var.set("Cancelled at point selection."); return
            if key == ord('c'):
                points = []
                print("Points cleared. Please select 8 points again.")
                continue # Immediately restart the loop to show the cleared frame
            if key == 13 and len(points) == 8: break
        cv2.destroyAllWindows()
        
        result = minimize(get_straightness_error, [0.0, w/2, h/2], args=(np.array(points, dtype=np.float32), h, w), method='L-BFGS-B', bounds=[(-0.5, 0.5), (0, w), (0, h)])
        k1_auto, cx_auto, cy_auto = result.x
        aspect_ratio = self.get_aspect_ratio_from_dialog()
        final_params_tuple = self.fine_tune_and_preview(frame, (k1_auto, cx_auto, cy_auto), np.array([points[:4]], dtype=np.float32), aspect_ratio)
        cv2.destroyAllWindows(); cap.release()
        if final_params_tuple == 'quit': self.status_var.set("Cancelled at fine-tuning."); return
        self.correction_params = final_params_tuple
        self.lbl_params_status.config(text="Correction Parameters: Ready", foreground="green")
        self.process_video(self.correction_params, start_frame_num, cap)

    def process_video(self, params, start_frame_num, cap):
        k1_final, cx_final, cy_final, M_final, output_dims, new_cam_matrix_final = params
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        final_camera_matrix = np.array([[w, 0, cx_final], [0, w, cy_final], [0, 0, 1]], dtype=np.float32)
        final_dist_coeffs = np.array([k1_final, 0, 0, 0, 0], dtype=np.float32)
        output_path = self.generate_output_path(self.video_path)
        self.log_analysis_results(output_path, params)
        
        self.status_var.set(f"Processing: {os.path.basename(self.video_path)}"); self.root.update()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
        
        # Configure progress bar
        self.progress.config(mode='determinate', maximum=total_frames - start_frame_num, value=0)
 
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), output_dims)

        processed_frames = 0
        
        while True:
            ret, f = cap.read()
            if not ret: break
            lens_corrected = cv2.undistort(f, final_camera_matrix, final_dist_coeffs, None, new_cam_matrix_final)
            final_frame = cv2.warpPerspective(lens_corrected, M_final, output_dims)
            out.write(final_frame); processed_frames += 1
            self.progress['value'] = processed_frames # <--- UPDATE BAR
            self.status_var.set(f"Processing frame {start_frame_num + processed_frames}/{total_frames}"); self.root.update()
        out.release(); cap.release(); self.progress['value'] = 0
        messagebox.showinfo("Success", f"Video processed successfully!\nSaved to: {output_path}")
        self.status_var.set("Ready.")

    def log_analysis_results(self, output_video_path, params):
        """
        Logs the complete analysis details to the central CSV log file.
        FINAL ROBUST VERSION: Saves matrices as a simple, flat, semicolon-delimited string.
        Also handles params=None when just trimming without correction.
        """
        if params:
            k1, cx, cy, M, dims, new_cam = params
            M_flat_str = ";".join(map(str, M.flatten()))
            new_cam_flat_str = ";".join(map(str, new_cam.flatten()))
            k1_str, cx_str, cy_str = f"{k1:.6f}", f"{cx:.2f}", f"{cy:.2f}"
            dims_str = f"{dims[0]},{dims[1]}"
        else:
            k1_str, cx_str, cy_str = "NONE", "NONE", "NONE"
            M_flat_str, new_cam_flat_str = "NONE", "NONE"
            dims_str = "ORIGINAL"

        log_data = {
            "Time_analyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "File_path": self.video_path,
            "video_file_name": os.path.basename(self.video_path),
            "Cam_file_name": os.path.basename(self.cam_file_path),
            "looming_file_name": os.path.basename(self.looming_file_path),
            "looming_on_time_ms": getattr(self, 'looming_on_time', ''),
            "looming_on_frame": getattr(self, 'looming_on_frame', ''),
            "pre_event_time_min": self.pre_event_minutes.get(),
            "pre_event_frame": getattr(self, 'pre_event_frame', ''),
            "lens_k1": k1_str,
            "lens_cx": cx_str,
            "lens_cy": cy_str,
            "M_matrix_flat": M_flat_str, 
            "output_dims": dims_str,
            "new_cam_matrix_flat": new_cam_flat_str 
        }
    
        new_log_df = pd.DataFrame([log_data])
    
        if not os.path.exists(self.log_file_path):
            new_log_df.to_csv(self.log_file_path, index=False)
        else:
            new_log_df.to_csv(self.log_file_path, mode='a', header=False, index=False)    

    def load_last_parameters_from_log(self):
        """
        Loads the last used parameters from the central log file on startup.
        FINAL ROBUST VERSION: Parses the flat string and reshapes it into a matrix.
        """
        if not os.path.exists(self.log_file_path):
            return
        try:
            log_df = pd.read_csv(self.log_file_path)
            if log_df.empty:
                return
                
            last_row = log_df.iloc[-1]
            
            # If the last operation was just a trim, don't load parameters
            if str(last_row['lens_k1']) == "NONE":
                return

            k1 = float(last_row['lens_k1'])
            cx = float(last_row['lens_cx'])
            cy = float(last_row['lens_cy'])
            
            # --- THE DEFINITIVE FIX: Split the flat string, convert to float, and reshape ---
            M_flat_list = [float(i) for i in last_row['M_matrix_flat'].split(';')]
            M = np.array(M_flat_list).reshape(3, 3)
            
            dims = tuple(map(int, last_row['output_dims'].split(',')))
            
            new_cam_flat_list = [float(i) for i in last_row['new_cam_matrix_flat'].split(';')]
            new_cam = np.array(new_cam_flat_list).reshape(3, 3)
            
            # Update the shape check as well
            if M.shape != (3, 3): raise ValueError(f"M_matrix has incorrect shape after parsing: {M.shape}")
            if new_cam.shape != (3, 3): raise ValueError(f"new_cam_matrix has incorrect shape after parsing: {new_cam.shape}")
            
            self.correction_params = (k1, cx, cy, M, dims, new_cam)
            
            self.lbl_params_status.config(text="Correction Parameters: Ready (from log)", foreground="green")
            self.status_var.set("Loaded last used parameters from log file.")
        except Exception as e:
            self.status_var.set(f"Could not load parameters from log: {e}")
            # Silencing error message slightly since failure to load just means user will recalibrate.
            # messagebox.showerror("Log File Error", f"Failed to parse the log file '{self.log_file_path}'.\n\nError: {e}\n\nThe log file might be corrupted. You can try deleting it to start fresh.")

    def get_aspect_ratio_from_dialog(self):
        ratio_str = simpledialog.askstring("Aspect Ratio", "Enter desired aspect ratio...", parent=self.root, initialvalue="1:1")
        if ratio_str:
            try:
                if ':' in ratio_str: w, h = map(float, ratio_str.split(':')); return w / h if h > 0 else 1
                else: return float(ratio_str)
            except: return 1
        return 1

    def fine_tune_and_preview(self, frame, initial_params, original_corner_points, aspect_ratio):
        h, w = frame.shape[:2]; window_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"; cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        k1_auto, cx_auto, cy_auto = initial_params
        initial_k1_slider_pos, initial_cx_slider_pos, initial_cy_slider_pos = int((k1_auto * 10000) + 4000), int(cx_auto), int(cy_auto)
        cv2.createTrackbar("Correction", window_name, initial_k1_slider_pos, 8000, lambda x:None); cv2.createTrackbar("Center X", window_name, initial_cx_slider_pos, w, lambda x: None); cv2.createTrackbar("Center Y", window_name, initial_cy_slider_pos, h, lambda x: None)
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
                
                max_width = h # Set the output width to the original video's height
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

    # --- FIX 1: Make generate_output_path a method of the class ---
    def generate_output_path(self, input_path):
        directory, filename = os.path.split(input_path); output_folder = os.path.join(directory, "output"); os.makedirs(output_folder, exist_ok=True)
        filename_without_ext, _ = os.path.splitext(filename); timestamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(output_folder, f"{filename_without_ext}_{timestamp}.mp4")

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoCorrectorApp(root)
    root.mainloop()
