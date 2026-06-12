# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:09:58 2026
1. for simple task like trimming and save, we do the batch!
2. for lens correction, we do one by one
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
import datetime

# --- Global variable for mouse clicks ---
points = []

# --- All the correction logic functions from before remain the same ---
def click_event(event, x, y, flags, params):
    global points #; frame_for_clicks = params['frame']
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

def draw_dashed_rectangle(img, top_left, bottom_right, color, thickness=1, dash_length=10):
    """Draws a dashed rectangle on the given image."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Helper to draw a single dashed line
    def draw_dashed_line(pt1, pt2):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        num_dashes = int(dist / (dash_length * 2))
        for i in range(num_dashes):
            start_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i * 2) / (num_dashes * 2))
            start_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i * 2) / (num_dashes * 2))
            end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i * 2 + 1) / (num_dashes * 2))
            end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i * 2 + 1) / (num_dashes * 2))
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
    # Draw the four sides
    draw_dashed_line((x1, y1), (x2, y1)) # Top
    draw_dashed_line((x2, y1), (x2, y2)) # Right
    draw_dashed_line((x2, y2), (x1, y2)) # Bottom
    draw_dashed_line((x1, y2), (x1, y1)) # Left

class VideoCorrectorApp:
    def __init__(self, root):
        self.root = root; 
        self.root.title("Video Analysis & Correction Station"); 
        self.root.geometry("1200x1000")
        self.cam_df, self.looming_df = None, None; 
        self.cam_file_path, self.looming_file_path, self.video_path = "", "", ""
        self.correction_params = None
        self.log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_log.csv")
        
        style = ttk.Style(self.root); style.theme_use("clam")
        
        self.create_widgets()
        self.load_last_parameters_from_log()

    def create_widgets(self):
        # 1. Setup Shared Status Bar FIRST so it is always at the bottom
        style = ttk.Style(self.root)
        style.configure("Red.TLabel", foreground="red")
        
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, style="Red.TLabel")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Increase row height for all Treeviews
        style.configure("Treeview", rowheight=30)

        # 2. Create Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_single = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_single, text="Single Video Analysis")
        self.notebook.add(self.tab_batch, text="Batch Trim & Save")

        self.create_single_mode_widgets()
        self.create_batch_mode_widgets()

    def create_single_mode_widgets(self):
        """Builds the original interface inside Tab 1"""
        main_frame = ttk.Frame(self.tab_single, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

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
        ttk.Button(corr_frame, text="Start Correction & Logging", command=self.start_correction_process).grid(row=1, column=0, columnspan=3, pady=10, ipady=5)

    def create_batch_mode_widgets(self):
        """Builds the new Batch Trim interface inside Tab 2"""
        batch_frame = ttk.Frame(self.tab_batch, padding="10")
        batch_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Tools ---
        tools_frame = ttk.Frame(batch_frame)
        tools_frame.pack(fill=tk.X, pady=5)
        ttk.Button(tools_frame, text="1. Load Videos", command=self.batch_load_videos).pack(side=tk.LEFT, padx=10)
        ttk.Button(tools_frame, text="2. Auto-Map CSVs", command=self.batch_auto_map).pack(side=tk.LEFT, padx=10)
        ttk.Button(tools_frame, text="Save Mapping CSV", command=self.batch_save_mapping).pack(side=tk.RIGHT, padx=10)
        ttk.Button(tools_frame, text="Load Mapping CSV", command=self.batch_load_mapping).pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(tools_frame, text="Clear List", command=self.batch_clear_list).pack(side=tk.RIGHT, padx=10)
        ttk.Button(tools_frame, text="Select All", command=self.batch_select_all).pack(side=tk.RIGHT, padx=10)

        # --- Table ---
        self.batch_tree = ttk.Treeview(batch_frame, columns=("No", "Video", "Cam", "Looming", "Status"), show='headings')
        self.batch_tree.heading("No", text="#")
        self.batch_tree.heading("Video", text="Video File")
        self.batch_tree.heading("Cam", text="Cam File")
        self.batch_tree.heading("Looming", text="Looming File")
        self.batch_tree.heading("Status", text="Status")
        
        self.batch_tree.column("No", width=50, anchor=tk.CENTER)
        self.batch_tree.column("Video", width=250)
        self.batch_tree.column("Cam", width=250)
        self.batch_tree.column("Looming", width=250)
        self.batch_tree.column("Status", width=100)
        
        vsb = ttk.Scrollbar(batch_frame, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=vsb.set)
        self.batch_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        vsb.place(in_=self.batch_tree, relx=1.0, rely=0, relheight=1.0, anchor="ne")

        # --- Bottom Execution ---
        run_frame = ttk.Frame(batch_frame)
        run_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(run_frame, text="Set Cam for Selected", command=lambda: self.batch_set_file("Cam")).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_frame, text="Set Looming for Selected", command=lambda: self.batch_set_file("Looming")).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(run_frame, text="3. Pre-event Time (min):").pack(side=tk.LEFT, padx=(30, 5))
        self.batch_pre_event = tk.StringVar(value="1")
        ttk.Entry(run_frame, textvariable=self.batch_pre_event, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(run_frame, text="4. Start Batch Trim & Save", command=self.run_batch_trim).pack(side=tk.RIGHT, padx=5, ipady=5)

    # --- Helper: Closest After Logic ---
    def get_closest_after_file(self, video_path, prefix):
        directory = os.path.dirname(video_path)
        vid_name = os.path.basename(video_path)
        match = re.match(r'^(.*?)_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', vid_name)
        if not match: return None
        
        base_name, vid_ts_str = match.group(1), match.group(2)
        date_str = vid_ts_str.split('T')[0]
        vid_dt = datetime.datetime.strptime(vid_ts_str, "%Y-%m-%dT%H_%M_%S")
        
        best_file, min_diff = None, float('inf')
        for f in os.listdir(directory):
            if f.startswith(f"{base_name}_{prefix}_") and date_str in f and f.endswith(".csv"):
                m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', f)
                if m:
                    try:
                        cand_dt = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H_%M_%S")
                        diff = (cand_dt - vid_dt).total_seconds()
                        if diff >= -5 and diff < min_diff:
                            min_diff = diff
                            best_file = os.path.join(directory, f)
                    except: continue
        return best_file

    # ==========================================
    # === SINGLE VIDEO ANALYSIS LOGIC (Tab 1) ==
    # ==========================================
    def load_video_file(self): 
        self.reset_data()  # <--- ADD THIS LINE HERE
        path = filedialog.askopenfilename(title="Select video file",
                                          filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
                                          )
        
        if path: 
            self.lbl_video_file.config(text=os.path.basename(path))
            self.video_path = path
            self.btn_auto_find.config(state=tk.NORMAL)
            self.load_last_parameters_from_log

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
        
    def load_cam_file(self): self.cam_file_path, self.cam_df = self._load_csv_file("Cam", self.lbl_cam_file, ["Millis", "Cam_Frame"])
    def load_looming_file(self):
        self.looming_file_path, self.looming_df = self._load_csv_file("Looming", self.lbl_looming_file, ["TimeStamp", "Millis", "TTL_looming"])
        if self.looming_df is not None: self.populate_looming_table()

    def _load_csv_file(self, file_type, label_widget, required_cols, path=None):
        if not path: 
            path = filedialog.askopenfilename(title=f"Select {file_type} file",
                                          filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
                                          )
        if path:
            label_widget.config(text=os.path.basename(path))
            self.status_var.set(f"{file_type.capitalize()} file loaded.");
            try:
                df = pd.read_csv(path); assert all(col in df.columns for col in required_cols)
                label_widget.config(text=os.path.basename(path)) # Update label even on auto-find
                return path, df
            except Exception as e: messagebox.showerror("Error", f"Failed to load or validate {file_type} file:\n{e}"); return "", None
        return "", None
    
    def auto_find_files(self):
        cam = self.get_closest_after_file(self.video_path, "cam")
        if cam: self.cam_file_path, self.cam_df = self._load_csv_file("Cam", self.lbl_cam_file, ["Millis", "Cam_Frame"], path=cam)
        loom = self.get_closest_after_file(self.video_path, "Looming")
        if loom: 
            self.looming_file_path, self.looming_df = self._load_csv_file("Looming", self.lbl_looming_file, ["TimeStamp", "Millis", "TTL_looming"], path=loom)
            self.populate_looming_table()
        self.status_var.set("Auto-find complete.")

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
            sel = self.tree.selection()
            if sel: looming_millis = float(self.tree.item(sel[0])['values'][1]); self.status_var.set(f"Using selected event at {looming_millis} ms.")
            else: first_looming_event = self.looming_df[self.looming_df['TTL_looming'] == True].iloc[0]; looming_millis = first_looming_event['Millis']; self.status_var.set("Using first 'True' looming event by default.")
            closest_cam_row = self.cam_df.loc[(self.cam_df['Millis'] - looming_millis).abs().idxmin()]
            self.looming_on_frame, self.looming_on_time = int(closest_cam_row['Cam_Frame']), int(closest_cam_row['Millis'])
            pre_event_minutes = float(self.pre_event_minutes.get()); pre_event_target_millis = looming_millis - (pre_event_minutes * 60 * 1000)
            closest_pre_event_row = self.cam_df.loc[(self.cam_df['Millis'] - pre_event_target_millis).abs().idxmin()]
            self.pre_event_frame = int(closest_pre_event_row['Cam_Frame'])
            self.lbl_sync_result.config(text=f"Looming Frame: {self.looming_on_frame} | Pre-Event Frame: {self.pre_event_frame} | Total: {self.cam_df['Cam_Frame'].max()}")
            self.start_frame.set(str(self.pre_event_frame))
        except Exception as e: messagebox.showerror("Error", str(e))

    def start_correction_process(self):
        if not self.video_path: messagebox.showwarning("Warning", "Please load a video file first."); return
        try: start_frame_num = int(self.start_frame.get()); assert start_frame_num >= 0
        except: messagebox.showerror("Error", "Enter a valid, non-negative start frame number."); return
        if self.correction_params:
            choice = messagebox.askyesnocancel("Params Found", "Reuse saved parameters?")
            if choice: self.process_video(self.correction_params, start_frame_num); return
        self.run_calibration_workflow(start_frame_num)

    def run_calibration_workflow(self, start_frame_num):
        global points; self.status_var.set("Starting calibration..."); self.root.update()
        cap = cv2.VideoCapture(self.video_path); ret, frame = cap.read()
        if not ret: return
        h, w = frame.shape[:2]; points = []
        win = "Select Points | ENTER=Done | C=Clear | Q/ESC=Quit"
        cv2.namedWindow(win); cv2.setMouseCallback(win, click_event, {'frame': frame})
        while True:
            img = frame.copy()
            for i, p in enumerate(points): cv2.circle(img, p, 7, (0,255,0) if i<4 else (0,255,255), -1); cv2.putText(img, str(i+1), (p[0]+15,p[1]-15), 1, 1, (0,255,0), 2)
            cv2.imshow(win, img); key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), 27]: cv2.destroyAllWindows(); cap.release(); return
            if key == ord('c'): points = []
            if key == 13 and len(points) == 8: break
        cv2.destroyAllWindows()
        res = minimize(get_straightness_error, [0.0, w/2, h/2], args=(np.array(points, dtype=np.float32), h, w), method='L-BFGS-B', bounds=[(-0.5, 0.5), (0, w), (0, h)])
        final = self.fine_tune_and_preview(frame, res.x, np.array([points[:4]], dtype=np.float32), self.get_aspect_ratio_from_dialog())
        cv2.destroyAllWindows(); cap.release()
        if final != 'quit': self.correction_params = final; self.process_video(final, start_frame_num)

    def process_video(self, params, start_frame_num):
        # We receive output_dims exactly as calculated in fine_tune_and_preview
        k1_final, cx_final, cy_final, M_final, output_dims, new_cam_matrix_final = params
        
        cap_check = cv2.VideoCapture(self.video_path)
        h, w = cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT), cap_check.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_check.release()
        
        final_camera_matrix = np.array([[w, 0, cx_final], [0, w, cy_final], [0, 0, 1]], dtype=np.float32)
        final_dist_coeffs = np.array([k1_final, 0, 0, 0, 0], dtype=np.float32)
        
        output_path = self.generate_output_path(self.video_path)
        self.log_analysis_results(output_path, params)
        
        self.status_var.set(f"Processing: {os.path.basename(self.video_path)}")
        self.root.update()
        
        batch_cap = cv2.VideoCapture(self.video_path)
        batch_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
        
        # --- USE THE PASSED OUTPUT DIMS ---
        # VideoWriter will use the exact dimensions calculated during fine-tuning.
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), batch_cap.get(cv2.CAP_PROP_FPS), output_dims)
        
        total_frames = int(batch_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        while True:
            ret, f = batch_cap.read()
            if not ret: break
            
            # Apply distortion correction
            lens_corrected = cv2.undistort(f, final_camera_matrix, final_dist_coeffs, None, new_cam_matrix_final)
            
            # Apply perspective warp into the final dimensions
            final_frame = cv2.warpPerspective(lens_corrected, M_final, output_dims)
            
            out.write(final_frame)
            processed_frames += 1
            
            # if processed_frames % 30 == 0: 
            self.status_var.set(f"Processing frame {start_frame_num + processed_frames}/{total_frames}")
            self.root.update()
                
        out.release()
        batch_cap.release()
        
        messagebox.showinfo("Success", f"Video processed successfully!\nSaved to: {output_path}")
        self.status_var.set("Ready.")

    # def process_video(self, params, start_frame_num):
    #     k1, cx, cy, M, dims, ncam = params
    #     cap = cv2.VideoCapture(self.video_path); w = cap.get(3); h = cap.get(4)
    #     cmat = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]], dtype=np.float32); dist = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
    #     out_path = self.generate_output_path(self.video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
    #     writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), dims)
    #     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); processed = 0
    #     while True:
    #         ret, f = cap.read()
    #         if not ret: break
    #         writer.write(cv2.warpPerspective(cv2.undistort(f, cmat, dist, None, ncam), M, dims))
    #         processed += 1
    #         if processed % 30 == 0: self.status_var.set(f"Processing frame {start_frame_num + processed}/{total}"); self.root.update()
    #     writer.release(); cap.release(); self.log_analysis_results(out_path, params); messagebox.showinfo("Success", "Saved to " + out_path)

    # def log_analysis_results(self, out_path, params):
    #     k1, cx, cy, M, dims, ncam = params
    #     log = {"Time_analyzed": time.strftime("%Y-%m-%d %H:%M:%S"), "video": os.path.basename(self.video_path), "cam": os.path.basename(self.cam_file_path), "looming": os.path.basename(self.looming_file_path), "lens_k1": f"{k1:.6f}", "M_matrix_flat": ";".join(map(str, M.flatten())), "output_dims": f"{dims[0]},{dims[1]}", "new_cam_matrix_flat": ";".join(map(str, ncam.flatten()))}
    #     pd.DataFrame([log]).to_csv(self.log_file_path, mode='a', header=not os.path.exists(self.log_file_path), index=False)

    # def load_last_parameters_from_log(self):
    #     if not os.path.exists(self.log_file_path): return
    #     try:
    #         row = pd.read_csv(self.log_file_path).iloc[-1]
    #         M = np.array([float(i) for i in row['M_matrix_flat'].split(';')]).reshape(3, 3)
    #         nc = np.array([float(i) for i in row['new_cam_matrix_flat'].split(';')]).reshape(3, 3)
    #         self.correction_params = (float(row['lens_k1']), float(row['lens_cx']), float(row['lens_cy']), M, tuple(map(int, row['output_dims'].split(','))), nc)
    #         self.lbl_params_status.config(text="Ready (from log)", foreground="green")
    #     except: pass
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
        res = simpledialog.askstring("Aspect Ratio", "Enter ratio (e.g. 1:1)", parent=self.root, initialvalue="1:1")
        try:
            if ':' in res: w, h = map(float, res.split(':')); return w/h
            return float(res)
        except: return 1



    def fine_tune_and_preview(self, frame, initial_params, original_corner_points, aspect_ratio):
        h, w = frame.shape[:2]
        window_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        k1_auto, cx_auto, cy_auto = initial_params
        initial_k1_slider_pos = int((k1_auto * 5000) + 10000)
        initial_cx_slider_pos = int(cx_auto)
        initial_cy_slider_pos = int(cy_auto)

        cv2.createTrackbar("Correction", window_name, initial_k1_slider_pos, 20000, lambda x:None)
        cv2.createTrackbar("Center X", window_name, initial_cx_slider_pos, w, lambda x: None)
        cv2.createTrackbar("Center Y", window_name, initial_cy_slider_pos, h, lambda x: None)
        
        pt_TR = original_corner_points[0][1] 
        pt_BR = original_corner_points[0][2] 
        
        # 1. Height based on vertical difference in Y between TR and BR
        target_h = abs(pt_BR[1] - pt_TR[1])
        target_w = target_h * aspect_ratio
        
        # --- NEW MARGIN LOGIC ---
        y_margin = pt_TR[1]          # Original Top margin
        x_margin = y_margin * 4.0    # Making X-Offset LARGER (4x the Y margin)
        
        # 2. Set output dims with the new independent margins
        out_w = int(target_w + x_margin * 2)
        out_h = int(target_h + y_margin * 2)
        output_dims = (out_w, out_h)
        
        # 3. Lock destination points using the specific X and Y margins
        dst_points = np.array([
            [x_margin, y_margin],                       # Top-Left
            [x_margin + target_w, y_margin],            # Top-Right
            [x_margin + target_w, y_margin + target_h], # Bottom-Right
            [x_margin, y_margin + target_h]             # Bottom-Left
        ], dtype="float32")
        
        last_params = ()
        while True:
            k1_pos = cv2.getTrackbarPos("Correction", window_name)
            cx_pos = cv2.getTrackbarPos("Center X", window_name)
            cy_pos = cv2.getTrackbarPos("Center Y", window_name)
            
            if (k1_pos, cx_pos, cy_pos) != last_params:
                k1_fine = (k1_pos - 10000) / 5000.0
                temp_cam_matrix = np.array([[w, 0, cx_pos], [0, w, cy_pos], [0,0,1]], dtype=np.float32)
                temp_dist_coeffs = np.array([k1_fine, 0, 0, 0, 0], dtype=np.float32)
                
                new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(temp_cam_matrix, temp_dist_coeffs, (w,h), 1, (w,h))
                lens_corrected_frame = cv2.undistort(frame, temp_cam_matrix, temp_dist_coeffs, None, new_cam_matrix)
                corrected_corners = cv2.undistortPoints(original_corner_points, temp_cam_matrix, temp_dist_coeffs, P=new_cam_matrix).reshape(-1, 2)
                
                M = cv2.getPerspectiveTransform(corrected_corners, dst_points)
                preview_frame = cv2.warpPerspective(lens_corrected_frame, M, output_dims)
                
                # --- Drawing the guide with the new offsets ---
                preview_with_guide = preview_frame.copy()
                start_pt = (int(x_margin), int(y_margin))
                end_pt = (int(x_margin + target_w), int(y_margin + target_h))
                draw_dashed_rectangle(preview_with_guide, start_pt, end_pt, (0, 0, 255), thickness=2, dash_length=15)
                
                # Minimum width check for sliders
                min_ui_width = 1000
                if preview_with_guide.shape[1] < min_ui_width:
                    pad = (min_ui_width - preview_with_guide.shape[1]) // 2
                    final_display = cv2.copyMakeBorder(preview_with_guide, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
                else:
                    final_display = preview_with_guide

                cv2.imshow(window_name, final_display)
                last_params = (k1_pos, cx_pos, cy_pos)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27: return 'quit'
            if key == ord('a'): return (k1_fine, cx_pos, cy_pos, M, output_dims, new_cam_matrix)
            if key == ord('r'):
                cv2.setTrackbarPos("Correction", window_name, initial_k1_slider_pos)
                cv2.setTrackbarPos("Center X", window_name, initial_cx_slider_pos)
                cv2.setTrackbarPos("Center Y", window_name, initial_cy_slider_pos)

    # def fine_tune_and_preview(self, frame, initial_params, original_corner_points, aspect_ratio):
    #     h, w = frame.shape[:2]; window_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"; cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #     k1_auto, cx_auto, cy_auto = initial_params
    #     initial_k1_slider_pos, initial_cx_slider_pos, initial_cy_slider_pos = int((k1_auto * 10000) + 4000), int(cx_auto), int(cy_auto)
    #     cv2.createTrackbar("Correction", window_name, initial_k1_slider_pos, 8000, lambda x:None); cv2.createTrackbar("Center X", window_name, initial_cx_slider_pos, w, lambda x: None); cv2.createTrackbar("Center Y", window_name, initial_cy_slider_pos, h, lambda x: None)
    #     last_params = ()
    #     while True:
    #         k1_pos, cx_pos, cy_pos = cv2.getTrackbarPos("Correction", window_name), cv2.getTrackbarPos("Center X", window_name), cv2.getTrackbarPos("Center Y", window_name)
    #         if (k1_pos, cx_pos, cy_pos) != last_params:
    #             k1_fine = (k1_pos - 4000) / 10000.0
    #             temp_cam_matrix = np.array([[w, 0, cx_pos], [0, w, cy_pos], [0,0,1]], dtype=np.float32)
    #             temp_dist_coeffs = np.array([k1_fine, 0, 0, 0, 0], dtype=np.float32)
    #             new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(temp_cam_matrix, temp_dist_coeffs, (w,h), 1, (w,h))
    #             lens_corrected_frame = cv2.undistort(frame, temp_cam_matrix, temp_dist_coeffs, None, new_cam_matrix)
    #             corrected_corners = cv2.undistortPoints(original_corner_points, temp_cam_matrix, temp_dist_coeffs, P=new_cam_matrix).reshape(-1, 2)
                
    #             max_width = h # Set the output width to the original video's height
    #             max_height = int(max_width / aspect_ratio)
                
    #             dst_points = np.array([[0,0], [max_width-1,0], [max_width-1,max_height-1], [0,max_height-1]], dtype="float32")
    #             M = cv2.getPerspectiveTransform(corrected_corners, dst_points)
    #             preview_frame = cv2.warpPerspective(lens_corrected_frame, M, (max_width, max_height))
    #             cv2.imshow(window_name, preview_frame); last_params = (k1_pos, cx_pos, cy_pos)
    #         key = cv2.waitKey(20) & 0xFF
    #         if key == ord('q') or key == 27: return 'quit'
    #         if key == ord('a'): return (k1_fine, cx_pos, cy_pos, M, (max_width, max_height), new_cam_matrix)
    #         if key == ord('r'):
    #             cv2.setTrackbarPos("Correction", window_name, initial_k1_slider_pos); cv2.setTrackbarPos("Center X", window_name, initial_cx_slider_pos); cv2.setTrackbarPos("Center Y", window_name, initial_cy_slider_pos)

    def generate_output_path(self, path):
        out_dir = os.path.join(os.path.dirname(path), "output"); os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{time.strftime('%Y%m%d-%H%M%S')}.mp4")

    # ==========================================
    # === BATCH TRIM LOGIC (Tab 2) =============
    # ==========================================
    def batch_load_videos(self):
        paths = filedialog.askopenfilenames(title="Select Video Files",
                                            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
                                            )
        if not paths: return
        start_idx = len(self.batch_tree.get_children()) + 1
        for i, p in enumerate(paths):
            self.batch_tree.insert("", "end", values=(start_idx + i, p, "", "", "Pending CSVs"))
        self.status_var.set(f"Loaded {len(paths)} videos to batch list.")

    def batch_auto_map(self):
        mapped_count = 0
        for item_id in self.batch_tree.get_children():
            vals = self.batch_tree.item(item_id, 'values')
            no, vid_path = vals[0], vals[1]
            
            if vals[2] != "" and vals[3] != "": continue

            cam_f = self.get_closest_after_file(vid_path, "cam")
            loom_f = self.get_closest_after_file(vid_path, "Looming")
                
            status = "Ready" if cam_f and loom_f else "Missing CSVs"
            self.batch_tree.item(item_id, values=(no, vid_path, cam_f or "", loom_f or "", status))
            if cam_f and loom_f: mapped_count += 1
        self.status_var.set(f"Auto-mapped {mapped_count} video(s).")

    def batch_select_all(self):
        self.batch_tree.selection_set(self.batch_tree.get_children())

    def batch_set_file(self, ftype):
        sel = self.batch_tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "Select a row in the table first."); return
        path = filedialog.askopenfilename(title=f"Select {ftype} CSV",
                                          filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
                                          )
        if path:
            vals = list(self.batch_tree.item(sel[0], 'values'))
            if ftype == "Cam": vals[2] = path
            else: vals[3] = path
            vals[4] = "Ready" if vals[2] and vals[3] else "Missing CSVs"
            self.batch_tree.item(sel[0], values=vals)

    def batch_clear_list(self):
        for i in self.batch_tree.get_children(): self.batch_tree.delete(i)
        self.status_var.set("Batch list cleared.")

    def batch_save_mapping(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        data = []
        for item_id in self.batch_tree.get_children():
            vals = self.batch_tree.item(item_id, 'values')
            data.append({"No": vals[0], "Video": vals[1], "Cam": vals[2], "Looming": vals[3], "Status": vals[4]})
        pd.DataFrame(data).to_csv(path, index=False)
        self.status_var.set("Mapping saved.")

    def batch_load_mapping(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                self.batch_tree.insert("", "end", values=(r.get('No', ''), r.get('Video', ''), r.get('Cam', ''), r.get('Looming', ''), r.get('Status', '')))
            self.status_var.set("Mapping loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mapping: {e}")

    def run_batch_trim(self):
        items = self.batch_tree.get_children()
        if not items: return
        
        try: pre_min = float(self.batch_pre_event.get())
        except: messagebox.showerror("Error", "Invalid Pre-event Time."); return

        total_videos = len(items)
        completed = 0

        for idx, item_id in enumerate(items):
            vals = self.batch_tree.item(item_id, 'values')
            no, vid_path, cam_path, loom_path, status = vals
            
            if status != "Ready": continue

            self.batch_tree.item(item_id, values=(no, vid_path, cam_path, loom_path, "Processing..."))
            self.status_var.set(f"Batch {idx+1}/{total_videos}: Calculating sync for {os.path.basename(vid_path)}...")
            self.root.update()

            # 1. Calculate Start Frame
            try:
                c_df = pd.read_csv(cam_path)
                l_df = pd.read_csv(loom_path)
                first_loom = l_df[l_df['TTL_looming'] == True].iloc[0]
                loom_millis = first_loom['Millis']
                
                target_millis = loom_millis - (pre_min * 60 * 1000)
                closest_pre_row = c_df.loc[(c_df['Millis'] - target_millis).abs().idxmin()]
                start_frame = int(closest_pre_row['Cam_Frame'])
                target_frames = int(c_df['Cam_Frame'].max())
            except:
                self.batch_tree.item(item_id, values=(no, vid_path, cam_path, loom_path, "Error: CSV parse"))
                continue

            # 2. Wait for Cloud Download
            self.status_var.set(f"Batch {idx+1}/{total_videos}: Verifying cloud sync...")
            while True:
                cap = cv2.VideoCapture(vid_path)
                actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if actual_frames >= target_frames:
                    break
                self.status_var.set(f"Batch {idx+1}/{total_videos}: Waiting for cloud {actual_frames}/{target_frames}...")
                self.root.update()
                cap.release()
                time.sleep(2)
            
            # 3. Trim and Save
            w, h = int(cap.get(3)), int(cap.get(4))
            out_path = self.generate_output_path(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))
            
            processed = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                writer.write(f)
                processed += 1
                # if processed % 30 == 0:
                self.status_var.set(f"Batch {idx+1}/{total_videos}: Trimming frame {start_frame + processed}/{actual_frames}")
                self.root.update()
        
            writer.release(); cap.release()
            self.batch_tree.item(item_id, values=(no, vid_path, cam_path, loom_path, "Done"))
            completed += 1
            
        self.status_var.set(f"Batch completed! {completed}/{total_videos} successful.")
        messagebox.showinfo("Batch Complete", f"Successfully processed {completed} of {total_videos} videos.")

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoCorrectorApp(root)
    root.mainloop()
