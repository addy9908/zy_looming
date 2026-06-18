"""
Created on Fri Apr 17 16:57:59 2026
1. plot velocity vs time from raw Ethovision output files (Excel) or dlc files (.h5)
2. calculate and mask the back to shelter time, freezing time
3. save the results, raw plot data, and plots
@author: Zengyou Ye (addy9908@gmail.com)
"""

import sys
import os
import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path
import csv

from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableView, QLabel, QLineEdit, QCheckBox,
                             QFormLayout, QComboBox, QSplitter, QMessageBox, QAbstractItemView, QGroupBox)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QColor, QBrush, QStandardItemModel, QStandardItem

# --- OpenCV Imports ---
try:
    import cv2
    import scipy.optimize as opt
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==========================================
# MATHEMATICAL HELPERS (For 8-Point Calib)
# ==========================================
def to_3d(v):
    return np.hstack((v, np.zeros((v.shape[0], 1)))) if len(v.shape) > 1 else np.append(v, 0)

def distance_to_line(p, a, b):
    p_3d, a_3d, b_3d = to_3d(p), to_3d(a), to_3d(b)
    return np.linalg.norm(np.cross(b_3d - a_3d, a_3d - p_3d)) / (np.linalg.norm(b_3d - a_3d) + 1e-6)

def get_straightness_error(params, src_points, h, w):
    k1, cx, cy = params
    camera_matrix = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(np.array([src_points], dtype=np.float32), camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 8, 2)
    p1, p2, p3, p4, p5, p6, p7, p8 = undistorted_pts[0]
    return sum([distance_to_line(p5, p1, p2), distance_to_line(p6, p2, p3), distance_to_line(p7, p3, p4), distance_to_line(p8, p4, p1)])


# ==========================================
# CUSTOM UI WIDGETS
# ==========================================
class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QStandardItemModel(self))

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked: item.setCheckState(Qt.Unchecked)
        else: item.setCheckState(Qt.Checked)

    def add_items(self, items, default_checked=None):
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            if default_checked and text in default_checked: item.setData(Qt.Checked, Qt.CheckStateRole)
            else: item.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.model().appendRow(item)

    def get_checked_items(self):
        return [self.model().item(i).text() for i in range(self.count()) if self.model().item(i).checkState() == Qt.Checked]


class PandasModel(QAbstractTableModel):
    """Bridges Pandas DataFrames with PyQt TableViews."""
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._df = df

    def rowCount(self, parent=None): return len(self._df.index)
    def columnCount(self, parent=None): return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float): return f"{val:.3f}"
            return str(val)
        elif role == Qt.BackgroundRole:
            # Highlight interpolated data slightly yellow
            col_name = str(self._df.columns[index.column()])
            if col_name.endswith('_filled'):
                orig_col = col_name.replace('_filled', '')
                if orig_col in self._df.columns:
                    val = self._df.iloc[index.row()][orig_col]
                    if pd.isna(val) or val == "-" or val == -1:
                        return QBrush(QColor(255, 255, 200)) 
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole: return self._df.columns[col]
        return None


# ==========================================
# MAIN APPLICATION
# ==========================================
class LoomingAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Looming Analyzer - DeepLabCut & EthoVision Engine")
        self.resize(1600, 1050)
        
        # Application State
        self.files = []
        self.file_data = {} # Dict storing metadata and settings for EACH loaded file
        self.current_file_idx = -1
        self.raw_data = None
        self.selected_idx = None
        self.output_dir = ""
        
        # Plotting Markers
        self.vel_marker = None
        self.trace_marker = None
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT PANEL (Settings & Loaders) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)
        
        # 1. Loaders Group
        grp_loaders = QGroupBox("1. Data Loaders")
        loader_lay = QVBoxLayout()
        
        self.btn_load_dlc = QPushButton("Load DeepLabCut (.h5)")
        self.btn_load_dlc.clicked.connect(lambda: self.load_files("dlc"))
        
        self.btn_load_etho = QPushButton("Load EthoVision (.xlsx / .txt)")
        self.btn_load_etho.clicked.connect(lambda: self.load_files("ethovision"))
        
        self.btn_load_saved = QPushButton("Load Saved Analysis (.csv)")
        self.btn_load_saved.clicked.connect(lambda: self.load_files("saved_analysis"))
        
        loader_lay.addWidget(self.btn_load_dlc)
        loader_lay.addWidget(self.btn_load_etho)
        loader_lay.addWidget(self.btn_load_saved)
        
        id_lay = QHBoxLayout()
        id_lay.addWidget(QLabel("Mouse ID Index (split by '_'):"))
        self.txt_mouse_id_idx = QLineEdit("2")
        self.txt_mouse_id_idx.setFixedWidth(30)
        self.btn_apply_id_idx = QPushButton("Apply to All")
        self.btn_apply_id_idx.clicked.connect(self.apply_mouse_id_index)
        
        id_lay.addWidget(self.txt_mouse_id_idx)
        id_lay.addWidget(self.btn_apply_id_idx)
        loader_lay.addLayout(id_lay)
        
        grp_loaders.setLayout(loader_lay)
        left_layout.addWidget(grp_loaders)
        
        # 2. Chamber & Looming Parameters
        grp_params = QGroupBox("2. Chamber & Looming Parameters")
        param_form = QFormLayout()
        
        self.txt_chamber_w = QLineEdit("30.5")
        self.txt_chamber_h = QLineEdit("51.0")
        self.txt_shelter_h = QLineEdit("10.0") # Shelter is always bottom-anchored, width=chamber_w
        self.txt_x_offset = QLineEdit("0.0") 
        self.txt_calib_y_offset = QLineEdit("10.0")
        
        self.txt_stim_start = QLineEdit("60.0")
        self.txt_pre_stim = QLineEdit("-5.0")
        self.txt_rolling_avg = QLineEdit("6")
        self.txt_freeze_thresh = QLineEdit("12")
        
        param_form.addRow("Chamber W (cm):", self.txt_chamber_w)
        param_form.addRow("Chamber H (cm):", self.txt_chamber_h)
        param_form.addRow("Shelter Height (cm):", self.txt_shelter_h)
        param_form.addRow("Manual X-Offset (cm):", self.txt_x_offset) 
        param_form.addRow("Calib Y-Offset (cm):", self.txt_calib_y_offset)
        param_form.addRow("Looming Start (s):", self.txt_stim_start)
        param_form.addRow("Plot Pre-Range (s):", self.txt_pre_stim)
        param_form.addRow("Velocity Smoothing:", self.txt_rolling_avg)
        param_form.addRow("Freeze Thresh (cm/s):", self.txt_freeze_thresh)
        
        for txt in [self.txt_chamber_w, self.txt_chamber_h, self.txt_shelter_h, 
                    self.txt_x_offset, self.txt_calib_y_offset, self.txt_freeze_thresh,
                    self.txt_stim_start, self.txt_pre_stim, self.txt_rolling_avg]:
            txt.editingFinished.connect(self.recalculate_data)
            
        grp_params.setLayout(param_form)
        left_layout.addWidget(grp_params)

        # 3. DLC specific Settings
        self.grp_dlc = QGroupBox("3. DeepLabCut Settings")
        dlc_form = QFormLayout()
        
        self.txt_fps = QLineEdit("30.0")
        self.txt_px_cm = QLineEdit("10.0")
        self.txt_vid_w = QLineEdit("1920")
        self.txt_vid_h = QLineEdit("1080")
        self.txt_likelihood = QLineEdit("0.3")
        self.minimal_movement = QLineEdit("5")
        
        self.combo_bp = CheckableComboBox()
        parts = ["all_in_list", "mouse_center", "nose",  "neck", "mid_back", "tail_base",  
                 "left_shoulder", "right_shouder", "left_midside", "right_midside", 
                 "left_ear", "right_ear", "left_hip", "right_hip", "head_midpoint"]
        
        self.combo_bp.add_items(parts, default_checked=["mouse_center"])
        self.combo_bp.model().dataChanged.connect(self.recalculate_data)

        dlc_form.addRow("Video FPS:", self.txt_fps)
        dlc_form.addRow("Pixel per CM:", self.txt_px_cm)
        dlc_form.addRow("Video Width (px):", self.txt_vid_w)
        dlc_form.addRow("Video Height (px):", self.txt_vid_h)
        dlc_form.addRow("Likelihood cutoff:", self.txt_likelihood)
        dlc_form.addRow("Minimal movement (px):", self.minimal_movement)
        dlc_form.addRow("Body Parts:", self.combo_bp)
        
        for txt in [self.txt_fps, self.txt_px_cm, self.txt_vid_w, self.txt_vid_h, self.txt_likelihood, self.minimal_movement]:
            txt.editingFinished.connect(self.recalculate_data)
            
        # Calibration buttons
        calib_lay1 = QHBoxLayout()
        self.btn_draw_scale = QPushButton("📏 Draw Line for Scale") # <--- NEW BUTTON
        self.btn_draw_scale.clicked.connect(self.draw_line_for_scale)
        self.btn_calib = QPushButton("🔲 8-Pt Lens Calib")
        self.btn_calib.clicked.connect(self.calibrate_video)
        calib_lay1.addWidget(self.btn_draw_scale)
        calib_lay1.addWidget(self.btn_calib)
        
        calib_lay2 = QHBoxLayout()
        self.btn_save_calib = QPushButton("💾 Save JSON")
        self.btn_save_calib.clicked.connect(self.save_calibration_json)
        self.btn_load_calib = QPushButton("📂 Load JSON")
        self.btn_load_calib.clicked.connect(self.load_calibration_json)
        calib_lay2.addWidget(self.btn_save_calib)
        calib_lay2.addWidget(self.btn_load_calib)
        
        dlc_lay = QVBoxLayout()
        dlc_lay.addLayout(dlc_form)
        dlc_lay.addLayout(calib_lay1)
        dlc_lay.addLayout(calib_lay2)
        self.grp_dlc.setLayout(dlc_lay)
        left_layout.addWidget(self.grp_dlc)

        # 4. Exports
        grp_export = QGroupBox("4. Export Tools")
        export_lay = QVBoxLayout()
        self.btn_master_export = QPushButton("🚀 Master Export (Current File)")
        self.btn_master_export.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 6px;")
        self.btn_master_export.clicked.connect(self.master_export_current)
        export_lay.addWidget(self.btn_master_export)
        grp_export.setLayout(export_lay)
        left_layout.addWidget(grp_export)
        
        left_layout.addStretch()

        # --- RIGHT PANEL (Navigator, Plots, Table) ---
        right_splitter = QSplitter(Qt.Vertical)
        top_right_widget = QWidget()
        top_right_layout = QVBoxLayout(top_right_widget)
        
        # Top Bar: File Navigator & Meta Data (Split into 2 rows)
        nav_panel = QWidget()
        nav_vbox = QVBoxLayout(nav_panel)
        nav_vbox.setContentsMargins(0, 0, 0, 0)
        # Row 1: File selection
        nav_row1 = QHBoxLayout()
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.prev_file)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_file)
        
        nav_row1.addWidget(QLabel("<b>Selected File:</b>"))
        nav_row1.addWidget(self.file_combo, 1) # Expanding combobox
        nav_row1.addWidget(self.btn_prev)
        nav_row1.addWidget(self.btn_next)
        
        # Row 2: Metadata
        nav_row2 = QHBoxLayout()
        self.txt_mouse_id = QLineEdit()
        self.txt_photostim = QLineEdit()
        self.txt_vName = QLineEdit()
        self.txt_post_stim = QLineEdit("10.0")
        
        nav_row2.addWidget(QLabel("Mouse ID:"))
        nav_row2.addWidget(self.txt_mouse_id)
        nav_row2.addWidget(QLabel("PhotoStim:"))
        nav_row2.addWidget(self.txt_photostim)
        nav_row2.addWidget(QLabel("Video name:"))
        nav_row2.addWidget(self.txt_vName)
        nav_row2.addWidget(QLabel("Plot Post-Range (s):"))
        nav_row2.addWidget(self.txt_post_stim)
        
        self.txt_mouse_id.editingFinished.connect(self.on_metadata_changed)
        self.txt_photostim.editingFinished.connect(self.on_metadata_changed)
        self.txt_post_stim.editingFinished.connect(self.on_post_stim_changed)
        
        nav_vbox.addLayout(nav_row1)
        nav_vbox.addLayout(nav_row2)
        top_right_layout.addWidget(nav_panel)
        
        # Annotation & Plots Box
        ann_plot_lay = QHBoxLayout()
        
        # Trace Plot (Top left of right side)
        trace_widget = QWidget()
        trace_lay = QVBoxLayout(trace_widget)
        trace_lay.addWidget(QLabel("<b>Behavior Trace:</b>"))
        self.trace_fig = Figure(figsize=(4, 4))
        self.trace_ax = self.trace_fig.add_subplot(111)
        self.trace_fig.tight_layout()
        self.trace_canvas = FigureCanvas(self.trace_fig)
        self.trace_canvas.setFocusPolicy(Qt.ClickFocus)
        self.trace_canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.trace_canvas.mpl_connect('key_press_event', self.on_key_press)
        trace_lay.addWidget(self.trace_canvas)
        
        tbtn_lay = QHBoxLayout()
        btn_copy_t = QPushButton("Copy Image")
        btn_copy_t.clicked.connect(lambda: self.copy_plot_to_clipboard(self.trace_canvas))
        tbtn_lay.addWidget(btn_copy_t)
        trace_lay.addLayout(tbtn_lay)
        ann_plot_lay.addWidget(trace_widget, stretch=1)
        
        # Annotation Controls (Middle)
        ann_col = QVBoxLayout()
        ann_col.addWidget(QLabel("<b>Interactive Timestamps:</b>"))
        ann_col.addWidget(QLabel("<i>Click anywhere on a plot, then assign.</i>"))
        
        self.lbl_selected_point = QLabel("Selected Time: -- s")
        self.lbl_selected_point.setStyleSheet("font-weight: bold; color: #1565c0; font-size: 16px; margin: 10px 0px;")
        ann_col.addWidget(self.lbl_selected_point)
        
        # --- NEW: Master Auto-Detect Button ---
        self.btn_auto_detect = QPushButton("🤖 Auto-Detect (Freeze & Shelter)")
        self.btn_auto_detect.setStyleSheet("background-color: #e8f5e9; color: #2e7d32; font-weight: bold; padding: 6px;")
        self.btn_auto_detect.clicked.connect(self.auto_detect_behaviors)
        ann_col.addWidget(self.btn_auto_detect)
        
        # Annotator Buttons
        def make_ann_row(btn_text, key):
            row = QHBoxLayout()
            btn = QPushButton(btn_text)
            lbl = QLabel("--")
            btn.clicked.connect(lambda: self.set_annotation(key))
            row.addWidget(btn)
            row.addWidget(lbl)
            return row, lbl
            
        r1, self.lbl_freeze_start = make_ann_row("1. Set Freezing Start", "freeze_start")
        r2, self.lbl_freeze_end = make_ann_row("2. Set Freezing End", "freeze_end")
        r3, self.lbl_shelter = make_ann_row("3. Set Back to Shelter", "shelter_time")
        
        ann_col.addLayout(r1); ann_col.addLayout(r2); ann_col.addLayout(r3)
        ann_col.addStretch()
        
        # Metrics readout
        self.lbl_metrics = QLabel("Duration: --\nDistance: --")
        self.lbl_metrics.setStyleSheet("font-weight: bold; font-size: 14px; color: darkred;")
        ann_col.addWidget(self.lbl_metrics)
        ann_plot_lay.addLayout(ann_col, stretch=1)
        
        top_right_layout.addLayout(ann_plot_lay)
        right_splitter.addWidget(top_right_widget)
        
        # Table
        mid_right_widget = QWidget()
        mid_right_layout = QVBoxLayout(mid_right_widget)
        
        self.chk_clean_view = QCheckBox("Clean Table View (Hide raw coords & metadata)")
        self.chk_clean_view.setChecked(True) # On by default!
        self.chk_clean_view.stateChanged.connect(self.update_table_view)
        mid_right_layout.addWidget(self.chk_clean_view)
        
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.clicked.connect(self.on_table_clicked)
        mid_right_layout.addWidget(self.table_view)
        
        right_splitter.addWidget(mid_right_widget)
        
        # Bottom Velocity Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        v_btns = QHBoxLayout()
        v_btns.addWidget(self.toolbar)
        btn_copy_v = QPushButton("Copy Velocity Plot")
        btn_copy_v.clicked.connect(lambda: self.copy_plot_to_clipboard(self.canvas))
        v_btns.addWidget(btn_copy_v)
        
        plot_layout.addLayout(v_btns)
        plot_layout.addWidget(self.canvas)
        right_splitter.addWidget(plot_widget)
        
        right_splitter.setSizes([400, 250, 400])
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter)

    # ==========================================
    # FILE LOADING & PARSING
    # ==========================================
    def load_files(self, file_type):
        """Unified loader that tags files with their specific type."""
        exts = "*.h5" if file_type == "dlc" else ("*.csv" if file_type == "saved_analysis" else "*.xlsx *.csv *.txt")
        files, _ = QFileDialog.getOpenFileNames(self, f"Open {file_type.upper()} Files", "", f"Data ({exts})")
        
        if files:
            files = [f for f in files if "EthoVision_Batch_Results" not in os.path.basename(f)]
            if not files: return
            
            files.sort()
            self.files = files
            self.output_dir = os.path.join(os.path.dirname(self.files[0]), "Output")
            if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

            self.file_data = {}
            for f in self.files:
                # Setup default empty metadata                
                self.file_data[f] = {
                    'type': file_type,
                    'mouse_id': os.path.basename(f).split('_')[int(self.txt_mouse_id_idx.text())],
                    'video_name': None,
                    'photostim': 'OFF',
                    'freeze_start': None,
                    'freeze_end': None,
                    'shelter_time': None,
                    'post_stim': 10.0,
                    'stim_start': float(self.txt_stim_start.text()),
                    'correction_params': None # Holds calibration matrix if DLC
                }
                
                # try to update the video name
                if file_type == 'dlc' and '_superanimal' in f:
                    self.file_data[f]['video_name'] = os.path.basename(f).split('_superanimal')[0]
                elif file_type == "saved_analysis":
                    self.file_data[f]['mouse_id'] = os.path.basename(f).split('_')[0]
                    self.file_data[f]['photostim'] = os.path.basename(f).split('_')[1]
                    self.file_data[f]['video_name'] = os.path.basename(f).split('_')[2]

            # Update UI
            self.file_combo.blockSignals(True)
            self.file_combo.clear()
            self.file_combo.addItems([os.path.basename(f) for f in self.files])
            self.file_combo.blockSignals(False)
            
            self.current_file_idx = 0
            self.file_combo.setCurrentIndex(0)
            self.parse_file(self.files[0])

    def parse_file(self, filepath):
        """Routes the file to the correct parser based on its type tag."""
        try:
            self.selected_idx = None
            self.lbl_selected_point.setText("Selected Time: -- s")
            
            d = self.file_data[filepath]
            # self.txt_mouse_id.setText(d['mouse_id'])
            # self.txt_photostim.setText(d['photostim'])
            # self.txt_post_stim.setText(str(d['post_stim']))
            self.txt_stim_start.setText(str(d.get('stim_start', 60.0)))
            
            # Route to parser
            if d['type'] == 'dlc':
                self.grp_dlc.setEnabled(True)
                self.btn_calib.setEnabled(True)
                df = self.parse_dlc(filepath, d)
            elif d['type'] == 'ethovision':
                self.grp_dlc.setEnabled(False)
                # self.btn_calib.setEnabled(False)
                df = self.parse_ethovision(filepath, d)
            elif d['type'] == 'saved_analysis':
                self.grp_dlc.setEnabled(False)
                # self.btn_calib.setEnabled(False)
                df = self. parse_saved_analysis(filepath,d)
            else:
                self.statusBar().showMessage("check the file type", 3000)
                
             # --- NEW: Decode Annotations from Event String ---
                if 'Event' in df.columns:
                    # SECURE FIX: Force NaNs to '0000' and explicitly cast to pure string
                    df['Event'] = df['Event'].fillna('0000').astype(str)
                    events = df['Event'].str.zfill(4) 
                    
                    stim_mask = events.str[0] == '1'
                    fs_mask = events.str[1] == '1'
                    fe_mask = events.str[2] == '1'
                    st_mask = events.str[3] == '1'
                    
                    if stim_mask.any():
                        d['stim_start'] = float(df.loc[stim_mask, 'Trial time'].iloc[0])
                        self.txt_stim_start.setText(str(d['stim_start']))
                        
                    d['freeze_start'] = float(df.loc[fs_mask, 'Rel_Time'].iloc[0]) if fs_mask.any() else None
                    d['freeze_end'] = float(df.loc[fe_mask, 'Rel_Time'].iloc[0]) if fe_mask.any() else None
                    d['shelter_time'] = float(df.loc[st_mask, 'Rel_Time'].iloc[0]) if st_mask.any() else None
                          
            self.raw_data = df
            self.txt_mouse_id.setText(d['mouse_id'])
            self.txt_photostim.setText(d['photostim'])
            self.txt_vName.setText(d['video_name'])
            self.txt_vName.setCursorPosition(0)
            self.txt_post_stim.setText(str(d['post_stim']))

            self.recalculate_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse file: {e}")

    # ==========================================
    # UPDATED: PARSE DEEPLABCUT (.h5)
    # ==========================================
    def parse_saved_analysis(self, filepath, meta):
        """Loads an exported analysis CSV, using row length to safely detect the data table."""
        parsed_meta = {}
        skip_count = 0
        
        # 1. Read the top metadata rows safely
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue
                
                # --- BRILLIANT CHECK: If a row has more than 2 elements, it MUST be the data table! ---
                if len(row) > 2:
                    skip_count = i
                    break
                
                # CASE 2: We hit our designated separator!
                if str(row[0]).strip() == "---":
                    skip_count = i + 1
                    break
                    
                # Otherwise, it's a standard metadata row (length <= 2)
                if len(row) == 2:
                    k, v = row[0].strip(), row[1].strip()
                    parsed_meta[k] = v

        # Helper function to safely parse floats
        def to_float(val):
            try: return float(val) if val else None
            except: return None

        # 2. Restore the Metadata
        meta['video_name'] = parsed_meta.get('Video File', None)
        meta['stim_start'] = to_float(parsed_meta.get('Stim Start')) or 60.0
        meta['freeze_start'] = to_float(parsed_meta.get('Freeze Start'))
        meta['freeze_end'] = to_float(parsed_meta.get('Freeze End'))
        meta['shelter_time'] = to_float(parsed_meta.get('Shelter Time'))
        self.txt_chamber_w.setText(parsed_meta.get('Chamber_w', '30.5'))
        self.txt_chamber_h.setText(parsed_meta.get('Chamber_h', '51'))
        self.txt_shelter_h.setText(parsed_meta.get('Shelter_h','10'))
        self.txt_freeze_thresh.setText(parsed_meta.get('Freeze_threshold','12'))

        # 3. Read the actual DataFrame
        df = pd.read_csv(filepath, skiprows=skip_count)
        
        # Make sure the Event column is strictly treated as a zero-padded 4-character string
        if 'Event' in df.columns:
            df['Event'] = df['Event'].astype(str).str.zfill(4)
            
        return df

    def parse_dlc(self, filepath, meta):
        """
        Parses DLC H5 tracking files (only has frame number =len(df_h5) but not timestamp.
        Cleans low-likelihood points, converts pixels to physical CM, 
        and flips the Y-axis so (0,0) is firmly at the bottom-left of the chamber.
        """
        self.txt_px_cm.setEnabled(True)
        self.txt_vid_w.setEnabled(True)
        self.txt_vid_h.setEnabled(True)
        
        df_h5 = pd.read_hdf(filepath)
        df_h5.replace([-1, -1.0], np.nan, inplace=True)
        
        # 1. Clean low-likelihood points
        try: thresh = float(self.txt_likelihood.text())
        except: thresh = 0.3
        
        x_parts, y_parts = [], []
        selected_parts = self.combo_bp.get_checked_items()
        
        if "all_in_list" in selected_parts:
            selected_parts = [self.combo_bp.itemText(i) for i in range(self.combo_bp.count())]
        
        h5_parts = df_h5.columns.get_level_values('bodyparts').unique()
        bps = [p for p in selected_parts if p in h5_parts]
            
        if not bps: bps = list(h5_parts)

        for p in bps:
            idx_x = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'x')
            idx_y = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'y')
            idx_l = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'likelihood')
            
            x = df_h5.loc[:, idx_x].iloc[:, 0].copy()
            y = df_h5.loc[:, idx_y].iloc[:, 0].copy()
            l = df_h5.loc[:, idx_l].iloc[:, 0].copy()
            
            # Mask out coordinates with low tracking confidence
            mask = l < thresh
            x[mask] = np.nan; y[mask] = np.nan
            x_parts.append(x); y_parts.append(y)

        # Average the selected bodyparts (or single bodypart if only one is checked)
        x_px_raw = pd.concat(x_parts, axis=1).mean(axis=1)
        y_px_raw = pd.concat(y_parts, axis=1).mean(axis=1)
        
        # Create final tracking DataFrame
        fps = float(self.txt_fps.text())
        df = pd.DataFrame()
        df['Frame'] = np.arange(len(df_h5))
        df['Trial time'] = df['Frame'] / fps
        
        # Store original raw pixels (with NaNs)
        df['X_raw'] = x_px_raw
        df['Y_raw'] = y_px_raw
        
        # Handle Missing Values: Forward-fill tracking drops
        x_px_filled = x_px_raw.ffill().bfill()
        y_px_filled = y_px_raw.ffill().bfill()
        
        # ---------------------------------------------------------
        # --- NEW: MINIMAL MOVEMENT FILTER (Ethovision trick 1) ---
        # ---------------------------------------------------------
        # You can tie this to a GUI text box later, e.g., float(self.txt_threshold.text())
        try: movement_threshold = float(self.minimal_movement.text())
        except: movement_threshold = 5.0 
        
        x_vals = x_px_filled.values
        y_vals = y_px_filled.values
        
        if len(x_vals) > 0:
            last_x, last_y = x_vals[0], y_vals[0]
            for i in range(1, len(x_vals)):
                # Calculate pixel distance from the last accepted coordinate
                dist = np.sqrt((x_vals[i] - last_x)**2 + (y_vals[i] - last_y)**2)
                
                if dist < movement_threshold:
                    # Ignore movement: lock coordinate to the last accepted position
                    x_vals[i] = last_x
                    y_vals[i] = last_y
                else:
                    # Valid movement: update the last accepted position
                    last_x = x_vals[i]
                    last_y = y_vals[i]
            
            # Put the filtered values back into the pandas Series
            x_px_filled = pd.Series(x_vals, index=x_px_filled.index)
            y_px_filled = pd.Series(y_vals, index=y_px_filled.index)
        
        # --- NEW: SPATIAL SMOOTHING (The EthoVision trick) ---
        # Smooth the path over a 3-5 frame window to remove "vibration" jitter
        try: smooth_win = int(self.txt_rolling_avg.text())
        except: smooth_win = 6
        
        if smooth_win > 1:
            x_px_filled = x_px_filled.rolling(window=smooth_win, center=True, min_periods=1).mean()
            y_px_filled = y_px_filled.rolling(window=smooth_win, center=True, min_periods=1).mean()
        
        # 2. Coordinate Scaling & Camera Mapping
        cp = meta.get('correction_params')
        if cp is not None:
            # --- 3D CALIBRATED MATRIX MAP ---
            w, _ = cp['orig_w'], cp['orig_h']
            camera_matrix = np.array([[w, 0, cp['cx']], [0, w, cp['cy']], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([cp['k1'], 0, 0, 0, 0], dtype=np.float32)
            
            # Apply Lens Distortion Correction (Requires Top-Left origin!)
            #Pack and reshape to (N, 1, 2) so OpenCV doesn't crash
            pts = np.vstack((x_px_filled, y_px_filled)).T.reshape(-1, 1, 2).astype(np.float32)
            undistorted_pts = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=cp['new_cam_matrix'])
            transformed_pts = cv2.perspectiveTransform(undistorted_pts, cp['M']).reshape(-1, 2)
            
            target_w_px = cp['chamber_dims'][0]
            target_h_px = cp['chamber_dims'][1]
            px_cm = target_w_px / float(self.txt_chamber_w.text())
            
            self.txt_px_cm.setText(f"{px_cm:.2f}")
            self.txt_px_cm.setEnabled(False)
            
            self.txt_vid_w.setText(f"{cp['frame_dims'][0]}")
            self.txt_vid_w.setEnabled(False)
            
            self.txt_vid_h.setText(f"{cp['frame_dims'][1]}")
            self.txt_vid_h.setEnabled(False)
            
            
            x_margin = (w - target_w_px) / 2.0
            y_margin = (cp['frame_dims'][1] - target_h_px) / 2.0
            
            df['X_cm'] = (transformed_pts[:, 0] - x_margin) / px_cm
            df['Y_cm'] = (target_h_px - (transformed_pts[:, 1] - y_margin)) / px_cm # Flip Y
            
            # --- NEW: APPLY CALIBRATION Y-OFFSET ---
            try: y_offset = float(self.txt_calib_y_offset.text())
            except: y_offset = 10.0
            try: x_offset = float(self.txt_x_offset.text())
            except: x_offset = 0
            
            df['Y_cm'] = df['Y_cm'] + y_offset 
            df['X_cm'] = df['X_cm'] + x_offset 
            
        else:
            # --- SIMPLE PIXEL-TO-CM FALLBACK MAP ---
            px_cm = float(self.txt_px_cm.text())
            video_h = float(self.txt_vid_h.text())
            
            df['X_cm'] = x_px_filled / px_cm
            df['Y_cm'] = (video_h - y_px_filled) / px_cm # Flip Y (Top-left 0 becomes Bottom-left 0)
            
        # 3. Calculate Speed & Distance Metrics
        dist = np.sqrt(df['X_cm'].diff()**2 + df['Y_cm'].diff()**2).fillna(0.0)
        df['Distance_cm'] = dist
        df['Raw_Velocity_cm_s'] = dist * fps
        
        # 4. Zone Readout
        try: shelter_h = float(self.txt_shelter_h.text())
        except: shelter_h = 10.0
        df['In zone'] = (df['Y_cm'] <= shelter_h).astype(int)
        
        return df


    # ==========================================
    # UPDATED: PARSE ETHOVISION (.xlsx/.txt)
    # ==========================================    
    def parse_ethovision(self, filepath, meta):
        """Parses Ethovision files and auto-normalizes them to a (0,0) bottom-left origin."""
        self.txt_fps.setEnabled(True)
        if filepath.endswith('.xlsx'):
            temp_df = pd.read_excel(filepath, nrows=50, header=None)
            idx_row = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
            meta_df = temp_df.iloc[:idx_row]
            m_id, p_stim, v_name = "N/A", "N/A","Unknown"
            for _, row in meta_df.iterrows():
                if pd.notna(row[0]) and pd.notna(row[1]):
                    k, v = str(row[0]).lower(), str(row[1])
                    if "mouse" in k: m_id = v
                    elif "stim" in k: p_stim = v
                    # elif k == "video file": v_name = os.path.splitext(os.path.basename(v))[0]
                    elif k == "video file": v_name = Path(v).stem
            self.file_data[filepath]['mouse_id'] = str(m_id).strip()
            self.file_data[filepath]['photostim'] = str(p_stim).strip()
            self.file_data[filepath]['video_name'] = str(v_name).strip()
            df = pd.read_excel(filepath, skiprows=idx_row)
        else:
            df = pd.read_csv(filepath)
            
        df.columns = df.columns.str.strip()
        df.replace("-", np.nan, inplace=True)
        
        xc = next((c for c in df.columns if 'x center' in c.lower()), None)
        yc = next((c for c in df.columns if 'y center' in c.lower()), None)
        
        if xc and yc:
            df['X_raw'] = pd.to_numeric(df[xc], errors='coerce')
            df['Y_raw'] = pd.to_numeric(df[yc], errors='coerce')
            
            # Forward-fill any missing/lost tracking coordinates
            x_filled = df['X_raw'].ffill().bfill()
            y_filled = df['Y_raw'].ffill().bfill()
            
            try: chamber_w = float(self.txt_chamber_w.text())
            except: chamber_w = 30.5
            try: chamber_h = float(self.txt_chamber_h.text())
            except: chamber_h = 51.0
            
            # --- TRUE MATHEMATICAL NORMALIZATION ---
            try: x_off = float(self.txt_x_offset.text())
            except: x_off = 0.0
            try: self.txt_calib_y_offset.setText('0'); y_off = float(self.txt_calib_y_offset.text())
            except: y_off = 0.0
            
            # Assuming EthoVision origin (0,0) is roughly the center of the arena
            df['X_cm'] = x_filled + (chamber_w / 2.0) + x_off
            df['Y_cm'] = y_filled + (chamber_h / 2.0) + y_off
            
            # Shelter Logic (Now completely synchronized with DLC!)
            try: shelter_h = float(self.txt_shelter_h.text())
            except: shelter_h = 10.0
            df['In zone'] = (df['Y_cm'] <= shelter_h).astype(int)
            
            # 2. FIX: Calculate uniform Distance and Velocity from pure coordinates!
            dist = np.sqrt(df['X_cm'].diff()**2 + df['Y_cm'].diff()**2).fillna(0.0)
            df['Distance_cm'] = dist
            
            # Dynamically calculate FPS from the Trial time timestamps
            try: 
                fps = 1.0 / df['Trial time'].diff().median()
                self.txt_fps.setText(f"{fps:.2f}")
                self.txt_fps.setEnabled(False)
                
            except: 
                try: fps = float(self.txt_fps.text())
                except: fps = 30.0
                
            df['Raw_Velocity_cm_s'] = dist * fps
            
        return df


    def apply_mouse_id_index(self):
        """Mass-updates the Mouse_ID for all files based on the requested filename index."""
        if not self.files: 
            return
            
        try: 
            idx = int(self.txt_mouse_id_idx.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Mouse ID Index must be an integer.")
            return
            
        for f in self.files:
            base_name = os.path.basename(f)
            # Remove extensions so they don't get included in the split
            clean_name = base_name.replace('.csv', '').replace('.xlsx', '').replace('.txt', '').replace('.h5', '')
            parts = clean_name.split('_')
            
            # Apply the index if it exists, otherwise fall back to the whole name
            if 0 <= idx < len(parts):
                self.file_data[f]['mouse_id'] = parts[idx]
            else:
                self.file_data[f]['mouse_id'] = clean_name 
                
        # Update the currently viewed file UI
        curr_path = self.get_current_filepath()
        if curr_path:
            self.txt_mouse_id.setText(self.file_data[curr_path]['mouse_id'])
            self.update_plots() # Refreshes the plot titles with new names
            
        self.statusBar().showMessage(f"Applied Mouse ID from index {idx} to {len(self.files)} files.", 3000)

    # ==========================================
    # UPDATED: DATA RECALCULATION ENGINE
    # ==========================================
    def recalculate_data(self):
        """Triggers tracking recalculations whenever a setting or scale parameter is modified."""
        if self.raw_data is None: return
        
        start = float(self.txt_stim_start.text())
        path = self.get_current_filepath()
        if not path: return
        
        d = self.file_data[path]
        d['stim_start'] = start
        
        # --- THE FIX: If the scale/fps inputs changed on a DLC file, re-trigger the parsing math! ---
        if d['type'] == 'dlc':
            dlc_parameters = [self.txt_chamber_w, self.txt_chamber_h,self.txt_px_cm, self.txt_fps, self.txt_likelihood, self.txt_vid_h, self.txt_vid_w, self.minimal_movement, self.combo_bp.model(), self.txt_x_offset, self.txt_calib_y_offset, self.txt_shelter_h]
            if self.sender() in dlc_parameters:
                self.raw_data = self.parse_dlc(path, d)
        elif d['type'] == 'ethovision':
            etho_parameters = [self.txt_chamber_w, self.txt_chamber_h, 
                               self.txt_x_offset, self.txt_calib_y_offset, 
                               self.txt_shelter_h]
            if self.sender() in etho_parameters:
                # IN-MEMORY UPDATE: Lightning fast, no hard drive reading required!
                try: cw = float(self.txt_chamber_w.text())
                except: cw = 30.5
                try: ch = float(self.txt_chamber_h.text())
                except: ch = 51.0
                try: x_off = float(self.txt_x_offset.text())
                except: x_off = 0.0
                try: y_off = float(self.txt_calib_y_offset.text())
                except: y_off = 0.0
                try: sh = float(self.txt_shelter_h.text())
                except: sh = 10.0

                x_filled = self.raw_data['X_raw'].ffill().bfill()
                y_filled = self.raw_data['Y_raw'].ffill().bfill()

                self.raw_data['X_cm'] = x_filled + (cw / 2.0) + x_off
                self.raw_data['Y_cm'] = y_filled + (ch / 2.0) + y_off
                self.raw_data['In zone'] = (self.raw_data['Y_cm'] <= sh).astype(int)                
        
        try:
            window = int(self.txt_rolling_avg.text()) if self.txt_rolling_avg.text().isdigit() else 1
            
            self.raw_data['Trial time'] = pd.to_numeric(self.raw_data['Trial time'], errors='coerce')
            self.raw_data['Rel_Time'] = self.raw_data['Trial time'] - start
            
            if 'Raw_Velocity_cm_s' in self.raw_data.columns:
                if window > 1:
                    self.raw_data['Velocity_cm_s'] = self.raw_data['Raw_Velocity_cm_s'].rolling(window=window, center=True, min_periods=1).mean()
                else:
                    self.raw_data['Velocity_cm_s'] = self.raw_data['Raw_Velocity_cm_s']
                    
            # Strip helper raw column out of final visual table layout
            self.update_plots()
            self.update_table_view() 
        except Exception as e:
            print("Recalculate Error:", e)


    # ==========================================
    # INTERACTION & PLOTTING
    # ==========================================
    def on_plot_click(self, event):
        """Unified click handler: Finds the closest time point whether clicking Trace or Velocity."""
        if self.raw_data is None or not event.xdata: return
        
        if event.inaxes == self.ax: # Clicked Velocity Plot (X is Time)
            self.canvas.setFocus() # <--- Force keyboard focus here
            idx = (self.raw_data['Rel_Time'] - event.xdata).abs().idxmin()
            self.set_selected_point(idx)
            
        elif event.inaxes == self.trace_ax: # Clicked Trace Plot (X/Y are spatial coords)
            self.trace_canvas.setFocus() # <--- Force keyboard focus here
            df = self.raw_data
            if 'X_cm' in df.columns:
                # Find the closest physical CM coordinate
                idx = ((df['X_cm'] - event.xdata)**2 + (df['Y_cm'] - event.ydata)**2).idxmin()
                self.set_selected_point(idx)

    def on_key_press(self, event):
        """Handles Left/Right arrow keys to nudge the selected point frame-by-frame."""
        if event.key == 'left':
            self.move_selected_point(-1)
        elif event.key == 'right':
            self.move_selected_point(1)

    def move_selected_point(self, step):
        """Moves the current selected point backward or forward by 'step' frames."""
        if self.raw_data is None or self.selected_idx is None: 
            return
            
        try:
            # Find the integer row number of the currently selected index
            current_pos = self.raw_data.index.get_loc(self.selected_idx)
            new_pos = current_pos + step
            
            # Ensure we don't go out of bounds
            if 0 <= new_pos < len(self.raw_data):
                new_idx = self.raw_data.index[new_pos]
                self.set_selected_point(new_idx)
        except KeyError:
            pass

    def set_selected_point(self, idx):
        if self.raw_data is None or idx not in self.raw_data.index: return
        self.selected_idx = idx
        self.table_view.selectRow(idx)
        t = self.raw_data.at[idx, 'Rel_Time']
        self.lbl_selected_point.setText(f"Selected Time: {t:.2f} s")
        self._update_plot_markers()

    def _update_plot_markers(self):
        idx = self.selected_idx
        if idx is None or self.raw_data is None: return
        
        # Update Velocity Marker (with safe removal!)
        if self.vel_marker: 
            try: self.vel_marker.remove()
            except: pass
            
        rel_t = self.raw_data.at[idx, 'Rel_Time']
        y_vel = self.raw_data.at[idx, 'Velocity_cm_s'] if 'Velocity_cm_s' in self.raw_data.columns else 0
        self.vel_marker, = self.ax.plot(rel_t, y_vel, marker='o', color='black', markersize=8, zorder=10)
        self.canvas.draw_idle()
        
        # Update Trace Marker (with safe removal!)
        if self.trace_marker: 
            try: self.trace_marker.remove()
            except: pass
            
        if 'X_cm' in self.raw_data.columns:
            x_val = self.raw_data.at[idx, 'X_cm']
            y_val = self.raw_data.at[idx, 'Y_cm']
            if pd.notna(x_val):
                self.trace_marker, = self.trace_ax.plot(x_val, y_val, marker='o', color='black', markeredgecolor='white', markersize=8, zorder=10)
                self.trace_canvas.draw_idle()

    def auto_detect_behaviors(self):
        """Automatically finds Freezing Start, Freezing End, and Shelter Entry after the stimulus."""
        path = self.get_current_filepath()
        if not path or self.raw_data is None: 
            return
            
        df = self.raw_data
        post_stim_df = df[df['Rel_Time'] >= 0]
        if post_stim_df.empty: return
        
        # 1. AUTO-DETECT SHELTER
        zone_df = post_stim_df[post_stim_df['In zone'] == 1]
        if not zone_df.empty:
            self.file_data[path]['shelter_time'] = zone_df['Rel_Time'].iloc[0]
        else:
            self.file_data[path]['shelter_time'] = None

        # 2. AUTO-DETECT FREEZING
        try: thresh = float(self.txt_freeze_thresh.text())
        except: thresh = 1.0
        
        vel_col = 'Velocity_cm_s' if 'Velocity_cm_s' in df.columns else 'Velocity'
        
        # Find first frame below threshold
        freeze_start_df = post_stim_df[post_stim_df[vel_col] <= thresh]
        
        if not freeze_start_df.empty:
            f_start = freeze_start_df['Rel_Time'].iloc[0]
            self.file_data[path]['freeze_start'] = f_start
            
            # Find first frame ABOVE threshold AFTER the freeze started
            freeze_end_df = df[(df['Rel_Time'] > f_start) & (df[vel_col] > thresh)]
            if not freeze_end_df.empty:
                # If they reach the shelter BEFORE they stop freezing (e.g. drifting slowly into it)
                # Cap the freeze end at the shelter time to make logical sense.
                f_end = freeze_end_df['Rel_Time'].iloc[0]
                st = self.file_data[path]['shelter_time']
                
                if st is not None and f_end > st:
                    self.file_data[path]['freeze_end'] = st
                else:
                    self.file_data[path]['freeze_end'] = f_end
            else:
                self.file_data[path]['freeze_end'] = None
        else:
            self.file_data[path]['freeze_start'] = None
            self.file_data[path]['freeze_end'] = None

        # Instantly redraw the plots to show the new automated ranges!
        self.update_plots()
        self.update_table_view()
        self.statusBar().showMessage("Auto-Detection Complete!", 3000)

    def set_annotation(self, key):
        """Assigns the currently selected time to a behavior state."""
        path = self.get_current_filepath()
        if path and self.selected_idx is not None: 
            t = self.raw_data.at[self.selected_idx, 'Rel_Time']
            self.file_data[path][key] = t
            self.update_plots()
            self.update_table_view()

    def _update_event_col(self):
        """Dynamically generates the 4-bit Event column using pure Pandas index alignment."""
        if self.raw_data is None or self.raw_data.empty: return
        path = self.get_current_filepath()
        d = self.file_data[path]
        
        self.raw_data['Event'] = '0000' # Initialize all rows as 0000 (stim/freeze_start/freeze_end/shelter)

        def set_event_bit(time_val, bit_index):
            if time_val is not None:
                # Find the exact row index closest to the event time
                idx = (self.raw_data['Rel_Time'] - time_val).abs().idxmin()
                # Convert string to list, flip the bit, and save back
                current_event = list(self.raw_data.at[idx, 'Event'])
                current_event[bit_index] = '1'
                self.raw_data.at[idx, 'Event'] = "".join(current_event)

        # Assign the 4 bits
        set_event_bit(0.0, 0)                  # Bit 1: Stim Start
        set_event_bit(d.get('freeze_start'), 1)    # Bit 2: Freeze Start
        set_event_bit(d.get('freeze_end'), 2)      # Bit 3: Freeze End
        set_event_bit(d.get('shelter_time'), 3)    # Bit 4: Shelter Time

    def update_table_view(self):
        """Refreshes the PandasModel depending on the Clean View checkbox."""
        if self.raw_data is None: return
        
        # Always ensure the Event column is up to date before rendering!
        self._update_event_col()
        
        if self.chk_clean_view.isChecked():
            # Define desired columns, gracefully ignoring any that might not exist yet
            desired_cols = ['Trial time', 'Rel_Time', 'X_cm', 'Y_cm', 'Distance_cm', 'Velocity_cm_s', 'In zone', 'Event']
            valid_cols = [c for c in desired_cols if c in self.raw_data.columns]
            display_df = self.raw_data[valid_cols]
        else:
            # Show everything except messy temporary OpenCV variables
            display_df = self.raw_data.drop(columns=['Raw_Velocity_cm_s'], errors='ignore')
            
        self.table_view.setModel(PandasModel(display_df))


    def update_plots(self):
        """Redraws both Velocity and Trace plots based on selected window."""
        if self.raw_data is None: return
        path = self.get_current_filepath()
        d = self.file_data[path]
        
        pre = float(self.txt_pre_stim.text())
        post = float(self.txt_post_stim.text())
        full_df = self.raw_data[(self.raw_data['Rel_Time'] >= pre) & (self.raw_data['Rel_Time'] <= post)].copy()
        if full_df.empty:             
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No data available", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=self.ax.transAxes, 
                    fontsize=14, 
                    color='gray', 
                    weight='bold')
            
            self.canvas.draw()
            
            self.trace_ax.clear()
            self.trace_ax.text(0.5, 0.5, "No data available", 
                          horizontalalignment='center', 
                          verticalalignment='center', 
                          transform=self.trace_ax.transAxes, 
                          fontsize=14, 
                          color='gray', 
                          weight='bold')
            
            self.trace_canvas.draw()
                       
            return

        # Reset markers because ax.clear() will destroy them
        self.vel_marker = None
        self.trace_marker = None

        fs, fe, st = d['freeze_start'], d['freeze_end'], d['shelter_time']

        # --- THE CUTOFF LOGIC ---
        # If shelter_time is defined, slice data to cut off everything after it
        if st is not None:
            visible_df = full_df[full_df['Rel_Time'] <= st].copy()
        else:
            visible_df = full_df.copy()

        # ==========================================
        # 1. Update Velocity Plot
        # ==========================================
        self.ax.clear()
        self.ax.set_title(f"Velocity Profile | {d['mouse_id']} |{d['photostim']}")
        
        if 'Velocity_cm_s' in visible_df.columns:
                    self.ax.plot(visible_df['Rel_Time'], visible_df['Velocity_cm_s'], color='#1976d2', label='Velocity')
            
        self.ax.axvline(0, color='black', linestyle='--', label="Looming")
        self.ax.set_xlim(pre, post) # Keep stable timeline bounds
        self.ax.set_ylabel("Velocity (cm/s)")
        self.ax.set_xlabel("Time from Looming (s)")

        # Shade behavior zones
        if fs is not None:
            end_shade = fe if fe is not None else post
            self.ax.axvspan(fs, end_shade, color='#d32f2f', alpha=0.2, label='Freezing')
        if st is not None:
            self.ax.axvline(st, color='green', label='Shelter', linewidth=2)

        self.ax.legend(loc='upper right', fontsize='small')
        self.canvas.draw()

        # Update Labels
        self.lbl_freeze_start.setText(f"{fs:.2f}s" if fs is not None else "--")
        self.lbl_freeze_end.setText(f"{fe:.2f}s" if fe is not None else "--")
        self.lbl_shelter.setText(f"{st:.2f}s" if st is not None else "--")
        
        # Calculate Metrics for UI
        dur = f"{fe - fs:.2f} s" if (fs is not None and fe is not None) else "--"
        dist_v = "--"
        dist_col = 'Distance_cm'
        if fe is not None and st is not None and dist_col in visible_df.columns:
            sum_dist = visible_df[(visible_df['Rel_Time']>fe) & (visible_df['Rel_Time']<=st)][dist_col].sum()
            dist_v = f"{sum_dist:.2f} cm"
        self.lbl_metrics.setText(f"Freeze Duration: {dur}\nDistance (End->Shelter): {dist_v}")

        # ==========================================
        # 2. Update Trace Plot
        # ==========================================
        self.trace_ax.clear()
        self.trace_ax.set_title(f"Movement Trace (cm) | {d['mouse_id']} | {d['photostim']}", fontsize=10)
        
        if 'X_cm' in visible_df.columns:
            x, y = visible_df['X_cm'], visible_df['Y_cm']
            
            # Split Pre/Post looming paths (using the sliced visible_df!)
            self.trace_ax.plot(x[visible_df['Rel_Time']<0], y[visible_df['Rel_Time']<0], color='gray', alpha=0.5)
            self.trace_ax.plot(x[visible_df['Rel_Time']>=0], y[visible_df['Rel_Time']>=0], color='#1976d2')
            
            # --- NEW: Draw Black Marker at Stimulus Onset (Rel_Time = 0) ---
            stim_onset_df = visible_df[visible_df['Rel_Time'] >= 0]
            if not stim_onset_df.empty:
                self.trace_ax.plot(stim_onset_df.iloc[0]['X_cm'], stim_onset_df.iloc[0]['Y_cm'], 
                                   'o', color='black', markeredgecolor='white', markersize=7, zorder=6)

            # Highlight Freezing Zone in Orange
            if fs is not None:
                end_shade = fe if fe is not None else post
                fr_x = x[(visible_df['Rel_Time']>=fs) & (visible_df['Rel_Time']<=end_shade)]
                fr_y = y[(visible_df['Rel_Time']>=fs) & (visible_df['Rel_Time']<=end_shade)]
                self.trace_ax.plot(fr_x, fr_y, color='#d32f2f', linewidth=3, alpha=0.5)
                
            # --- Smart Spatial Limits ---
            try: chamber_w = float(self.txt_chamber_w.text())
            except: chamber_w = 30.5
            try: chamber_h = float(self.txt_chamber_h.text())
            except: chamber_h = 51.0
            try: shelter_h = float(self.txt_shelter_h.text())
            except: shelter_h = 10.0
            try: y_offset = float(self.txt_calib_y_offset.text())
            except: y_offset = 10.0
            
            if d['type'] == 'dlc':
                if d.get('correction_params') is None:
                    # Uncalibrated DLC -> Full Frame bounds scaled by px_cm
                    try: px_cm = float(self.txt_px_cm.text())
                    except: px_cm = 10.0
                    lim_w = float(self.txt_vid_w.text()) / px_cm
                    lim_h = float(self.txt_vid_h.text()) / px_cm
                    
                    self.trace_ax.set_xlim(0, lim_w)
                    self.trace_ax.set_ylim(0, lim_h)
                else:
                    # Calibrated DLC -> Locked to Chamber W by (Chamber H + Y-offset)
                    lim_w = chamber_w
                    lim_h = chamber_h + y_offset # <--- MATH LOCKED!
                    
                    self.trace_ax.set_xlim(0, lim_w)
                    self.trace_ax.set_ylim(0, lim_h)
            else:
                # Ethovision  -> Locked firmly to chamber size
                lim_w = chamber_w
                if d['type'] in ['dlc', 'saved_analysis']:
                    lim_h = chamber_h + y_offset
                else:
                    lim_h = chamber_h # Raw Ethovision is purely physical
           
                self.trace_ax.set_xlim(0, lim_w)
                self.trace_ax.set_ylim(0, lim_h)

            # Draw Shelter (Green Box anchored at bottom)
            shelter_patch = Polygon([[0,0], [lim_w,0], [lim_w, shelter_h], [0, shelter_h]], 
                                    closed=True, facecolor='green', alpha=0.3)
            self.trace_ax.add_patch(shelter_patch)

        self.trace_ax.set_aspect('equal')
        self.trace_canvas.draw()
        
        # Redraw tracking markers if within visible range
        if self.selected_idx and self.selected_idx in visible_df.index: 
            self._update_plot_markers()

    def draw_line_for_scale(self):
        """Allows user to draw a line across the chamber to auto-calculate px/cm."""
        if not HAS_CV2: return
        path = self.get_current_filepath()
        if not path or self.file_data[path].get('type') != 'dlc': 
            QMessageBox.warning(self, "Info", "Select a DLC (.h5) file first.")
            return
            
        v_path, _ = QFileDialog.getOpenFileName(self, "Select Video", os.path.dirname(path), "Video (*.mp4 *.avi *.mkv)")
        if not v_path: return
        
        cap = cv2.VideoCapture(v_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return
        
        pts = []
        w_name = "Draw Line (Click 2 points across chamber width) | ENTER=Done | C=Clear | Q=Quit"
        cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
        
        def ce(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((x, y))
                
        cv2.setMouseCallback(w_name, ce)
        
        while True:
            disp = frame.copy()
            for p in pts:
                cv2.circle(disp, p, 5, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(disp, pts[0], pts[1], (0, 255, 0), 2)
            
            cv2.imshow(w_name, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in [27, ord('q')]: 
                cv2.destroyAllWindows()
                return
            if k == ord('c'): pts = []
            if k == 13 and len(pts) == 2: break
            
        cv2.destroyAllWindows()
        
        # Calculate px_cm
        dist_px = np.sqrt((pts[0][0] - pts[1][0])**2 + (pts[0][1] - pts[1][1])**2)
        try: chamber_w = float(self.txt_chamber_w.text())
        except: chamber_w = 30.5
        
        px_cm = dist_px / chamber_w
        self.txt_px_cm.setText(f"{px_cm:.2f}")
        
        # Erase any existing 8-pt calibration since we are switching to simple mode
        for f in self.files:
            if self.file_data[f]['type'] == 'dlc':
                self.file_data[f]['correction_params'] = None
                
        QMessageBox.information(self, "Success", f"Scale Set: {px_cm:.2f} px/cm")
        
        # Force DLC engine to re-parse the coordinates using the new scale
        self.parse_file(path)    

    # ==========================================
    # LENS & PERSPECTIVE CALIBRATION
    # ==========================================
    def calibrate_video(self):
        """Runs the 8-point OpenCV Lens Distortion and Homography correction.
        Note: OpenCV strictly uses the TOP-LEFT corner as (0,0), which is the same as DLC
        """
        
        if not HAS_CV2:
            QMessageBox.warning(self, "Error", "OpenCV is required. Run: pip install scipy opencv-python")
            return
            
        path = self.get_current_filepath()
        if not path or self.file_data[path].get('type') != 'dlc':
            QMessageBox.information(self, "Info", "Calibration is only intended for DeepLabCut (.h5) files.")
            return

        v_path, _ = QFileDialog.getOpenFileName(self, "Select Video", os.path.dirname(path), "Video (*.mp4 *.avi *.mkv)")
        if not v_path: return
        
        cap = cv2.VideoCapture(v_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return
        
        h, w = frame.shape[:2]
        
        # --- Phase 1: Point Selection ---
        self.calib_points = []
        def ce(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.calib_points) < 8:
                self.calib_points.append((x, y))
        
        w_name = "8-Pt Calib: 4 Corners FIRST, then 4 Midsides | ENTER=Done | C=Clear | Q=Quit"
        cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(w_name, ce)
        
        while True:
            dfrm = frame.copy()
            for i, p in enumerate(self.calib_points):
                color = (0, 255, 0) if i < 4 else (0, 255, 255) # Green for corners, Yellow for mids
                cv2.circle(dfrm, p, 7, color, -1)
                cv2.putText(dfrm, str(i + 1), (p[0] + 15, p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(w_name, dfrm)
            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), 27]: 
                cv2.destroyAllWindows()
                return
            if key == ord('c'): self.calib_points = []
            if key == 13 and len(self.calib_points) == 8: break
        cv2.destroyAllWindows()
        
        # --- Phase 2: Minimize Lens Error ---
        res = opt.minimize(get_straightness_error, [0.0, w/2, h/2], 
                           args=(np.array(self.calib_points, dtype=np.float32), h, w), 
                           method='L-BFGS-B', bounds=[(-0.5, 0.5), (0, w), (0, h)])
        k1_auto, cx_auto, cy_auto = res.x
        
        # Setup real-world aspect ratio
        try: aspect_ratio = float(self.txt_chamber_w.text()) / float(self.txt_chamber_h.text())
        except: aspect_ratio = 30.5 / 51.0
        
        # Calculate Dimensions
        orig_corners = np.array([self.calib_points[:4]], dtype=np.float32)
        pt_TR = orig_corners[0][1]
        pt_BR = orig_corners[0][2] 
        pt_BL = orig_corners[0][3] 
        
        target_w = abs(pt_BR[0] - pt_BL[0])
        target_h = target_w / aspect_ratio
        
        # Ensure Output width stays locked to video width, and perfectly centered
        out_w = w 
        x_margin = (w - target_w) / 2
        y_margin = max(0, pt_TR[1]) # Prevent negative clipping
        
        out_h = int(target_h + y_margin * 2)
        output_dims = (out_w, out_h)
        
        dst_points = np.array([
            [x_margin, y_margin],                       # Top-Left
            [x_margin + target_w, y_margin],            # Top-Right
            [x_margin + target_w, y_margin + target_h], # Bottom-Right
            [x_margin, y_margin + target_h]             # Bottom-Left
        ], dtype="float32")
        
        # --- Phase 3: Trackbar Fine-Tune ---
        tw_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"
        cv2.namedWindow(tw_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Set max limit to 20000 so center (0 correction) is at 10000
        initial_k1_pos = int((k1_auto * 5000) + 10000)
        initial_cx_pos = int(cx_auto)
        initial_cy_pos = int(cy_auto)
        
        cv2.createTrackbar("Correction", tw_name, initial_k1_pos, 20000, lambda x:None)
        cv2.createTrackbar("Center X", tw_name, initial_cx_pos, w, lambda x: None)
        cv2.createTrackbar("Center Y", tw_name, initial_cy_pos, h, lambda x: None)
        
        last_params = (); final_params = None
        
        while True:
            k1_pos = cv2.getTrackbarPos("Correction", tw_name)
            cx_pos = cv2.getTrackbarPos("Center X", tw_name)
            cy_pos = cv2.getTrackbarPos("Center Y", tw_name)
            
            if (k1_pos, cx_pos, cy_pos) != last_params:
                k1_fine = (k1_pos - 10000) / 5000.0
                
                temp_cam_matrix = np.array([[w, 0, cx_pos], [0, w, cy_pos], [0,0,1]], dtype=np.float32)
                temp_dist_coeffs = np.array([k1_fine, 0, 0, 0, 0], dtype=np.float32)
                
                new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(temp_cam_matrix, temp_dist_coeffs, (w,h), 1, (w,h))
                lens_corrected_frame = cv2.undistort(frame, temp_cam_matrix, temp_dist_coeffs, None, new_cam_matrix)
                corrected_corners = cv2.undistortPoints(orig_corners, temp_cam_matrix, temp_dist_coeffs, P=new_cam_matrix).reshape(-1, 2)
                
                M = cv2.getPerspectiveTransform(corrected_corners, dst_points)
                preview_frame = cv2.warpPerspective(lens_corrected_frame, M, output_dims)
                
                # Draw Box Guide
                preview_with_guide = preview_frame.copy()
                start_pt = (int(x_margin), int(y_margin))
                end_pt = (int(x_margin + target_w), int(y_margin + target_h))
                self.draw_dashed_rectangle(preview_with_guide, start_pt, end_pt, (0, 0, 255), thickness=2, dash_length=15)
                
                min_ui_width = 1000
                if preview_with_guide.shape[1] < min_ui_width:
                    pad = (min_ui_width - preview_with_guide.shape[1]) // 2
                    final_display = cv2.copyMakeBorder(preview_with_guide, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
                else:
                    final_display = preview_with_guide
                    
                cv2.imshow(tw_name, final_display)
                last_params = (k1_pos, cx_pos, cy_pos)
            
            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), 27]: break
            if key == ord('a'):                
                final_params = {
                    'k1': k1_fine, 'cx': cx_pos, 'cy': cy_pos, 'M': M,
                    'frame_dims': output_dims, 'chamber_dims': (target_w, target_h),
                    'new_cam_matrix': new_cam_matrix, 'orig_w': w, 'orig_h': h
                }
                break
            if key == ord('r'):
                cv2.setTrackbarPos("Correction", tw_name, initial_k1_pos)
                cv2.setTrackbarPos("Center X", tw_name, initial_cx_pos)
                cv2.setTrackbarPos("Center Y", tw_name, initial_cy_pos)
                
        cv2.destroyAllWindows()
        
        # --- Apply Matrix to Data ---
        if final_params:
            try: chamber_width = float(self.txt_chamber_w.text())
            except: chamber_width = 30.5
            
            px_cm = final_params['chamber_dims'][0] / chamber_width
            self.txt_px_cm.setText(f"{px_cm:.2f}")
            self.txt_vid_w.setText(str(final_params['frame_dims'][0]))
            self.txt_vid_h.setText(str(final_params['frame_dims'][1]))
            
            # Apply to all DLC files
            for f in self.files:
                if self.file_data[f]['type'] == 'dlc':
                    self.file_data[f]['correction_params'] = final_params
            
            self.parse_file(path) # Re-parse current file
            QMessageBox.information(self, "Success", "Lens Correction & Homography Applied!")

    def draw_dashed_rectangle(self, img, top_left, bottom_right, color, thickness=1, dash_length=10):
        """Draws a dashed rectangle on the given image array."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        def draw_dashed_line(pt1, pt2):
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist == 0: return
            num_dashes = int(dist / (dash_length * 2))
            for i in range(max(1, num_dashes)):
                start_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i * 2) / (num_dashes * 2))
                start_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i * 2) / (num_dashes * 2))
                end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (i * 2 + 1) / (num_dashes * 2))
                end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (i * 2 + 1) / (num_dashes * 2))
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
                
        draw_dashed_line((x1, y1), (x2, y1)) # Top
        draw_dashed_line((x2, y1), (x2, y2)) # Right
        draw_dashed_line((x2, y2), (x1, y2)) # Bottom
        draw_dashed_line((x1, y2), (x1, y1)) # Left

    # ==========================================
    # JSON / CALIBRATION SAVING
    # ==========================================
    def save_calibration_json(self):
        path = self.get_current_filepath()
        if not path or self.file_data[path].get('correction_params') is None: return
        cp = self.file_data[path]['correction_params']
        
        data = {
            'k1': float(cp['k1']), 
            'cx': float(cp['cx']), 
            'cy': float(cp['cy']),
            'M': cp['M'].tolist(), 
            'frame_dims': [float(x) for x in cp['frame_dims']],     # <--- FIXED float32 bug
            'chamber_dims': [float(x) for x in cp['chamber_dims']], # <--- FIXED
            'new_cam_matrix': cp['new_cam_matrix'].tolist(), 
            'orig_w': float(cp['orig_w']), 
            'orig_h': float(cp['orig_h'])
        }
        
        s_path, _ = QFileDialog.getSaveFileName(self, "Save JSON", self.output_dir, "JSON (*.json)")
        if s_path:
            with open(s_path, 'w') as f: json.dump(data, f, indent=4)

    def load_calibration_json(self):
        if not self.files: return
        l_path, _ = QFileDialog.getOpenFileName(self, "Load JSON", self.output_dir, "JSON (*.json)")
        if not l_path: return
        
        with open(l_path, 'r') as f: data = json.load(f)
        cp = {
            'k1': data['k1'], 'cx': data['cx'], 'cy': data['cy'],
            'M': np.array(data['M'], dtype=np.float32), 'frame_dims': tuple(data['frame_dims']),
            'chamber_dims': tuple(data['chamber_dims']), 'new_cam_matrix': np.array(data['new_cam_matrix'], dtype=np.float32),
            'orig_w': data['orig_w'], 'orig_h': data['orig_h']
        }
        
        for f in self.files:
            if self.file_data[f]['type'] == 'dlc':
                self.file_data[f]['correction_params'] = cp
                
        self.parse_file(self.get_current_filepath())

    # ==========================================
    # UI ROUTING HELPERS
    # ==========================================
    def get_current_filepath(self):
        if 0 <= self.current_file_idx < len(self.files): return self.files[self.current_file_idx]
        return None
        
    def prev_file(self):
        if self.current_file_idx > 0: self.file_combo.setCurrentIndex(self.current_file_idx - 1)
        
    def next_file(self):
        if self.current_file_idx < len(self.files) - 1: self.file_combo.setCurrentIndex(self.current_file_idx + 1)
        
    def on_file_selected(self, idx):
        if idx >= 0:
            self.current_file_idx = idx
            self.parse_file(self.files[idx])
            
    def on_metadata_changed(self):
        p = self.get_current_filepath()
        if p:
            self.file_data[p]['mouse_id'] = self.txt_mouse_id.text().strip()
            self.file_data[p]['photostim'] = self.txt_photostim.text().strip()
            self.update_plots()
            
    def on_post_stim_changed(self):
        p = self.get_current_filepath()
        if p:
            try:
                self.file_data[p]['post_stim'] = float(self.txt_post_stim.text())
                self.update_plots()
            except ValueError: pass
            
    def on_table_clicked(self, idx): 
        self.set_selected_point(idx.row())

    def copy_plot_to_clipboard(self, canvas):
        QApplication.clipboard().setPixmap(canvas.grab())
        self.statusBar().showMessage("Copied to clipboard!", 2000)

    # ==========================================
    # EXPORT PIPELINE
    # ==========================================
    def master_export_current(self):
        path = self.get_current_filepath()
        if not path or not self.output_dir: return
        d = self.file_data[path]
        
        bn = f"{d['mouse_id']}_{d['photostim']}"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save PNG Images
        self.figure.savefig(os.path.join(self.output_dir, f"{bn}_Velocity_{ts}.png"), dpi=200, bbox_inches='tight')
        self.trace_fig.savefig(os.path.join(self.output_dir, f"{bn}_Trace_{ts}.png"), dpi=200, bbox_inches='tight')
        
        # 2. Save Full Coordinates DataFrame
        if self.raw_data is not None:
            export_df = self.raw_data.copy()
            out_path = os.path.join(self.output_dir, f"{bn}_PlotData_{ts}.csv")
            
            # Create the header lines
            header_lines = [
                f'"Video File","{d.get("video_name", "")}"\n',
                f'"Stim Start","{d.get("stim_start", "")}"\n',
                f'"Freeze Start","{d.get("freeze_start", "") if d.get("freeze_start") is not None else ""}"\n',
                f'"Freeze End","{d.get("freeze_end", "") if d.get("freeze_end") is not None else ""}"\n',
                f'"Shelter Time","{d.get("shelter_time", "") if d.get("shelter_time") is not None else ""}"\n',
                f'"Chamber_w", {self.txt_chamber_w.text()}\n',
                f'"Chamber_h", {self.txt_chamber_h.text()}\n',
                f'"Shelter_h", {self.txt_shelter_h.text()}\n',
                f'"Freeze_threshold", {self.txt_freeze_thresh.text()}\n',
                '"---","---"\n' # A visual separator before the data starts
            ]
            
            # Write the file: header first, then the dataframe
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(header_lines)
                # Append the dataframe directly to the same file object
                export_df.to_csv(f, index=False)
            
        # 3. Calculate Advanced Metrics
        df = self.raw_data
        max_speed = "N/A"
        dist_to_shelter = "N/A"
        
        if df is not None:
            # Max speed AFTER looming starts
            post_df = df[df['Rel_Time'] >= 0]
            if 'Velocity_cm_s' in df.columns and not post_df.empty:
                max_speed = post_df['Velocity_cm_s'].max()
                
            # Distance traveled from end of freeze -> shelter
            fe, st = d['freeze_end'], d['shelter_time']
            if fe is not None and st is not None and 'Distance_cm' in df.columns:
                dist_to_shelter = df[(df['Rel_Time'] > fe) & (df['Rel_Time'] <= st)]['Distance_cm'].sum()

        # 4. Compile Row for the Master Results CSV
        # FIXED: Explicit 'is not None' checks so that 0.0s is evaluated correctly!
        freeze_dur = "N/A"
        if d['freeze_end'] is not None and d['freeze_start'] is not None:
            freeze_dur = d['freeze_end'] - d['freeze_start']
    
        m = {
            "Filename": os.path.basename(path), 
            "Mouse_ID": d['mouse_id'], 
            "PhotoStim": d['photostim'],
            "Stim_Start_Time": d['stim_start'], 
            "Post_Range": d['post_stim'],
            "Manual_Freeze_Start": d['freeze_start'], 
            "Manual_Freeze_End": d['freeze_end'], 
            "Manual_Shelter_Time": d['shelter_time'],
            "Calculated_Freeze_Duration": freeze_dur,
            "Distance_To_Shelter_cm": dist_to_shelter,
            "Max_Speed_cm_s": max_speed
        }
        
        df_new = pd.DataFrame([m])
        csv_p = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
        
        if os.path.exists(csv_p):
            df_old = pd.read_csv(csv_p)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(csv_p, index=False)
        self.statusBar().showMessage("Master Export Complete! Metrics, Data, and Plots saved.", 4000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoomingAnalyzer()
    win.show()
    sys.exit(app.exec_())
