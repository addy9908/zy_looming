import sys
import os
import datetime
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableView, QLabel, QLineEdit, 
                             QFormLayout, QComboBox, QSplitter, QMessageBox, QAbstractItemView, QDialog)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QColor, QBrush, QStandardItemModel, QStandardItem

try:
    import cv2
    import scipy.optimize as opt
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# --- 8-Point Correction Math Helpers ---
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

class CalibrationDialog(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("8-Pt Calibrate: TL -> TM -> TR -> RM -> BR -> BM -> BL -> LM")
        self.resize(900, 700)
        self.pts = []
        self.frame = frame
        
        layout = QVBoxLayout(self)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.imshow(self.frame)
        self.ax.axis('off')
        self.ax.set_title("Click 8 pts clockwise starting Top-Left (Corners & Midsides)", color='red')
        self.line, = self.ax.plot([], [], 'o-', color='yellow', linewidth=2, markersize=8)
        
        self.canvas.mpl_connect('button_press_event', self.onclick)
        
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        self.btn_done = QPushButton("Done")
        self.btn_done.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_done)
        layout.addLayout(btn_layout)

    def onclick(self, event):
        if not event.inaxes: return
        if len(self.pts) < 8:
            self.pts.append((event.xdata, event.ydata))
            self.update_plot()

    def clear(self):
        self.pts = []
        self.update_plot()

    def update_plot(self):
        if self.pts:
            xs, ys = zip(*self.pts)
            if len(self.pts) == 8: 
                xs = xs + (xs[0],)
                ys = ys + (ys[0],)
            self.line.set_data(xs, ys)
        else:
            self.line.set_data([], [])
        self.canvas.draw()
        
    def get_src_points(self):
        if len(self.pts) == 8:
            return np.array(self.pts, dtype="float32")
        return None

class ROIDialog(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Custom ROI (Left Click: Add | Right Click: Finish)")
        self.resize(900, 700)
        self.polygon_pts = []
        self.frame = frame
        
        layout = QVBoxLayout(self)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.imshow(self.frame)
        self.ax.axis('off')
        self.line, = self.ax.plot([], [], 'o-', color='purple', linewidth=2)
        
        self.canvas.mpl_connect('button_press_event', self.onclick)
        
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        self.btn_done = QPushButton("Done")
        self.btn_done.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_done)
        layout.addLayout(btn_layout)

    def onclick(self, event):
        if not event.inaxes: return
        if event.button == 1:
            self.polygon_pts.append((event.xdata, event.ydata))
        elif event.button == 3 and len(self.polygon_pts) > 1:
            self.polygon_pts.append(self.polygon_pts[0])
        self.update_plot()
        
    def clear(self):
        self.polygon_pts = []
        self.update_plot()

    def update_plot(self):
        if self.polygon_pts:
            xs, ys = zip(*self.polygon_pts)
            self.line.set_data(xs, ys)
        else:
            self.line.set_data([], [])
        self.canvas.draw()
        
    def get_path(self):
        if len(self.polygon_pts) > 2: return Path(self.polygon_pts)
        return None

class EthoVisionAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Enterprise - EthoVision & DLC Analyzer")
        self.resize(1500, 950)
        
        self.files = []
        self.current_file_idx = -1
        self.raw_data = None
        self.current_plot_df = None 
        self.plot_export_df = None  
        self.selected_idx = None
        self.output_dir = ""
        self.file_data = {} 
        
        self.vel_marker = None
        self.trace_marker = None
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT PANEL ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(330)
        
        self.btn_load = QPushButton("📂 Load Excel / h5 / CSV Files")
        self.btn_load.clicked.connect(self.load_files)
        left_layout.addWidget(self.btn_load)
        
        left_layout.addWidget(QLabel("\n<b>DeepLabCut Settings (.h5)</b>"))
        dlc_form = QFormLayout()
        self.txt_fps = QLineEdit("30.0")
        self.txt_px_cm = QLineEdit("10.0")
        self.txt_vid_h = QLineEdit("480")
        self.txt_likelihood = QLineEdit("0.3") 
        
        self.combo_bp = CheckableComboBox()
        parts = ["nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", 
                 "right_eye", "neck", "mid_back", "mouse_center", "mid_backend", "mid_backend2", 
                 "mid_backend3", "tail_base", "tail1", "tail2", "tail3", "tail4", "tail5", 
                 "left_shoulder", "left_shouder", "left_midside", "left_hip", "right_shoulder", 
                 "right_shouder", "right_midside", "right_hip", "tail_end", "head_midpoint",
                 "Average (center+back)", "Average (shoulders)", "Average (midsides)"]
        self.combo_bp.add_items(parts, default_checked=["mouse_center"])
        
        dlc_form.addRow("Video FPS:", self.txt_fps)
        dlc_form.addRow("Fallback Px/cm:", self.txt_px_cm)
        dlc_form.addRow("Fallback Vid H:", self.txt_vid_h)
        dlc_form.addRow("Likelihood Thresh:", self.txt_likelihood)
        dlc_form.addRow("Track Part(s):", self.combo_bp)
        left_layout.addLayout(dlc_form)
        
        dlc_btns = QHBoxLayout()
        self.btn_calib = QPushButton("📏 8-Pt Calibrate")
        self.btn_calib.clicked.connect(self.calibrate_video)
        self.btn_draw_roi = QPushButton("🎥 Draw ROI")
        self.btn_draw_roi.clicked.connect(self.draw_roi)
        dlc_btns.addWidget(self.btn_calib)
        dlc_btns.addWidget(self.btn_draw_roi)
        left_layout.addLayout(dlc_btns)
        
        left_layout.addWidget(QLabel("\n<b>Global Parameters</b>"))
        param_form = QFormLayout()
        self.txt_stim_start = QLineEdit("60")
        self.txt_pre_stim = QLineEdit("-5")
        self.txt_stim_name = QLineEdit("Looming-On")
        self.txt_rolling_avg = QLineEdit("1")
        self.txt_chamber_w = QLineEdit("40") 
        self.txt_chamber_h = QLineEdit("40")
        self.txt_shelter_x = QLineEdit("15")
        self.txt_shelter_y = QLineEdit("15")
        
        inputs = [self.txt_stim_start, self.txt_pre_stim, self.txt_rolling_avg, 
                  self.txt_chamber_w, self.txt_chamber_h, self.txt_shelter_x, 
                  self.txt_shelter_y, self.txt_fps, self.txt_px_cm, self.txt_vid_h, self.txt_likelihood]
        
        for txt in inputs:
            txt.editingFinished.connect(self.recalculate_data)
        self.combo_bp.model().dataChanged.connect(self.recalculate_data)

        param_form.addRow("Stim Start (s):", self.txt_stim_start)
        param_form.addRow("Pre-Range (s):", self.txt_pre_stim)
        param_form.addRow("Legend Name:", self.txt_stim_name)
        param_form.addRow("Rolling Window:", self.txt_rolling_avg)
        param_form.addRow("Chamber W (cm):", self.txt_chamber_w)
        param_form.addRow("Chamber H (cm):", self.txt_chamber_h)
        param_form.addRow("Shelter W (cm):", self.txt_shelter_x)
        param_form.addRow("Shelter H (cm):", self.txt_shelter_y)
        left_layout.addLayout(param_form)
        
        left_layout.addWidget(QLabel("\n<b>Export Tools</b>"))
        self.btn_master_export = QPushButton("🚀 Master Export (Current File)")
        self.btn_master_export.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 6px;")
        self.btn_master_export.clicked.connect(self.master_export_current)
        left_layout.addWidget(self.btn_master_export)
        
        self.btn_export_plot_data = QPushButton("Export Current Plot Data")
        self.btn_export_plot_data.clicked.connect(self.export_plot_data)
        left_layout.addWidget(self.btn_export_plot_data)
        
        self.btn_export_csv = QPushButton("Export Results Only (CSV)")
        self.btn_export_csv.clicked.connect(self.export_results)
        left_layout.addWidget(self.btn_export_csv)
        
        left_layout.addWidget(QLabel("\n<b>Moving Trace View</b>"))
        self.trace_fig = Figure(figsize=(3, 3))
        self.trace_ax = self.trace_fig.add_subplot(111)
        self.trace_fig.tight_layout(pad=0.5)
        self.trace_canvas = FigureCanvas(self.trace_fig)
        self.trace_canvas.setMinimumHeight(280)
        self.trace_canvas.mpl_connect('button_press_event', self.on_trace_click)
        left_layout.addWidget(self.trace_canvas)
        
        trace_btn_layout = QHBoxLayout()
        self.btn_copy_trace = QPushButton("Copy")
        self.btn_copy_trace.clicked.connect(lambda: self.copy_plot_to_clipboard(self.trace_canvas))
        self.btn_save_trace = QPushButton("Save PNG")
        self.btn_save_trace.clicked.connect(lambda: self.save_plot_as_png(self.trace_fig, "TracePlot"))
        trace_btn_layout.addWidget(self.btn_copy_trace)
        trace_btn_layout.addWidget(self.btn_save_trace)
        left_layout.addLayout(trace_btn_layout)
        left_layout.addStretch()

        # --- RIGHT PANEL ---
        right_splitter = QSplitter(Qt.Vertical)
        top_right_widget = QWidget()
        top_right_layout = QVBoxLayout(top_right_widget)
        
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.clicked.connect(self.prev_file)
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        self.btn_next = QPushButton("▶")
        self.btn_next.clicked.connect(self.next_file)
        nav_layout.addWidget(QLabel("<b>File:</b>"))
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.file_combo, 1)
        nav_layout.addWidget(self.btn_next)
        top_right_layout.addLayout(nav_layout)
        
        meta_layout = QHBoxLayout()
        self.txt_mouse_id = QLineEdit()
        self.txt_photostim = QLineEdit()
        self.txt_post_stim = QLineEdit("10.0")
        
        self.txt_mouse_id.editingFinished.connect(self.on_metadata_changed)
        self.txt_photostim.editingFinished.connect(self.on_metadata_changed)
        self.txt_post_stim.editingFinished.connect(self.on_post_stim_changed)
        
        meta_layout.addWidget(QLabel("Mouse ID:"))
        meta_layout.addWidget(self.txt_mouse_id)
        meta_layout.addWidget(QLabel("PhotoStim:"))
        meta_layout.addWidget(self.txt_photostim)
        meta_layout.addWidget(QLabel("<b>Post-Range:</b>"))
        meta_layout.addWidget(self.txt_post_stim)
        top_right_layout.addLayout(meta_layout)
        
        nav_and_ann_layout = QHBoxLayout()
        nav_col = QVBoxLayout()
        nav_col.addWidget(QLabel("<b>Point Navigator:</b>"))
        pt_btns = QHBoxLayout()
        self.btn_prev_point = QPushButton("Prev Frame")
        self.btn_prev_point.clicked.connect(self.prev_point)
        self.btn_next_point = QPushButton("Next Frame")
        self.btn_next_point.clicked.connect(self.next_point)
        pt_btns.addWidget(self.btn_prev_point)
        pt_btns.addWidget(self.btn_next_point)
        nav_col.addLayout(pt_btns)
        
        self.lbl_selected_point = QLabel("Selected: -- s")
        self.lbl_selected_point.setStyleSheet("font-weight: bold; color: #1565c0; font-size: 13px;")
        nav_col.addWidget(self.lbl_selected_point)
        nav_col.addStretch()
        nav_and_ann_layout.addLayout(nav_col)
        
        ann_col = QVBoxLayout()
        ann_col.addWidget(QLabel("<b>Manual Annotation:</b>"))
        row1 = QHBoxLayout()
        self.btn_set_fstart = QPushButton("Set Freezing Start")
        self.lbl_fstart = QLabel("0.00s")
        self.btn_set_fstart.clicked.connect(lambda: self.set_annotation('fstart'))
        row1.addWidget(self.btn_set_fstart); row1.addWidget(self.lbl_fstart)
        
        row2 = QHBoxLayout()
        self.btn_set_fend = QPushButton("Set Freezing End")
        self.lbl_fend = QLabel("--")
        self.btn_set_fend.clicked.connect(lambda: self.set_annotation('fend'))
        row2.addWidget(self.btn_set_fend); row2.addWidget(self.lbl_fend)
        
        row3 = QHBoxLayout()
        self.btn_set_shelter = QPushButton("Set Back to Shelter")
        self.lbl_shelter = QLabel("--")
        self.btn_set_shelter.clicked.connect(lambda: self.set_annotation('shelter'))
        row3.addWidget(self.btn_set_shelter); row3.addWidget(self.lbl_shelter)
        
        ann_col.addLayout(row1)
        ann_col.addLayout(row2)
        ann_col.addLayout(row3)
        nav_and_ann_layout.addLayout(ann_col)
        
        calc_col = QVBoxLayout()
        calc_col.addWidget(QLabel("<b>Metrics:</b>"))
        self.lbl_f_dur = QLabel("Dur: --")
        self.lbl_dist = QLabel("Dist (E->S): N/A")
        self.lbl_f_dur.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.lbl_dist.setStyleSheet("color: #2e7d32; font-weight: bold;")
        calc_col.addWidget(self.lbl_f_dur)
        calc_col.addWidget(self.lbl_dist)
        calc_col.addStretch()
        nav_and_ann_layout.addLayout(calc_col)
        
        top_right_layout.addLayout(nav_and_ann_layout)
        
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.clicked.connect(self.on_table_clicked)
        top_right_layout.addWidget(self.table_view)
        right_splitter.addWidget(top_right_widget)
        
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_velocity_click)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        v_btns = QHBoxLayout()
        v_btns.addWidget(self.toolbar)
        self.btn_copy_vel = QPushButton("Copy Velocity")
        self.btn_copy_vel.clicked.connect(lambda: self.copy_plot_to_clipboard(self.canvas))
        self.btn_save_vel = QPushButton("Save PNG")
        self.btn_save_vel.clicked.connect(lambda: self.save_plot_as_png(self.figure, "VelocityPlot"))
        v_btns.addWidget(self.btn_copy_vel)
        v_btns.addWidget(self.btn_save_vel)
        
        plot_layout.addLayout(v_btns)
        plot_layout.addWidget(self.canvas)
        right_splitter.addWidget(plot_widget)
        
        right_splitter.setSizes([480, 420])
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter)

    # --- Loading & Parsing ---
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Data (*.xlsx *.h5 *.csv)")
        if files:
            # Prevent accidental load of the batch summary
            files = [f for f in files if "EthoVision_Batch_Results" not in os.path.basename(f)]
            if not files: return
            
            files.sort(key=lambda x: os.path.basename(x).lower())
            self.files = files
            self.output_dir = os.path.join(os.path.dirname(self.files[0]), "Output")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            csv_path = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
            df_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
            self.file_data = {}

            for f in self.files:
                m_id, p_stim = "N/A", "N/A"
                if f.endswith('.xlsx'):
                    try:
                        tdf = pd.read_excel(f, nrows=50, header=None)
                        idx = tdf[tdf.eq("Trial time").any(axis=1)].index[0]
                        mdf = tdf.iloc[:idx]
                        for _, row in mdf.iterrows():
                            if pd.notna(row[0]):
                                k = str(row[0]).lower()
                                if "mouse" in k: m_id = str(row[1]).strip()
                                elif "stim" in k: p_stim = str(row[1]).strip()
                    except: pass
                elif f.endswith('.csv'):
                    fname = os.path.basename(f)
                    if '_PlotData_' in fname:
                        base = fname.split('_PlotData_')[0]
                        parts = base.rsplit('_', 1)
                        if len(parts) == 2: m_id, p_stim = parts[0], parts[1]
                        else: m_id = base
                else:
                    parts = os.path.basename(f).split('_')
                    m_id = parts[0] if len(parts) > 0 else "Unknown"

                self.file_data[f] = {
                    'mouse_id': m_id, 'photostim': p_stim, 
                    'fstart': 0.0, 'fend': None, 'shelter': None, 
                    'post_stim': 10.0, 'stim_start': float(self.txt_stim_start.text()),
                    'roi_path': None, 'frame_shape': None, 'homography_M': None
                }
                
                if df_csv is not None and not df_csv.empty:
                    if all(c in df_csv.columns for c in ['Filename', 'Mouse_ID', 'PhotoStim']):
                        if f.endswith('.csv'):
                            # For imported CSV Plot Data, we match using strictly Mouse_ID and PhotoStim
                            match = df_csv[(df_csv['Mouse_ID'].astype(str).str.strip() == m_id) &
                                           (df_csv['PhotoStim'].astype(str).str.strip() == p_stim)]
                        else:
                            # For raw files, match filename as well
                            match = df_csv[(df_csv['Filename'] == os.path.basename(f)) & 
                                           (df_csv['Mouse_ID'].astype(str).str.strip() == m_id) &
                                           (df_csv['PhotoStim'].astype(str).str.strip() == p_stim)]
                        
                        if not match.empty:
                            rec = match.iloc[-1]
                            self.file_data[f]['fstart'] = float(rec['Manual_Freeze_Start']) if pd.notna(rec['Manual_Freeze_Start']) and rec['Manual_Freeze_Start'] != "N/A" else 0.0
                            self.file_data[f]['fend'] = float(rec['Manual_Freeze_End']) if pd.notna(rec['Manual_Freeze_End']) and rec['Manual_Freeze_End'] != "N/A" else None
                            self.file_data[f]['shelter'] = float(rec['Manual_Shelter_Time']) if pd.notna(rec['Manual_Shelter_Time']) and rec['Manual_Shelter_Time'] != "N/A" else None
                            self.file_data[f]['post_stim'] = float(rec['Post_Range']) if pd.notna(rec['Post_Range']) and rec['Post_Range'] != "N/A" else 10.0
                            if 'Stim_Start_Time' in rec and pd.notna(rec['Stim_Start_Time']):
                                self.file_data[f]['stim_start'] = float(rec['Stim_Start_Time'])

            self.file_combo.blockSignals(True)
            self.file_combo.clear()
            self.file_combo.addItems([os.path.basename(f) for f in self.files])
            self.file_combo.blockSignals(False)
            
            self.current_file_idx = 0
            self.file_combo.setCurrentIndex(0)
            self.parse_file(self.files[0])

    def get_current_filepath(self):
        if 0 <= self.current_file_idx < len(self.files):
            return self.files[self.current_file_idx]
        return None

    def prev_file(self):
        if self.current_file_idx > 0:
            self.file_combo.setCurrentIndex(self.current_file_idx - 1)

    def next_file(self):
        if self.current_file_idx < len(self.files) - 1:
            self.file_combo.setCurrentIndex(self.current_file_idx + 1)
            
    def on_file_selected(self, idx):
        if idx >= 0:
            self.current_file_idx = idx
            self.parse_file(self.files[idx])

    def calibrate_video(self):
        if not HAS_CV2:
            QMessageBox.warning(self, "Error", "OpenCV is required. Run: pip install scipy opencv-python")
            return
            
        path = self.get_current_filepath()
        if not path or not path.endswith('.h5'):
            QMessageBox.information(self, "Info", "Calibration is intended for DeepLabCut (.h5) files.")
            return

        v_path, _ = QFileDialog.getOpenFileName(self, "Select Video", os.path.dirname(path), "Video (*.mp4 *.avi *.mkv)")
        if not v_path: return

        try:
            cap = cv2.VideoCapture(v_path); ret, frame = cap.read(); cap.release()
            if not ret: return
            
            h, w = frame.shape[:2]
            
            # Phase 1: Point Selection
            self.calib_points = []
            def ce(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(self.calib_points) < 8:
                    self.calib_points.append((x, y))
            
            w_name = "8-Pt Calib (TL->TM->TR->RM->BR->BM->BL->LM) | ENTER=Done | C=Clear | Q=Quit"
            cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(w_name, ce)
            
            while True:
                dfrm = frame.copy()
                for i, p in enumerate(self.calib_points):
                    color = (0, 255, 0) if i < 4 else (0, 255, 255)
                    cv2.circle(dfrm, p, 7, color, -1)
                    cv2.putText(dfrm, str(i + 1), (p[0] + 15, p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow(w_name, dfrm)
                key = cv2.waitKey(20) & 0xFF
                if key in [ord('q'), 27]: cv2.destroyAllWindows(); return
                if key == ord('c'): self.calib_points = []
                if key == 13 and len(self.calib_points) == 8: break
            cv2.destroyAllWindows()
            
            # Phase 2: Minimize Error
            res = opt.minimize(get_straightness_error, [0.0, w/2, h/2], args=(np.array(self.calib_points, dtype=np.float32), h, w), method='L-BFGS-B', bounds=[(-0.5, 0.5), (0, w), (0, h)])
            k1_auto, cx_auto, cy_auto = res.x
            
            try: aspect = float(self.txt_chamber_w.text()) / float(self.txt_chamber_h.text())
            except: aspect = 1.0
            
            # Phase 3: Trackbar Fine Tune
            tw_name = "Fine-Tune | A=Apply | R=Reset | Q=Quit"
            cv2.namedWindow(tw_name, cv2.WINDOW_NORMAL)
            i_k1, i_cx, i_cy = int((k1_auto * 10000) + 4000), int(cx_auto), int(cy_auto)
            cv2.createTrackbar("Correction", tw_name, i_k1, 8000, lambda x:None)
            cv2.createTrackbar("Center X", tw_name, i_cx, w, lambda x: None)
            cv2.createTrackbar("Center Y", tw_name, i_cy, h, lambda x: None)
            
            last_p = (); orig_corners = np.array([self.calib_points[:4]], dtype=np.float32); final_params = None
            while True:
                k1_pos, cx_pos, cy_pos = cv2.getTrackbarPos("Correction", tw_name), cv2.getTrackbarPos("Center X", tw_name), cv2.getTrackbarPos("Center Y", tw_name)
                if (k1_pos, cx_pos, cy_pos) != last_p:
                    k1_fine = (k1_pos - 4000) / 10000.0
                    temp_cam = np.array([[w, 0, cx_pos], [0, w, cy_pos], [0,0,1]], dtype=np.float32)
                    temp_dist = np.array([k1_fine, 0, 0, 0, 0], dtype=np.float32)
                    new_cam, _ = cv2.getOptimalNewCameraMatrix(temp_cam, temp_dist, (w,h), 1, (w,h))
                    lens_c = cv2.undistort(frame, temp_cam, temp_dist, None, new_cam)
                    cor_corners = cv2.undistortPoints(orig_corners, temp_cam, temp_dist, P=new_cam).reshape(-1, 2)
                    
                    max_w = h; max_h = int(max_w / aspect)
                    dst = np.array([[0,0], [max_w-1,0], [max_w-1,max_h-1], [0,max_h-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(cor_corners, dst)
                    prev = cv2.warpPerspective(lens_c, M, (max_w, max_h))
                    cv2.imshow(tw_name, prev); last_p = (k1_pos, cx_pos, cy_pos)
                
                key = cv2.waitKey(20) & 0xFF
                if key in [ord('q'), 27]: break
                if key == ord('a'):
                    final_params = {'k1': k1_fine, 'cx': cx_pos, 'cy': cy_pos, 'M': M, 'dims': (max_w, max_h), 'new_cam': new_cam, 'orig_w': w, 'orig_h': h}
                    break
                if key == ord('r'):
                    cv2.setTrackbarPos("Correction", tw_name, i_k1); cv2.setTrackbarPos("Center X", tw_name, i_cx); cv2.setTrackbarPos("Center Y", tw_name, i_cy)
            cv2.destroyAllWindows()
            
            if final_params:
                try: cw = float(self.txt_chamber_w.text())
                except: cw = 40.0
                px_cm = final_params['dims'][0] / cw
                self.txt_px_cm.setText(f"{px_cm:.2f}")
                self.txt_vid_h.setText(str(final_params['dims'][1]))
                for f in self.files:
                    self.file_data[f]['correction_params'] = final_params
                    self.file_data[f]['frame_shape'] = (w, h)
                self.recalculate_data()
                QMessageBox.information(self, "Success", "8-Pt Homography Calibration Applied to all files!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration Error: {e}")


    def draw_roi(self):
        if not HAS_CV2:
            QMessageBox.warning(self, "Error", "OpenCV required. Run: pip install opencv-python")
            return
            
        path = self.get_current_filepath()
        if not path or not path.endswith('.h5'):
            QMessageBox.information(self, "Info", "Select an .h5 file to draw Video ROI.")
            return

        v_path, _ = QFileDialog.getOpenFileName(self, "Select Video", os.path.dirname(path), "Video (*.mp4 *.avi *.mkv)")
        if v_path:
            try:
                cap = cv2.VideoCapture(v_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    d = ROIDialog(frame, self)
                    if d.exec_():
                        roi = d.get_path()
                        if roi:
                            for f in self.files:
                                self.file_data[f]['roi_path'] = roi
                                self.file_data[f]['frame_shape'] = (w, h)
                            
                            if self.raw_data is not None and 'X center_px' in self.raw_data.columns:
                                pts = np.vstack((self.raw_data['X center_px'], self.raw_data['Y center_px'])).T
                                self.raw_data['In zone'] = roi.contains_points(pts).astype(int)
                            
                            self.recalculate_data()
                            QMessageBox.information(self, "Success", "ROI saved and applied to ALL loaded files.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Video Error: {e}")

    def generate_shelter_roi(self, df, M, w, h, sx, sy):
        cx = (df['X center_filled'].max() + df['X center_filled'].min()) / 2
        cy = (df['Y center_filled'].max() + df['Y center_filled'].min()) / 2
        if M is not None: xm, ym = w, 0 
        else: xm, ym = cx + w/2, cy - h/2 
        return Path([(xm, ym), (xm - sx, ym), (xm, ym + sy)])

    def parse_file(self, filepath):
        try:
            self.selected_idx = None
            self.lbl_selected_point.setText("Selected: -- s")
            
            d = self.file_data[filepath]
            self.txt_mouse_id.setText(d['mouse_id'])
            self.txt_photostim.setText(d['photostim'])
            self.txt_post_stim.setText(str(d['post_stim']))
            self.txt_stim_start.setText(str(d.get('stim_start', 60.0)))
            
            if filepath.endswith('.xlsx'):
                self.btn_draw_roi.setEnabled(False)
                self.btn_calib.setEnabled(False)
                temp_df = pd.read_excel(filepath, nrows=50, header=None)
                idx_row = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
                df = pd.read_excel(filepath, skiprows=idx_row)
                df.columns = df.columns.str.strip()
                df.replace("-", np.nan, inplace=True)
                
                xc = next((c for c in df.columns if 'x center' in c.lower() and not c.endswith('_filled')), None)
                yc = next((c for c in df.columns if 'y center' in c.lower() and not c.endswith('_filled')), None)
                
                df['Interpolated'] = False
                if xc and yc:
                    df['Interpolated'] = df[xc].isna() | df[yc].isna()
                    df[f'{xc}_filled'] = pd.to_numeric(df[xc], errors='coerce').ffill()
                    df[f'{yc}_filled'] = pd.to_numeric(df[yc], errors='coerce').ffill()
                    
                if 'Distance moved' in df.columns:
                    df['Distance moved'] = pd.to_numeric(df['Distance moved'], errors='coerce')
                    if xc and yc:
                        cdist = np.sqrt(df[f'{xc}_filled'].diff()**2 + df[f'{yc}_filled'].diff()**2)
                        df['Distance moved_filled'] = df['Distance moved'].fillna(cdist).fillna(0.0)
                    else:
                        df['Distance moved_filled'] = df['Distance moved'].fillna(0.0)
                        
                if 'Velocity' in df.columns:
                    df['Velocity'] = pd.to_numeric(df['Velocity'], errors='coerce')
                    df['Velocity_filled'] = df['Velocity'].fillna(0.0)
                    df['Raw_Velocity'] = df['Velocity_filled']
                    
                if xc and yc:
                    try:
                        cw, ch = float(self.txt_chamber_w.text()), float(self.txt_chamber_h.text())
                        sx, sy = float(self.txt_shelter_x.text()), float(self.txt_shelter_y.text())
                        sp = self.generate_shelter_roi(df, None, cw, ch, sx, sy)
                        pts = np.vstack((df[f'{xc}_filled'], df[f'{yc}_filled'])).T
                        df['In zone'] = sp.contains_points(pts).astype(int)
                    except: df['In zone'] = 0

            elif filepath.endswith('.csv'):
                self.btn_draw_roi.setEnabled(False)
                self.btn_calib.setEnabled(False)
                df = pd.read_csv(filepath)
                
                # Reconstruct minimum required structure from imported CSV
                if 'Velocity' in df.columns and 'Velocity_filled' not in df.columns:
                    df['Velocity_filled'] = df['Velocity']
                if 'Velocity_filled' in df.columns and 'Raw_Velocity' not in df.columns:
                    df['Raw_Velocity'] = df['Velocity_filled']
                    
                if 'X center' in df.columns and 'X center_filled' not in df.columns:
                    df['X center_filled'] = df['X center']
                if 'Y center' in df.columns and 'Y center_filled' not in df.columns:
                    df['Y center_filled'] = df['Y center']
                    
            else:
                self.btn_draw_roi.setEnabled(True)
                self.btn_calib.setEnabled(True)
                df = self.parse_dlc(filepath)

            self.raw_data = df
            self.recalculate_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse file: {e}")

    def parse_dlc(self, filepath):
        df_h5 = pd.read_hdf(filepath)
        df_h5.replace([-1, -1.0], np.nan, inplace=True)
        
        bps = self.combo_bp.get_checked_items()
        expanded_bps = []
        for b in bps:
            if b == "Average (center+back)": expanded_bps.extend(["mouse_center", "mid_back"])
            elif b == "Average (shoulders)": expanded_bps.extend(["left_shoulder", "right_shoulder", "left_shouder"])
            elif b == "Average (midsides)": expanded_bps.extend(["left_midside", "right_midside"])
            else: expanded_bps.append(b)
            
        available_bps = df_h5.columns.get_level_values('bodyparts').unique()
        valid_bps = [p for p in expanded_bps if p in available_bps]
        
        if not valid_bps: valid_bps = [available_bps[0]]
            
        try: thresh = float(self.txt_likelihood.text())
        except: thresh = 0.3
        
        raw_x_parts, raw_y_parts, interp_masks = [], [], []
        
        for p in valid_bps:
            idx_x = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'x')
            idx_y = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'y')
            idx_l = (df_h5.columns.get_level_values('bodyparts') == p) & (df_h5.columns.get_level_values('coords') == 'likelihood')
            
            x = df_h5.loc[:, idx_x].iloc[:, 0].copy()
            y = df_h5.loc[:, idx_y].iloc[:, 0].copy()
            l = df_h5.loc[:, idx_l].iloc[:, 0].copy()
            
            mask = l < thresh
            x[mask] = np.nan; y[mask] = np.nan
            
            raw_x_parts.append(x); raw_y_parts.append(y); interp_masks.append(mask | x.isna()) 
            
        if not raw_x_parts:
            x_px = pd.Series(np.zeros(len(df_h5))); y_px = pd.Series(np.zeros(len(df_h5))); interp_masks = [pd.Series(np.ones(len(df_h5), dtype=bool))]
        else:
            x_px = pd.concat(raw_x_parts, axis=1).mean(axis=1).ffill()
            y_px = pd.concat(raw_y_parts, axis=1).mean(axis=1).ffill()
        
        try: fps = float(self.txt_fps.text())
        except: fps = 30.0
        
        df = pd.DataFrame()
        df['Frame'] = np.arange(len(df_h5))
        df['Trial time'] = df['Frame'] / fps
        df['X center_px'] = x_px
        df['Y center_px'] = y_px
        
        M = self.file_data[filepath].get('homography_M')
        if M is not None:
            pts = np.vstack((x_px, y_px)).T.reshape(-1, 1, 2).astype("float32")
            tf = cv2.perspectiveTransform(pts, M)
            df['X center'] = tf[:, 0, 0]
            df['Y center'] = tf[:, 0, 1]
        else:
            try: px_cm = float(self.txt_px_cm.text())
            except: px_cm = 10.0
            try: vid_h = float(self.txt_vid_h.text())
            except: vid_h = 480.0
            
            df['X center'] = x_px / px_cm
            df['Y center'] = (vid_h - y_px) / px_cm 
        
        df['X center_filled'] = df['X center']
        df['Y center_filled'] = df['Y center']
        
        dist = np.sqrt(df['X center_filled'].diff()**2 + df['Y center_filled'].diff()**2).fillna(0.0)
        df['Distance moved'] = dist
        df['Distance moved_filled'] = dist
        
        vel = dist * fps
        df['Velocity'] = vel
        df['Velocity_filled'] = vel
        df['Raw_Velocity'] = vel
        
        df['Interpolated'] = pd.concat(interp_masks, axis=1).any(axis=1)
        
        roi_path = self.file_data[filepath].get('roi_path')
        if roi_path:
            pts_px = np.vstack((df['X center_px'], df['Y center_px'])).T
            df['In zone'] = roi_path.contains_points(pts_px).astype(int)
        else:
            try:
                cw, ch = float(self.txt_chamber_w.text()), float(self.txt_chamber_h.text())
                sx, sy = float(self.txt_shelter_x.text()), float(self.txt_shelter_y.text())
                sp = self.generate_shelter_roi(df, M, cw, ch, sx, sy)
                pts_cm = np.vstack((df['X center_filled'], df['Y center_filled'])).T
                df['In zone'] = sp.contains_points(pts_cm).astype(int)
            except: df['In zone'] = 0
            
        return df

    def recalculate_data(self):
        if self.raw_data is None: return
        
        start = float(self.txt_stim_start.text())
        path = self.get_current_filepath()
        
        # Save active stim start globally into file data map
        if path: self.file_data[path]['stim_start'] = start
        
        if path and path.endswith('.h5') and self.sender() in [self.txt_fps, self.txt_px_cm, self.txt_vid_h, self.txt_likelihood, self.combo_bp.model()]:
            self.raw_data = self.parse_dlc(path)
            
        try:
            window = int(self.txt_rolling_avg.text()) if self.txt_rolling_avg.text().isdigit() else 1
            
            self.raw_data['Trial time'] = pd.to_numeric(self.raw_data['Trial time'], errors='coerce')
            self.raw_data['Rel_Time'] = self.raw_data['Trial time'] - start
            
            if 'Velocity_filled' in self.raw_data.columns and 'Raw_Velocity' in self.raw_data.columns:
                if window > 1:
                    self.raw_data['Velocity_filled'] = self.raw_data['Raw_Velocity'].rolling(window=window, center=True, min_periods=1).mean()
                else:
                    self.raw_data['Velocity_filled'] = self.raw_data['Raw_Velocity']
                    
            display_df = self.raw_data.drop(columns=['Raw_Velocity'], errors='ignore')
            self.table_view.setModel(PandasModel(display_df))
            self.update_plot()
        except Exception as e:
            print(e)

    # --- Interaction ---
    def on_metadata_changed(self):
        path = self.get_current_filepath()
        if path:
            self.file_data[path]['mouse_id'] = self.txt_mouse_id.text().strip()
            self.file_data[path]['photostim'] = self.txt_photostim.text().strip()
            self.update_plot()

    def on_post_stim_changed(self):
        path = self.get_current_filepath()
        if path:
            try:
                self.file_data[path]['post_stim'] = float(self.txt_post_stim.text())
                self.update_plot()
            except ValueError: pass

    def set_annotation(self, key):
        path = self.get_current_filepath()
        if path and self.selected_idx is not None: 
            self.file_data[path][key] = self.raw_data.at[self.selected_idx, 'Rel_Time']
            self.update_plot()

    def on_table_clicked(self, idx): 
        self.set_selected_point(idx.row())

    def on_velocity_click(self, event):
        if self.current_plot_df is not None and event.inaxes == self.ax and event.xdata: 
            self.set_selected_point((self.current_plot_df['Rel_Time'] - event.xdata).abs().idxmin())

    def on_trace_click(self, event):
        if self.current_plot_df is not None and event.inaxes == self.trace_ax and event.xdata:
            df = self.current_plot_df
            xc = next((c for c in df.columns if 'x center' in c.lower() and '_filled' in c.lower()), None)
            yc = next((c for c in df.columns if 'y center' in c.lower() and '_filled' in c.lower()), None)
            if xc and yc:
                self.set_selected_point(((df[xc]-event.xdata)**2 + (df[yc]-event.ydata)**2).idxmin())

    def set_selected_point(self, idx):
        if self.raw_data is None or idx not in self.raw_data.index: return
        self.selected_idx = idx
        self.table_view.selectRow(idx)
        self.lbl_selected_point.setText(f"Selected: {self.raw_data.at[idx, 'Rel_Time']:.2f} s")
        self._update_plot_markers()

    def _update_plot_markers(self):
        if self.selected_idx is None or self.raw_data is None: return
        idx = self.selected_idx
        if idx not in self.raw_data.index: return
        
        rel_t = self.raw_data.at[idx, 'Rel_Time']
        y_vel = self.raw_data.at[idx, 'Velocity_filled'] if 'Velocity_filled' in self.raw_data.columns else 0
        
        if self.vel_marker:
            try: self.vel_marker.remove()
            except: pass
        self.vel_marker, = self.ax.plot(rel_t, y_vel, marker='o', color='black', markersize=8, zorder=10)
        self.canvas.draw_idle()
        
        if self.trace_marker:
            try: self.trace_marker.remove()
            except: pass
            
        xc = next((c for c in self.raw_data.columns if 'x center' in c.lower() and '_filled' in c.lower()), None)
        yc = next((c for c in self.raw_data.columns if 'y center' in c.lower() and '_filled' in c.lower()), None)
        
        if xc and yc:
            x_val = self.raw_data.at[idx, xc]
            y_val = self.raw_data.at[idx, yc]
            if pd.notna(x_val) and pd.notna(y_val):
                self.trace_marker, = self.trace_ax.plot(x_val, y_val, marker='o', color='black', 
                                                        markeredgecolor='white', markersize=8, zorder=10)
                self.trace_canvas.draw_idle()

    def prev_point(self):
        if self.current_plot_df is not None and not self.current_plot_df.empty:
            pos = self.current_plot_df.index.get_loc(self.selected_idx) if self.selected_idx else 0
            if pos > 0: self.set_selected_point(self.current_plot_df.index[pos-1])

    def next_point(self):
        if self.current_plot_df is not None and not self.current_plot_df.empty:
            pos = self.current_plot_df.index.get_loc(self.selected_idx) if self.selected_idx else -1
            if pos < len(self.current_plot_df)-1: self.set_selected_point(self.current_plot_df.index[pos+1])

    # --- Plotting ---
    def update_plot(self):
        if self.raw_data is None: return
        path = self.get_current_filepath()
        d = self.file_data[path]
        
        self.vel_marker = None
        self.trace_marker = None
        
        pre = float(self.txt_pre_stim.text())
        post = d['post_stim']
        full_df = self.raw_data[(self.raw_data['Rel_Time'] >= pre) & (self.raw_data['Rel_Time'] <= post)].copy()
        
        if full_df.empty: return
        
        if d['shelter']:
            vis_df = full_df[full_df['Rel_Time'] <= d['shelter']].copy()
        else:
            vis_df = full_df.copy()
            
        self.ax.clear()
        self.ax.set_title(f"{d['mouse_id']} | {d['photostim']}")
        
        vel_col = 'Velocity_filled' if 'Velocity_filled' in vis_df.columns else 'Velocity'
        
        self.ax.plot(vis_df['Rel_Time'], vis_df[vel_col], color='#1976d2', label='Velocity')
        self.ax.axvline(0, color='black', linestyle='--', label=self.txt_stim_name.text())
        self.ax.set_xlim(pre, post)
        
        f_dur = "--"
        dist_v = "N/A"
        
        if d['fstart'] is not None and d['fend'] is not None:
            self.ax.axvspan(d['fstart'], d['fend'], color='red', alpha=0.2, label='Freezing')
            f_dur = f"{d['fend']-d['fstart']:.2f} s"
            
        if d['shelter']: 
            self.ax.axvline(d['shelter'], color='green', label='Shelter')
            
        dist_col = 'Distance moved_filled' if 'Distance moved_filled' in vis_df.columns else 'Distance moved'
        if d['fend'] and d['shelter'] and dist_col in vis_df.columns:
            sum_dist = vis_df[(vis_df['Rel_Time']>d['fend']) & (vis_df['Rel_Time']<=d['shelter'])][dist_col].sum()
            dist_v = f"{sum_dist:.2f}"
        
        self.lbl_fstart.setText(f"{d['fstart']:.2f}s" if d['fstart'] is not None else "--")
        self.lbl_fend.setText(f"{d['fend']:.2f}s" if d['fend'] is not None else "--")
        self.lbl_shelter.setText(f"{d['shelter']:.2f}s" if d['shelter'] is not None else "--")
        
        self.lbl_f_dur.setText(f"Dur: {f_dur}")
        self.lbl_dist.setText(f"Dist (E->S): {dist_v}")
        
        self.ax.legend(loc='upper right', fontsize='small')
        self.canvas.draw()
        
        cols_to_keep = [c for c in full_df.columns if any(k in c.lower() for k in ['frame', 'time', 'vel', 'dist', 'center', 'zone', 'interpolated'])]
        self.current_plot_df = full_df[cols_to_keep].copy()
        self.plot_export_df = vis_df[cols_to_keep].copy()
        
        self.update_trace_plot(vis_df, full_df, d)
        
        if self.selected_idx and self.selected_idx in full_df.index: 
            self._update_plot_markers()

    def update_trace_plot(self, vis, full, d):
        self.trace_ax.clear()
        self.trace_ax.set_title(f"{d['mouse_id']} | {d['photostim']}", fontsize=10)
            
        xc = next((c for c in vis.columns if 'x center' in c.lower() and '_filled' in c.lower()), None)
        yc = next((c for c in vis.columns if 'y center' in c.lower() and '_filled' in c.lower()), None)
        
        if xc and yc:
            self.trace_ax.plot(vis[vis['Rel_Time']<0][xc], vis[vis['Rel_Time']<0][yc], color='gray', alpha=0.5, label='Pre')
            self.trace_ax.plot(vis[vis['Rel_Time']>=0][xc], vis[vis['Rel_Time']>=0][yc], color='#1976d2', label='Post')
            
            on = vis[vis['Rel_Time']>=0]
            if not on.empty: 
                self.trace_ax.plot(on.iloc[0][xc], on.iloc[0][yc], 'o', color='red', markersize=6, zorder=5)
                
            fs, fe = d['fstart'], d['fend']
            if fs is not None and fe is not None:
                fr = vis[(vis['Rel_Time']>=fs) & (vis['Rel_Time']<=fe)]
                if not fr.empty: 
                    self.trace_ax.plot(fr[xc], fr[yc], color='#ff7f00', linewidth=3, zorder=4)
            
            try:
                cw = float(self.txt_chamber_w.text())
                ch = float(self.txt_chamber_h.text())
                sx = float(self.txt_shelter_x.text())
                sy = float(self.txt_shelter_y.text())
                M = d.get('homography_M')
                
                # Set trace limits based on mapping
                if M is not None:
                    self.trace_ax.set_xlim(0, cw)
                    self.trace_ax.set_ylim(0, ch)
                else:
                    cx = (full[xc].max() + full[xc].min()) / 2
                    cy = (full[yc].max() + full[yc].min()) / 2
                    xm, ym = cx + cw/2, cy - ch/2
                    self.trace_ax.set_xlim(cx - cw/2, xm)
                    self.trace_ax.set_ylim(ym, cy + ch/2)

                # Draw Custom ROI Mapping
                roi_path = d.get('roi_path')
                if roi_path and d.get('frame_shape'):
                    if M is not None:
                        pts = np.array(roi_path.vertices, dtype="float32").reshape(-1, 1, 2)
                        tf = cv2.perspectiveTransform(pts, M)[:, 0, :]
                        patch = Polygon(tf, closed=True, fill=False, edgecolor='purple', linestyle='--', linewidth=2, zorder=2)
                        self.trace_ax.add_patch(patch)
                    else:
                        w_px, h_px = d['frame_shape']
                        px_cm = float(self.txt_px_cm.text())
                        self.trace_ax.set_xlim(0, w_px / px_cm)
                        self.trace_ax.set_ylim(0, h_px / px_cm)
                        sc_verts = [(x/px_cm, (h_px - y)/px_cm) for x, y in roi_path.vertices]
                        patch = Polygon(sc_verts, closed=True, fill=False, edgecolor='purple', linestyle='--', linewidth=2, zorder=2)
                        self.trace_ax.add_patch(patch)
                else:
                    # Draw Shelter Green Triangle Mapping
                    if M is not None:
                        tri_pts = [[cw, 0], [cw - sx, 0], [cw, sy]]
                    else:
                        tri_pts = [[xm, ym], [xm - sx, ym], [xm, ym + sy]]
                    self.trace_ax.add_patch(Polygon(tri_pts, closed=True, facecolor='green', alpha=0.3, zorder=2))
            except: pass
            
        self.trace_ax.set_aspect('equal')
        self.trace_ax.set_xticks([])
        self.trace_ax.set_yticks([])
        self.trace_canvas.draw()

    # --- Copy/Save Helpers ---
    def copy_plot_to_clipboard(self, canvas):
        QApplication.clipboard().setPixmap(canvas.grab())
        self.statusBar().showMessage("Copied to clipboard!", 2000)

    def get_base_filename(self):
        m_id = self.txt_mouse_id.text().strip().replace(" ", "_")
        p_stim = self.txt_photostim.text().strip().replace(" ", "_")
        return f"{m_id}_{p_stim}"

    def save_plot_as_png(self, figure, plot_type):
        if not self.output_dir: return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.get_base_filename()}_{plot_type}_{ts}.png"
        p = os.path.join(self.output_dir, filename)
        figure.savefig(p, dpi=300, bbox_inches='tight')
        self.statusBar().showMessage(f"Saved: {filename}", 4000)

    # --- Exports ---
    def get_metrics_for_file(self, filepath):
        try:
            d = self.file_data.get(filepath)
            if not d or d['fstart'] is None or d['fend'] is None: return None
            
            if filepath.endswith('.xlsx'):
                temp_df = pd.read_excel(filepath, nrows=50, header=None)
                idx = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
                df = pd.read_excel(filepath, skiprows=idx)
                df.columns = df.columns.str.strip()
                df.replace("-", np.nan, inplace=True)
                
                xc = next((c for c in df.columns if 'x center' in c.lower()), None)
                yc = next((c for c in df.columns if 'y center' in c.lower()), None)
                if xc and yc:
                    df[f'{xc}_filled'] = pd.to_numeric(df[xc], errors='coerce').ffill()
                    df[f'{yc}_filled'] = pd.to_numeric(df[yc], errors='coerce').ffill()
                if 'Distance moved' in df.columns:
                    df['Distance moved'] = pd.to_numeric(df['Distance moved'], errors='coerce')
                    if xc and yc:
                        df['Distance moved_filled'] = df['Distance moved'].fillna(np.sqrt(df[f'{xc}_filled'].diff()**2 + df[f'{yc}_filled'].diff()**2)).fillna(0.0)
                    else:
                        df['Distance moved_filled'] = df['Distance moved'].fillna(0.0)
                if 'Velocity' in df.columns:
                    df['Velocity_filled'] = pd.to_numeric(df['Velocity'], errors='coerce').fillna(0.0)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = self.parse_dlc(filepath)

            stim_start = float(self.txt_stim_start.text())
            df['Trial time'] = pd.to_numeric(df['Trial time'], errors='coerce')
            df['Rel_Time'] = df['Trial time'] - stim_start
            
            vel_col = 'Velocity_filled' if 'Velocity_filled' in df.columns else 'Velocity'
            max_speed = "N/A"
            if vel_col in df.columns:
                post_df = df[df['Rel_Time'] >= 0]
                if not post_df.empty: max_speed = post_df[vel_col].max()
            
            dist_col = 'Distance moved_filled' if 'Distance moved_filled' in df.columns else 'Distance moved'
            dist = "N/A"
            if d['shelter'] is not None and dist_col in df.columns:
                mask = (df['Rel_Time'] > d['fend']) & (df['Rel_Time'] <= d['shelter'])
                dist = df.loc[mask, dist_col].sum()
                
            roi_time = "N/A"
            if 'In zone' in df.columns and df['In zone'].sum() > 0:
                in_zone_post = df[(df['Rel_Time'] > 0) & (df['In zone'] == 1)]
                if not in_zone_post.empty: roi_time = in_zone_post['Rel_Time'].iloc[0]

            return {
                "Filename": os.path.basename(filepath),
                "Mouse_ID": d['mouse_id'],
                "PhotoStim": d['photostim'],
                "Stim_Start_Time": stim_start,
                "Post_Range": d['post_stim'],
                "Manual_Freeze_Start": d['fstart'],
                "Manual_Freeze_End": d['fend'],
                "Manual_Shelter_Time": d['shelter'] if d['shelter'] is not None else "N/A",
                "First_In_ROI_Time": roi_time,
                "Calculated_Freeze_Duration": d['fend'] - d['fstart'],
                "Distance_To_Shelter": dist,
                "Max_Speed": max_speed,
                "Analysis_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Export Error: {e}")
            return None

    def master_export_current(self):
        path = self.get_current_filepath()
        if not path or not self.output_dir: return
        d = self.file_data[path]
        
        if d['fend'] is None: 
            QMessageBox.warning(self, "Incomplete", "Define Freezing End first.")
            return
            
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bn = self.get_base_filename()
        
        self.figure.savefig(os.path.join(self.output_dir, f"{bn}_VelocityPlot_{ts}.png"), dpi=200, bbox_inches='tight')
        self.trace_fig.savefig(os.path.join(self.output_dir, f"{bn}_TracePlot_{ts}.png"), dpi=200, bbox_inches='tight')
        
        if self.plot_export_df is not None:
            self.plot_export_df.to_csv(os.path.join(self.output_dir, f"{bn}_PlotData_{ts}.csv"), index=False)
            
        m = self.get_metrics_for_file(path)
        if m:
            df_new = pd.DataFrame([m])
            csv_p = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
            if os.path.exists(csv_p):
                df_old = pd.read_csv(csv_p)
                keys = ['Filename', 'Mouse_ID', 'PhotoStim']
                if all(k in df_old.columns for k in keys):
                    df_old[keys] = df_old[keys].astype(str)
                    df_new[keys] = df_new[keys].astype(str)
                    df_old = df_old[~df_old.set_index(keys).index.isin(df_new.set_index(keys).index)]
                    pd.concat([df_old, df_new], ignore_index=True).to_csv(csv_p, index=False)
                else:
                    pd.concat([df_old, df_new], ignore_index=True).to_csv(csv_p, index=False)
            else:
                df_new.to_csv(csv_p, index=False)
            self.statusBar().showMessage("Master Export Complete", 4000)

    def export_results(self):
        if not self.files or not self.output_dir: return
        all_m = []
        for f in self.files:
            m = self.get_metrics_for_file(f)
            if m: all_m.append(m)
            
        if not all_m: 
            QMessageBox.information(self, "Export Skipped", "No files with freezing times.")
            return
            
        df_new = pd.DataFrame(all_m)
        csv_p = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
        
        if os.path.exists(csv_p):
            df_old = pd.read_csv(csv_p)
            keys = ['Filename','Mouse_ID','PhotoStim']
            if all(k in df_old.columns for k in keys):
                df_old[keys] = df_old[keys].astype(str)
                df_new[keys] = df_new[keys].astype(str)
                
                overlap = pd.merge(df_old, df_new[keys], on=keys, how='inner')
                if not overlap.empty:
                    reply = QMessageBox.question(self, "Overwrite?", 
                                                 f"{len(overlap)} records exist. Overwrite?",
                                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                    if reply == QMessageBox.Cancel: return
                    if reply == QMessageBox.Yes:
                        df_old = df_old[~df_old.set_index(keys).index.isin(df_new.set_index(keys).index)]
                        
                df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(csv_p, index=False)
        QMessageBox.information(self, "Success", "Summary Export Complete!")

    def export_plot_data(self):
        if self.plot_export_df is not None and self.output_dir:
            bn = self.get_base_filename()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(self.output_dir, f"{bn}_PlotData_{ts}.csv")
            self.plot_export_df.to_csv(p, index=False)
            self.statusBar().showMessage(f"Saved Data: {p}", 3000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = EthoVisionAnalyzer()
    win.show()
    sys.exit(app.exec_())
