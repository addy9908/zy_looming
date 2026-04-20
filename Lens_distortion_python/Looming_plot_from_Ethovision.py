# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:57:59 2026
1. plot velocity vs time from raw Ethovision output files (Excel)
2. calculate and mask the back to shelter time, freezing time
3. save the results, raw plot data, and plots
@author: yez4
"""

import sys
import os
import datetime
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableView, QLabel, QLineEdit, 
                             QFormLayout, QComboBox, QSplitter, QMessageBox, QAbstractItemView)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QColor, QBrush

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
            col_name = self._df.columns[index.column()]
            if str(col_name).endswith('_filled'):
                orig_col = str(col_name).replace('_filled', '')
                if orig_col in self._df.columns:
                    val = self._df.iloc[index.row()][orig_col]
                    if pd.isna(val) or val == "-":
                        return QBrush(QColor(255, 255, 200)) 
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._df.columns[col]
        return None

class EthoVisionAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Enterprise - EthoVision Analyzer (Final)")
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
        
        # --- LEFT PANEL: Global Params & Trace ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(330)
        
        self.btn_load = QPushButton("📂 Load Excel Files")
        self.btn_load.clicked.connect(self.load_files)
        left_layout.addWidget(self.btn_load)
        
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
        
        for txt in [self.txt_stim_start, self.txt_pre_stim, self.txt_rolling_avg, 
                    self.txt_chamber_w, self.txt_chamber_h, self.txt_shelter_x, self.txt_shelter_y]:
            txt.editingFinished.connect(self.recalculate_data)

        param_form.addRow("Stim Start (s):", self.txt_stim_start)
        param_form.addRow("Pre-Range (s):", self.txt_pre_stim)
        param_form.addRow("Legend Name:", self.txt_stim_name)
        param_form.addRow("Rolling Window:", self.txt_rolling_avg)
        param_form.addRow("Chamber Width (cm):", self.txt_chamber_w)
        param_form.addRow("Chamber Height (cm):", self.txt_chamber_h)
        param_form.addRow("Shelter Triangle X:", self.txt_shelter_x)
        param_form.addRow("Shelter Triangle Y:", self.txt_shelter_y)
        left_layout.addLayout(param_form)
        
        left_layout.addWidget(QLabel("\n<b>Export Tools</b>"))
        self.btn_master_export = QPushButton("🚀 Master Export (Current File)")
        self.btn_master_export.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 8px;")
        self.btn_master_export.clicked.connect(self.master_export_current)
        left_layout.addWidget(self.btn_master_export)
        
        self.btn_export_csv = QPushButton("Export Results Only (CSV)")
        self.btn_export_csv.clicked.connect(self.export_results)
        left_layout.addWidget(self.btn_export_csv)
        
        self.btn_export_plot_data = QPushButton("Export Current Plot Data")
        self.btn_export_plot_data.clicked.connect(self.export_plot_data)
        left_layout.addWidget(self.btn_export_plot_data)
        
        left_layout.addWidget(QLabel("\n<b>Moving Trace View</b>"))
        self.trace_fig = Figure(figsize=(3, 3))
        self.trace_ax = self.trace_fig.add_subplot(111)
        self.trace_fig.tight_layout(pad=0.5)
        self.trace_canvas = FigureCanvas(self.trace_fig)
        self.trace_canvas.setMinimumHeight(280)
        self.trace_canvas.mpl_connect('button_press_event', self.on_trace_click)
        left_layout.addWidget(self.trace_canvas)
        
        trace_btn_layout = QHBoxLayout()
        self.btn_copy_trace = QPushButton("Copy Trace")
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
        nav_layout.addWidget(QLabel("<b>File Selection:</b>"))
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
        meta_layout.addWidget(QLabel("<b>Post-Range (s):</b>"))
        meta_layout.addWidget(self.txt_post_stim)
        top_right_layout.addLayout(meta_layout)
        
        nav_ann_layout = QHBoxLayout()
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
        nav_ann_layout.addLayout(nav_col)
        
        ann_col = QVBoxLayout()
        ann_col.addWidget(QLabel("<b>Manual Annotation:</b>"))
        row1 = QHBoxLayout()
        self.btn_fstart = QPushButton("Set Freezing Start")
        self.lbl_fstart = QLabel("0.00s")
        self.btn_fstart.clicked.connect(lambda: self.set_annotation('fstart'))
        row1.addWidget(self.btn_fstart); row1.addWidget(self.lbl_fstart)
        
        row2 = QHBoxLayout()
        self.btn_fend = QPushButton("Set Freezing End")
        self.lbl_fend = QLabel("--")
        self.btn_fend.clicked.connect(lambda: self.set_annotation('fend'))
        row2.addWidget(self.btn_fend); row2.addWidget(self.lbl_fend)
        
        row3 = QHBoxLayout()
        self.btn_shelter = QPushButton("Set Back to Shelter")
        self.lbl_shelter = QLabel("--")
        self.btn_shelter.clicked.connect(lambda: self.set_annotation('shelter'))
        row3.addWidget(self.btn_shelter); row3.addWidget(self.lbl_shelter)
        
        ann_col.addLayout(row1)
        ann_col.addLayout(row2)
        ann_col.addLayout(row3)
        nav_ann_layout.addLayout(ann_col)
        
        calc_col = QVBoxLayout()
        calc_col.addWidget(QLabel("<b>Metrics:</b>"))
        self.lbl_f_dur = QLabel("Freezing Dur: --")
        self.lbl_time = QLabel("Time (E->S): N/A")
        
        self.lbl_dist = QLabel("Dist (E->S): N/A")
        self.lbl_f_dur.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.lbl_time.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.lbl_dist.setStyleSheet("color: #2e7d32; font-weight: bold;")
        calc_col.addWidget(self.lbl_f_dur)
        calc_col.addWidget(self.lbl_time)
        calc_col.addWidget(self.lbl_dist)
        nav_ann_layout.addLayout(calc_col)
        
        top_right_layout.addLayout(nav_ann_layout)
        
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

    # --- Data Logic ---
    def _read_meta(self, filepath):
        try:
            temp_df = pd.read_excel(filepath, nrows=50, header=None)
            header_row_idx = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
            meta_df = temp_df.iloc[:header_row_idx]
            m_id, p_stim = "N/A", "N/A"
            for _, row in meta_df.iterrows():
                if pd.notna(row[0]) and pd.notna(row[1]):
                    k, v = str(row[0]).lower(), str(row[1])
                    if "mouse" in k: m_id = v
                    elif "stim" in k: p_stim = v
            return str(m_id).strip(), str(p_stim).strip()
        except: return "N/A", "N/A"

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Ethovision", "", "Excel Files (*.xlsx)")
        if files:
            files.sort(key=lambda x: os.path.basename(x).lower())
            self.files = files
            self.output_dir = os.path.join(os.path.dirname(self.files[0]), "Output")
            if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
            
            csv_path = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
            df_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
            self.file_data = {}

            for f in self.files:
                m_id, p_stim = self._read_meta(f)
                self.file_data[f] = {'mouse_id': m_id, 'photostim': p_stim, 'fstart': 0.0, 'fend': None, 'shelter': None, 'post_stim': 10.0}
                
                if df_csv is not None and not df_csv.empty and all(c in df_csv.columns for c in ['Filename','Mouse_ID','PhotoStim']):
                    match = df_csv[(df_csv['Filename']==os.path.basename(f)) & (df_csv['Mouse_ID'].astype(str)==m_id) & (df_csv['PhotoStim'].astype(str)==p_stim)]
                    if not match.empty:
                        rec = match.iloc[-1]
                        self.file_data[f]['fstart'] = float(rec['Manual_Freeze_Start']) if pd.notna(rec['Manual_Freeze_Start']) and rec['Manual_Freeze_Start'] != "N/A" else 0.0
                        self.file_data[f]['fend'] = float(rec['Manual_Freeze_End']) if pd.notna(rec['Manual_Freeze_End']) and rec['Manual_Freeze_End'] != "N/A" else None
                        self.file_data[f]['shelter'] = float(rec['Manual_Shelter_Time']) if pd.notna(rec['Manual_Shelter_Time']) and rec['Manual_Shelter_Time'] != "N/A" else None
                        self.file_data[f]['post_stim'] = float(rec['Post_Range']) if pd.notna(rec['Post_Range']) and rec['Post_Range'] != "N/A" else 10.0

            self.file_combo.blockSignals(True)
            self.file_combo.clear()
            self.file_combo.addItems([os.path.basename(f) for f in self.files])
            self.file_combo.blockSignals(False)
            self.current_file_idx = 0
            self.file_combo.setCurrentIndex(0)
            self.parse_ethovision_file(self.files[0])

    def parse_ethovision_file(self, filepath):
        try:
            self.selected_idx = None
            self.lbl_selected_point.setText("Selected: -- s")
            d = self.file_data[filepath]
            self.txt_mouse_id.setText(d['mouse_id'])
            self.txt_photostim.setText(d['photostim'])
            self.txt_post_stim.setText(str(d['post_stim']))
            
            temp_df = pd.read_excel(filepath, nrows=50, header=None)
            header_row_idx = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
            df = pd.read_excel(filepath, skiprows=header_row_idx)
            df.columns = df.columns.str.strip()
            
            df.replace("-", np.nan, inplace=True)
            xc = next((c for c in df.columns if 'x center' in c.lower() and not c.endswith('_filled')), None)
            yc = next((c for c in df.columns if 'y center' in c.lower() and not c.endswith('_filled')), None)
            
            if xc and yc:
                df[f'{xc}_filled'] = pd.to_numeric(df[xc], errors='coerce').ffill()
                df[f'{yc}_filled'] = pd.to_numeric(df[yc], errors='coerce').ffill()
            
            if 'Distance moved' in df.columns:
                df['Distance moved'] = pd.to_numeric(df['Distance moved'], errors='coerce')
                if xc and yc:
                    calc_dist = np.sqrt(df[f'{xc}_filled'].diff()**2 + df[f'{yc}_filled'].diff()**2)
                    df['Distance moved_filled'] = df['Distance moved'].fillna(calc_dist).fillna(0.0)
                else:
                    df['Distance moved_filled'] = df['Distance moved'].fillna(0.0)
            
            if 'Velocity' in df.columns:
                df['Velocity'] = pd.to_numeric(df['Velocity'], errors='coerce')
                df['Velocity_filled'] = df['Velocity'].fillna(0.0)
                df['Raw_Velocity'] = df['Velocity_filled']

            self.raw_data = df
            self.recalculate_data()
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")

    def recalculate_data(self):
        if self.raw_data is None: return
        try:
            start = float(self.txt_stim_start.text())
            window = int(self.txt_rolling_avg.text()) if self.txt_rolling_avg.text().isdigit() else 1
            
            self.raw_data['Trial time'] = pd.to_numeric(self.raw_data['Trial time'], errors='coerce')
            self.raw_data['Rel_Time'] = self.raw_data['Trial time'] - start
            
            if 'Velocity_filled' in self.raw_data.columns and 'Raw_Velocity' in self.raw_data.columns:
                if window > 1:
                    self.raw_data['Velocity_filled'] = self.raw_data['Raw_Velocity'].rolling(window=window, center=True, min_periods=1).mean()
                else:
                    self.raw_data['Velocity_filled'] = self.raw_data['Raw_Velocity']
                    
            self.table_view.setModel(PandasModel(self.raw_data.drop(columns=['Raw_Velocity'], errors='ignore')))
            self.update_plot()
        except Exception as e: print(e)

    # --- Interaction ---
    def on_metadata_changed(self):
        path = self.files[self.current_file_idx]
        self.file_data[path]['mouse_id'] = self.txt_mouse_id.text().strip()
        self.file_data[path]['photostim'] = self.txt_photostim.text().strip()
        self.update_plot()

    def on_post_stim_changed(self):
        path = self.files[self.current_file_idx]
        try: 
            self.file_data[path]['post_stim'] = float(self.txt_post_stim.text())
            self.update_plot()
        except: pass

    def set_annotation(self, key):
        path = self.files[self.current_file_idx]
        if self.selected_idx is not None: 
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
        path = self.files[self.current_file_idx]
        d = self.file_data[path]
        
        self.vel_marker = None
        self.trace_marker = None
        
        pre, post = float(self.txt_pre_stim.text()), d['post_stim']
        full_df = self.raw_data[(self.raw_data['Rel_Time']>=pre) & (self.raw_data['Rel_Time']<=post)].copy()
        
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
        time_v = "N/A"
        
        if d['fstart'] is not None and d['fend'] is not None:
            self.ax.axvspan(d['fstart'], d['fend'], color='red', alpha=0.2, label='Freezing')
            f_dur = f"{d['fend']-d['fstart']:.2f} s"
            
        if d['shelter']: 
            self.ax.axvline(d['shelter'], color='green', label='Shelter')
            
        dist_col = 'Distance moved_filled' if 'Distance moved_filled' in vis_df.columns else 'Distance moved'
        if d['fend'] and d['shelter'] and dist_col in vis_df.columns:
            sum_dist = vis_df[(vis_df['Rel_Time']>d['fend']) & (vis_df['Rel_Time']<=d['shelter'])][dist_col].sum()
            sum_time = d['shelter'] - d['fend']
            dist_v = f"{sum_dist:.2f}"
            time_v = f"{sum_time:.2f}"
        
        self.lbl_fstart.setText(f"{d['fstart']:.2f}s" if d['fstart'] is not None else "--")
        self.lbl_fend.setText(f"{d['fend']:.2f}s" if d['fend'] is not None else "--")
        self.lbl_shelter.setText(f"{d['shelter']:.2f}s" if d['shelter'] is not None else "--")
        
        self.lbl_f_dur.setText(f"Freezing Dur: {f_dur}")
        self.lbl_time.setText(f"Dist (E->S): {time_v}")
        self.lbl_dist.setText(f"Dist (E->S): {dist_v}")
        
        self.ax.legend(loc='upper right', fontsize='small')
        self.canvas.draw()
        
        cols_to_keep = [c for c in full_df.columns if any(k in c.lower() for k in ['time', 'vel', 'dist', 'center'])]
        self.current_plot_df = full_df[cols_to_keep].copy()
        self.plot_export_df = vis_df[cols_to_keep].copy()
        
        self.update_trace_plot(vis_df, full_df, d['fstart'], d['fend'])
        
        if self.selected_idx and self.selected_idx in full_df.index: 
            self._update_plot_markers()

    def update_trace_plot(self, vis, full, fs, fe):
        self.trace_ax.clear()
        path = self.files[self.current_file_idx]
        d = self.file_data[path]
        
        self.trace_ax.set_title(f"{d['mouse_id']} | {d['photostim']}", fontsize=9)
        
        xc = next((c for c in vis.columns if 'x center' in c.lower() and '_filled' in c.lower()), None)
        yc = next((c for c in vis.columns if 'y center' in c.lower() and '_filled' in c.lower()), None)
        
        if xc and yc:
            self.trace_ax.plot(vis[vis['Rel_Time']<0][xc], vis[vis['Rel_Time']<0][yc], color='gray', alpha=0.5, label='Pre')
            self.trace_ax.plot(vis[vis['Rel_Time']>=0][xc], vis[vis['Rel_Time']>=0][yc], color='#1976d2', label='Post')
            
            on = vis[vis['Rel_Time']>=0]
            if not on.empty: 
                self.trace_ax.plot(on.iloc[0][xc], on.iloc[0][yc], 'o', color='red', markersize=6)
                
            if fs is not None and fe is not None:
                fr = vis[(vis['Rel_Time']>=fs) & (vis['Rel_Time']<=fe)]
                if not fr.empty: 
                    self.trace_ax.plot(fr[xc], fr[yc], color='#ff7f00', linewidth=3)
                    
            try:
                w, h = float(self.txt_chamber_w.text()), float(self.txt_chamber_h.text())
                sx, sy = float(self.txt_shelter_x.text()), float(self.txt_shelter_y.text())
                
                cx, cy = (full[xc].max()+full[xc].min())/2, (full[yc].max()+full[yc].min())/2
                xm, ym = cx+w/2, cy-h/2
                
                self.trace_ax.set_xlim(cx-w/2, xm)
                self.trace_ax.set_ylim(ym, cy+h/2)
                
                self.trace_ax.add_patch(Polygon([[xm, ym], [xm-sx, ym], [xm, ym+sy]], closed=True, facecolor='green', alpha=0.3))
            except: pass
            
        self.trace_ax.set_aspect('equal')
        self.trace_ax.set_xticks([])
        self.trace_ax.set_yticks([])
        self.trace_canvas.draw()

    # --- Batch Processor Helper ---
    def process_file_metrics(self, filepath):
        """Reads a file completely in the background to calculate metrics for export"""
        try:
            d = self.file_data.get(filepath)
            if not d: return None
            
            fstart = d['fstart']
            fend = d['fend']
            shelter = d['shelter']
            post_range = d['post_stim']
            mouse_id = d['mouse_id']
            photo_stim = d['photostim']
            
            if fstart is None or fend is None:
                return None
            
            temp_df = pd.read_excel(filepath, nrows=50, header=None)
            header_row_idx = temp_df[temp_df.eq("Trial time").any(axis=1)].index[0]
            df = pd.read_excel(filepath, skiprows=header_row_idx)
            df.columns = df.columns.str.strip()
            df.replace("-", np.nan, inplace=True)
            
            stim_start = float(self.txt_stim_start.text())
            df['Trial time'] = pd.to_numeric(df['Trial time'], errors='coerce')
            df['Rel_Time'] = df['Trial time'] - stim_start
            
            xc = next((c for c in df.columns if 'x center' in c.lower()), None)
            yc = next((c for c in df.columns if 'y center' in c.lower()), None)
            if xc and yc:
                df[f'{xc}_filled'] = pd.to_numeric(df[xc], errors='coerce').ffill()
                df[f'{yc}_filled'] = pd.to_numeric(df[yc], errors='coerce').ffill()
            
            dist_col = 'Distance moved'
            if dist_col in df.columns:
                df[dist_col] = pd.to_numeric(df[dist_col], errors='coerce')
                if xc and yc:
                    calc_dist = np.sqrt(df[f'{xc}_filled'].diff()**2 + df[f'{yc}_filled'].diff()**2)
                    df[f'{dist_col}_filled'] = df[dist_col].fillna(calc_dist).fillna(0.0)
                else:
                    df[f'{dist_col}_filled'] = df[dist_col].fillna(0.0)
            
            vel_col = 'Velocity'
            max_speed = "N/A"
            if vel_col in df.columns:
                df[vel_col] = pd.to_numeric(df[vel_col], errors='coerce')
                df[f'{vel_col}_filled'] = df[vel_col].fillna(0.0)
                # Only check max speed in post stim
                post_df = df[df['Rel_Time'] >= 0]
                if not post_df.empty:
                    max_speed = post_df[f'{vel_col}_filled'].max()
            
            f_dur = fend - fstart
            dist = "N/A"
            if shelter is not None and f'{dist_col}_filled' in df.columns:
                mask = (df['Rel_Time'] > fend) & (df['Rel_Time'] <= shelter)
                dist = df.loc[mask, f'{dist_col}_filled'].sum()
                
            return {
                "Filename": os.path.basename(filepath),
                "Mouse_ID": mouse_id,
                "PhotoStim": photo_stim,
                "Stim_Start_Time": stim_start,
                "Post_Range": post_range,
                "Manual_Freeze_Start": fstart,
                "Manual_Freeze_End": fend,
                "Calculated_Freeze_Duration": f_dur,
                "Manual_Shelter_Time": shelter if shelter is not None else "N/A",
                "Time_to_Shelter": shelter-fend if shelter is not None else "N/A",
                "Distance_To_Shelter": dist,
                "Max_Speed": max_speed,
                "Analysis_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Failed processing {filepath}: {e}")
            return None

    # --- Exports ---
    def master_export_current(self):
        path = self.files[self.current_file_idx]
        d = self.file_data[path]
        
        if d['fend'] is None: 
            QMessageBox.warning(self, "Incomplete", "Define Freezing End first.")
            return
            
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bn = f"{d['mouse_id']}_{d['photostim']}".replace(" ", "_")
        
        self.figure.savefig(os.path.join(self.output_dir, f"{bn}_Velocity_{ts}.png"), dpi=200, bbox_inches='tight')
        self.trace_fig.savefig(os.path.join(self.output_dir, f"{bn}_Trace_{ts}.png"), dpi=200, bbox_inches='tight')
        
        if self.plot_export_df is not None: 
            self.plot_export_df.to_csv(os.path.join(self.output_dir, f"{bn}_PlotData_{ts}.csv"), index=False)
            
        # Append to batch CSV
        m = self.process_file_metrics(path)
        if m:
            df_new = pd.DataFrame([m])
            csv_p = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
            if os.path.exists(csv_p):
                df_old = pd.read_csv(csv_p)
                keys = ['Filename','Mouse_ID','PhotoStim']
                if all(k in df_old.columns for k in keys):
                    df_old[keys] = df_old[keys].astype(str)
                    df_new[keys] = df_new[keys].astype(str)
                    df_old = df_old[~df_old.set_index(keys).index.isin(df_new.set_index(keys).index)]
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_csv(csv_p, index=False)
            
        self.statusBar().showMessage("Master Export Complete (Current File)", 3000)

    def export_results(self):
        all_m = []
        for f in self.files:
            m = self.process_file_metrics(f)
            if m: all_m.append(m)
            
        if not all_m: 
            QMessageBox.information(self, "Export Skipped", "No files had calculated freezing times. Nothing exported.")
            return
            
        df_new = pd.DataFrame(all_m)
        csv_p = os.path.join(self.output_dir, "EthoVision_Batch_Results.csv")
        
        if os.path.exists(csv_p):
            df_old = pd.read_csv(csv_p)
            keys = ['Filename','Mouse_ID','PhotoStim']
            if all(k in df_old.columns for k in keys):
                df_old[keys] = df_old[keys].astype(str)
                df_new[keys] = df_new[keys].astype(str)
                
                # Check for overlap to prompt overwrite
                overlap = pd.merge(df_old, df_new[keys], on=keys, how='inner')
                if not overlap.empty:
                    reply = QMessageBox.question(self, "Overwrite?", 
                                                 f"Records for {len(overlap)} file(s) exist.\nOverwrite?",
                                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                    if reply == QMessageBox.Cancel: return
                    if reply == QMessageBox.Yes:
                        df_old = df_old[~df_old.set_index(keys).index.isin(df_new.set_index(keys).index)]
                        
                df_new = pd.concat([df_old, df_new], ignore_index=True)
                
        df_new.to_csv(csv_p, index=False)
        QMessageBox.information(self, "Success", f"Results saved to {csv_p}")

    def export_plot_data(self):
        if self.plot_export_df is not None:
            bn = self.get_base_filename()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(self.output_dir, f"{bn}_PlotData_{ts}.csv")
            self.plot_export_df.to_csv(p, index=False)
            self.statusBar().showMessage(f"Saved Data: {p}", 3000)

    def on_file_selected(self, idx):
        if idx >= 0: self.current_file_idx = idx; self.parse_ethovision_file(self.files[idx])
    def prev_file(self): 
        if self.current_file_idx > 0: self.file_combo.setCurrentIndex(self.current_file_idx - 1)
    def next_file(self): 
        if self.current_file_idx < len(self.files) - 1: self.file_combo.setCurrentIndex(self.current_file_idx + 1)
    def copy_plot_to_clipboard(self, canvas): QApplication.clipboard().setPixmap(canvas.grab())
    def save_plot_as_png(self, fig, name): 
        p = os.path.join(self.output_dir, f"{self.get_base_filename()}_{name}_{datetime.datetime.now().strftime('%H%M%S')}.png")
        fig.savefig(p, dpi=200); self.statusBar().showMessage(f"Saved: {p}", 3000)
    def get_base_filename(self): return f"{self.txt_mouse_id.text()}_{self.txt_photostim.text()}".replace(" ","_")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = EthoVisionAnalyzer()
    win.show()
    sys.exit(app.exec_())

