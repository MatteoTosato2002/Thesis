import sys
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import pyqtgraph as pg
from pyqtgraph import exporters

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTabWidget,
    QScrollArea, QMessageBox, QListWidget, QDialog, QDialogButtonBox,
    QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox, QFrame, QSizePolicy,
    QGridLayout, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem,
    QSlider, QToolButton, QRadioButton, QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QAction, QIcon, QPainter, QPolygonF
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
import subprocess

# This is the file with all the data-analysis code (reading logs, merging CSVs, etc.)
# The GUI below only calls into it and displays what comes back.
import pipeline_deff2_mod as pipeline

# A normal pyqtgraph plot zooms in/out when you scroll over it, which is annoying
# when the plot is embedded in a scrollable form. This subclass just ignores the wheel.
class NoWheelPlotWidget(pg.PlotWidget):
    def wheelEvent(self, event):
        # Tell Qt "I did not use this scroll event", so the page scrolls instead of the plot zooming.
        event.ignore()

# Small popup window: user types in a measured length (pixels) and the video second it belongs to.
class AddMeasurementDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Measurement")
        layout = QFormLayout(self)
        self.px_input = QDoubleSpinBox()
        self.px_input.setRange(0, 100000)
        self.px_input.setDecimals(3)
        self.sec_input = QSpinBox()
        self.sec_input.setRange(0, 100000)
        layout.addRow("Pitrinu Length (px):", self.px_input)
        layout.addRow("Video Second:", self.sec_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    # Read back what the user typed, as plain numbers.
    def get_data(self): return self.px_input.value(), self.sec_input.value()

# Lets the user pick a CSV file (instead of typing measurements one by one) and previews
# how many valid rows were found before confirming.
class LoadMeasurementCSVDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Measurements from CSV")
        self.measurements = []
        layout = QVBoxLayout(self)

        lbl = QLabel("Select a CSV file with 'Length_px' and 'S' columns")
        layout.addWidget(lbl)

        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        self.lbl_preview = QLabel("No file selected")
        self.lbl_preview.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # Opens the native "choose a file" window and, once picked, loads a preview of it.
    def browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV (*.csv)")
        if f:
            self.file_input.setText(f)
            self.load_preview(f)

    # Reads the chosen CSV and turns valid rows into measurement dicts, showing a summary message.
    def load_preview(self, csv_path):
        try:
            df = pd.read_csv(csv_path)

            if 'Length_px' not in df.columns or 'S' not in df.columns:
                self.lbl_preview.setText(f"❌ CSV must contain 'Length_px' and 'S' columns")
                self.lbl_preview.setStyleSheet("color: red;")
                self.measurements = []
                return

            self.measurements = []
            for _, row in df.iterrows():
                try:
                    px = float(row['Length_px'])
                    sec_value = str(row['S']).strip()

                    # Extract number from strings like "156.png" or "image_123.jpg"
                    # First, remove extension (everything after the last dot)
                    if '.' in sec_value:
                        sec_value = sec_value.rsplit('.', 1)[0]

                    # Extract all consecutive digits from the string
                    import re
                    numbers = re.findall(r'\d+', sec_value)
                    if numbers:
                        # Take the last sequence of digits (usually the frame/image number)
                        sec = int(numbers[-1])
                    else:
                        # No digits found, skip this row
                        continue

                    self.measurements.append({'len_pitrinu_px': px, 'sec_vid': sec})
                except (ValueError, TypeError):
                    continue

            self.lbl_preview.setText(f"✓ Loaded {len(self.measurements)} measurements")
            self.lbl_preview.setStyleSheet("color: green;")
        except (KeyError, ValueError, TypeError) as e:
            self.lbl_preview.setText(f"❌ Error: Invalid CSV format - {str(e)}")
            self.lbl_preview.setStyleSheet("color: red;")
            self.measurements = []
        except Exception as e:
            self.lbl_preview.setText(f"❌ Error: {str(e)}")
            self.lbl_preview.setStyleSheet("color: red;")
            self.measurements = []

    def get_measurements(self):
        return self.measurements


# Builds a compact grid of "label + text box" pairs (used for all the numeric parameters
# like focal length, sensor width, etc.) so we don't repeat the same layout code everywhere.
class QGridLayout_Helper:
    def __init__(self):
        self.widget = QWidget()
        self.layout = QGridLayout(self.widget)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.row = 0
        self.col = 0
        self.max_cols = 4

    # Adds one "label + input box" pair to the grid, wrapping to a new row every max_cols columns.
    def add_param(self, label_text, default_val):
        lbl = QLabel(label_text)
        inp = QLineEdit(default_val)
        inp.setFixedWidth(80)
        self.layout.addWidget(lbl, self.row, self.col)
        self.layout.addWidget(inp, self.row, self.col + 1)
        self.col += 2
        if self.col >= self.max_cols:
            self.col = 0
            self.row += 1
        return inp


# A picture viewer where the user can click a chain of points on the image (a "polyline")
# to measure a length in pixels, e.g. to measure an animal in a photo.
class NativeImageViewer(QGraphicsView):
    measurement_made = pyqtSignal(float) # Sends out the measured length in pixels once the user finishes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Rendering settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Use a crosshair cursor so it's clear the user is meant to click precise points
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Let this widget receive keyboard events (needed for zoom/finish/cancel shortcuts)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Data storage
        self.image_cv = None
        self.pixmap_item = None

        # Polyline state: the points clicked so far, the drawn segments, and the "rubber band"
        # preview line that follows the mouse before the next click.
        self.points = []
        self.temp_lines = []
        self.rubber_band = None
        self.is_measuring = False

        # Pen used to draw the confirmed measurement lines: bright yellow, 5px thick.
        self.pen = QPen(QColor(255, 255, 0))
        self.pen.setWidth(5)
        self.pen.setCosmetic(True)

        # Pen used for the dashed preview line that follows the mouse before the next click.
        self.pen_rubber = QPen(QColor(255, 255, 0))
        self.pen_rubber.setStyle(Qt.PenStyle.DashLine)
        self.pen_rubber.setWidth(4)
        self.pen_rubber.setCosmetic(True)

    # Loads a new image (as a numpy array) into the viewer and resets any in-progress measurement.
    def set_image(self, cv_img):
        self.image_cv = cv_img
        self.update_display(cv_img)
        self.points = []
        self.clear_temp_graphics()

    # Converts a numpy image array into a Qt image and shows it in the scene.
    def update_display(self, cv_img_to_show):
        self.scene.clear()
        height, width, channel = cv_img_to_show.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img_to_show.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(0, 0, width, height)

    # Handles mouse clicks: right button pans the view, left button places a measurement point.
    def mousePressEvent(self, event):
        # Right-click starts panning (dragging the view around)
        if event.button() == Qt.MouseButton.RightButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        # Left-click adds a new point to the polyline
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap_item:
            scene_pos = self.mapToScene(event.pos())

            if not self.is_measuring:
                # First click: start a brand-new measurement
                self.is_measuring = True
                self.points = [scene_pos]
                self.rubber_band = QGraphicsLineItem(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y())
                self.rubber_band.setPen(self.pen_rubber)
                self.scene.addItem(self.rubber_band)
            else:
                # Next click: close the previous segment with a solid line and start a new preview line
                prev_point = self.points[-1]
                self.points.append(scene_pos)

                segment = QGraphicsLineItem(prev_point.x(), prev_point.y(), scene_pos.x(), scene_pos.y())
                segment.setPen(self.pen)
                self.scene.addItem(segment)
                self.temp_lines.append(segment)

                self.rubber_band.setLine(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y())

        super().mousePressEvent(event)

    # While measuring, keeps the dashed preview line stuck to the current mouse position.
    def mouseMoveEvent(self, event):
        if self.is_measuring and self.rubber_band:
            new_pos = self.mapToScene(event.pos())
            start_pos = self.points[-1]
            self.rubber_band.setLine(start_pos.x(), start_pos.y(), new_pos.x(), new_pos.y())

        super().mouseMoveEvent(event)

    # When the right mouse button is released, stop panning and bring back the crosshair cursor.
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)

        super().mouseReleaseEvent(event)

    # Keyboard shortcuts: Ctrl/Cmd +/- to zoom, F to finish the measurement, Esc to cancel it.
    def keyPressEvent(self, event):
        # Check whether Ctrl (Windows/Linux) or Cmd (Mac) is held down
        modifiers = event.modifiers()
        is_ctrl_or_cmd = (modifiers & Qt.KeyboardModifier.ControlModifier)

        # --- ZOOM WITH KEYBOARD (CTRL + KEYS) ---
        if is_ctrl_or_cmd:
            zoom_in_factor = 1.10
            zoom_out_factor = 1 / zoom_in_factor

            # "+" key (or "=", same physical key on many keyboards)
            if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
                self.scale(zoom_in_factor, zoom_in_factor)
                return # Stop here, don't let the event propagate further

            # "-" key
            elif event.key() == Qt.Key.Key_Minus:
                self.scale(zoom_out_factor, zoom_out_factor)
                return # Stop here, don't let the event propagate further

        # --- MEASUREMENT SHORTCUTS (F and ESC) ---
        # These work even without Ctrl held down
        if event.key() == Qt.Key.Key_F:
            if self.is_measuring and len(self.points) > 1:
                self.finish_measurement()
            else:
                self.cancel_measurement()

        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_measurement()

        else:
            # Not a key we care about: let Qt's default handling deal with it
            super().keyPressEvent(event)

    # Adds up the length of every segment in the polyline (basic distance formula) and
    # reports the total length in pixels to whoever is listening.
    def finish_measurement(self):
        total_length = 0.0
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            dist = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            total_length += dist

        self.measurement_made.emit(total_length)

        if self.rubber_band:
            self.scene.removeItem(self.rubber_band)
            self.rubber_band = None

        self.is_measuring = False
        self.points = []

    # Aborts the current in-progress measurement without reporting a length.
    def cancel_measurement(self):
        self.is_measuring = False
        self.points = []
        self.clear_temp_graphics()

    # Removes the preview line and all confirmed segments from the scene.
    def clear_temp_graphics(self):
        if self.rubber_band:
            self.scene.removeItem(self.rubber_band)
        self.rubber_band = None
        for line in self.temp_lines:
            self.scene.removeItem(line)
        self.temp_lines = []

# This is the whole application window: it holds one tab per "pipeline" (a sequence of
# processing steps applied to drone footage + sensor logs).
class WhaleSyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhaleSync - Professional GUI v11 (Polyline)")
        self.resize(1200, 900)
        self.measurements_main = []
        self.measurements_extra = []
        self.measurements_mavic = []

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.tab_main = QWidget(); self.setup_tab_main(); self.tabs.addTab(self.tab_main, "Complete Pipeline")
        self.tab_extra = QWidget(); self.setup_tab_extra(); self.tabs.addTab(self.tab_extra, "Incomplete Pipeline")
        # The "Sync SRT <-> DAT" tab is no longer used: pipeline.sync_srt_with_dat is commented
        # out below, since merging SRT/DAT now goes through merge_DAT_SRT_standard instead.
        # self.tab_sync = QWidget(); self.setup_tab_sync(); self.tabs.addTab(self.tab_sync, "Sync SRT <-> DAT")
        self.tab_mavic = QWidget(); self.setup_tab_mavic(); self.tabs.addTab(self.tab_mavic, "Mavic Sync (SRT + Log)")

        # The "Image Analysis (Polyline)" tab is hidden from the interface (the code below,
        # setup_tab_analysis and NativeImageViewer, is kept untouched in case we re-enable it).
        # self.tab_analysis = QWidget(); self.setup_tab_analysis(); self.tabs.addTab(self.tab_analysis, "Image Analysis (Polyline)")

    # TAB 1: COMPLETE PIPELINE
    # Builds the "Complete Pipeline" tab: pick video/laser/DAT files, set the camera parameters, optionally add manual measurements, then run everything at once.
    def setup_tab_main(self):
        outer = QVBoxLayout(self.tab_main)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); scroll.setWidget(content)
        outer.addWidget(scroll)
        self.form_main = QVBoxLayout(content)

        # Files
        gb_files = QGroupBox("Input Files")
        l_files = QFormLayout()
        self.m_vid = self.create_file_input(l_files, "Video (or SRT):", "Video/SRT (*.mp4 *.mov *.MP4 *.MOV *.srt)")
        self.m_las = self.create_file_input(l_files, "Laser Log:", "Txt (*.txt *.csv *.log)")
        self.m_apr = self.create_file_input(l_files, "DAT CSV (optional):", "CSV (*.csv)")
        self.m_out_folder = self.create_folder_input(l_files, "Output Folder:")
        self.m_out_name = self.create_text_input(l_files, "Output Filename:", "merged_output.csv")
        gb_files.setLayout(l_files); self.form_main.addWidget(gb_files)

        # Parameters
        gb_params = QGroupBox("General Parameters")
        l_params = QGridLayout_Helper()
        self.m_rp = l_params.add_param("Rows Before:", "0")
        self.m_rd = l_params.add_param("Rows After:", "0")
        self.m_wimg = l_params.add_param("W Image (px):", "3840.0")
        self.m_wsens = l_params.add_param("W Sensor (mm):", "17.3")
        self.m_foc = l_params.add_param("Focal Length:", "24.962606")
        self.m_true = l_params.add_param("True Length (m):", "1.95")
        self.m_alt = l_params.add_param("Altitude (m):", "0.0")
        self.m_laser_offset = l_params.add_param("Laser Time Offset (s):", "0")
        self.m_srt_offset = l_params.add_param("SRT Time Offset (s):", "0")
        self.m_dat_hour_offset = l_params.add_param("DAT Hour Offset:", "0")
        self.m_srt_hour_offset = l_params.add_param("SRT Hour Offset:", "0")
        self.m_height_min = l_params.add_param("Laser Height Min (m):", "")
        self.m_height_max = l_params.add_param("Laser Height Max (m):", "")

        self.cb_correct_main = QCheckBox("Apply Correction & Fill (Requires Measurements)")
        self.cb_correct_main.setChecked(True)
        l_params.layout.addWidget(self.cb_correct_main, l_params.row + 1, 0, 1, 4)
        self.cb_dat_feet_main = QCheckBox("DAT Altitude in Feet (convert to meters)")
        self.cb_dat_feet_main.setChecked(False)
        l_params.layout.addWidget(self.cb_dat_feet_main, l_params.row + 2, 0, 1, 4)

        gb_params.setLayout(l_params.layout); self.form_main.addWidget(gb_params)

        # Measurements
        gb_meas = QGroupBox("Manual Measurements")
        l_meas = QVBoxLayout()
        self.list_meas_main = QListWidget(); self.list_meas_main.setFixedHeight(80)
        l_meas.addWidget(self.list_meas_main)
        h_btns = QHBoxLayout()
        btn_add = QPushButton("Add Measurement"); btn_add.clicked.connect(lambda: self.add_measurement_logic(self.measurements_main, self.list_meas_main))
        btn_del = QPushButton("Delete Selected"); btn_del.clicked.connect(lambda: self.delete_selected_measurement(self.measurements_main, self.list_meas_main))
        btn_clr = QPushButton("Clear All"); btn_clr.clicked.connect(lambda: self.clear_measurements(self.measurements_main, self.list_meas_main))
        btn_csv = QPushButton("Load from CSV"); btn_csv.clicked.connect(lambda: self.load_measurements_from_csv(self.measurements_main, self.list_meas_main))
        h_btns.addWidget(btn_add); h_btns.addWidget(btn_del); h_btns.addWidget(btn_clr); h_btns.addWidget(btn_csv)
        l_meas.addLayout(h_btns)
        gb_meas.setLayout(l_meas); self.form_main.addWidget(gb_meas)

        # Run
        btn_run = QPushButton("RUN COMPLETE PIPELINE")
        btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_run.clicked.connect(self.run_pipeline_main)
        self.form_main.addWidget(btn_run)

        # Results
        self.form_main.addWidget(self.create_separator())
        self.lbl_results_main = QLabel("Results will appear here...")
        self.lbl_results_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_main.setWordWrap(True)
        self.form_main.addWidget(self.lbl_results_main)
        self.results_container_main = QVBoxLayout(); self.form_main.addLayout(self.results_container_main)

        # Animal Pipeline
        self.form_main.addWidget(self.create_separator())
        gb_animal = QGroupBox("Animal Video Pipeline (Optional)")
        l_animal = QFormLayout()
        self.ma_vid = self.create_file_input(l_animal, "Animal Video (or SRT):", "Video/SRT (*.mp4 *.mov *.MP4 *.MOV *.srt)")
        self.ma_csv = self.create_file_input(l_animal, "Merged File:", "CSV (*.csv)")
        self.ma_out_folder = self.create_folder_input(l_animal, "Output Folder:")
        self.ma_out_name = self.create_text_input(l_animal, "Output Filename:", "merged_output_animal.csv")
        self.ma_rp = self.create_text_input(l_animal, "Rows Before:", "0")
        self.ma_rd = self.create_text_input(l_animal, "Rows After:", "0")
        self.ma_srt_offset = self.create_text_input(l_animal, "SRT Time Offset (s):", "0")
        self.ma_hour_offset = self.create_text_input(l_animal, "SRT Hour Offset:", "0")
        btn_run_animal = QPushButton("Run Animal Video Pipeline")
        btn_run_animal.clicked.connect(self.run_pipeline_animal_logic)
        l_animal.addRow(btn_run_animal)
        gb_animal.setLayout(l_animal); self.form_main.addWidget(gb_animal)
        self.lbl_results_animal = QLabel("")
        self.lbl_results_animal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_animal.setWordWrap(True)
        self.form_main.addWidget(self.lbl_results_animal)
        self.results_container_animal = QVBoxLayout(); self.form_main.addLayout(self.results_container_animal)

    # TAB 2: INCOMPLETE PIPELINE
    # Builds the "Incomplete Pipeline" tab.
    #  Unlike the complete pipeline, this one is split into three separate phases the user can run one at a time (useful when the video/SRT is missing or broken and syncing has to be done by hand, phase by phase).
    def setup_tab_extra(self):
        outer = QVBoxLayout(self.tab_extra)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); scroll.setWidget(content)
        outer.addWidget(scroll)
        self.form_extra = QVBoxLayout(content)

        gb_base = QGroupBox("Phase 1: Base (Laser + DAT)")
        l_base = QFormLayout()
        self.e_las = self.create_file_input(l_base, "Laser Log:", "Txt (*.txt *.csv)")
        self.e_apr = self.create_file_input(l_base, "DAT CSV:", "CSV (*.csv)")
        self.e_out_folder = self.create_folder_input(l_base, "Output Folder:")
        self.e_out_name = self.create_text_input(l_base, "Output Filename:", "merged_output.csv")
        self.e_laser_offset = self.create_text_input(l_base, "Laser Time Offset (s):", "0")
        self.e_hour_offset = self.create_text_input(l_base, "Hour Offset:", "0")
        self.e_base_height_min = self.create_text_input(l_base, "Laser Height Min (m):", "")
        self.e_base_height_max = self.create_text_input(l_base, "Laser Height Max (m):", "")
        self.cb_dat_feet_inc = QCheckBox("DAT Altitude in Feet (convert to meters)")
        self.cb_dat_feet_inc.setChecked(False)
        l_base.addRow(self.cb_dat_feet_inc)
        btn_base = QPushButton("Run Base Pipeline")
        btn_base.clicked.connect(self.run_pipeline_inc_base)
        l_base.addRow(btn_base)
        gb_base.setLayout(l_base); self.form_extra.addWidget(gb_base)
        self.lbl_results_base = QLabel("")
        self.lbl_results_base.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_base.setWordWrap(True)
        self.form_extra.addWidget(self.lbl_results_base)
        self.results_container_base = QVBoxLayout(); self.form_extra.addLayout(self.results_container_base)
        self.form_extra.addWidget(self.create_separator())

        gb_sync = QGroupBox("Phase 2: Sync No SRT")
        l_sync = QFormLayout()
        self.e_merged = self.create_file_input(l_sync, "Merged File:", "CSV (*.csv)")

        gb_params_sync = QGroupBox("Parameters")
        l_p2 = QGridLayout_Helper()
        self.e_secrel = l_p2.add_param("Sec Relative:", "0")
        self.e_secvid = l_p2.add_param("Sec Video:", "0")
        self.e_wimg = l_p2.add_param("W Img:", "3840.0")
        self.e_wsens = l_p2.add_param("W Sens:", "17.3")
        self.e_foc = l_p2.add_param("Focal:", "24.96")
        self.e_true = l_p2.add_param("True Len:", "1.95")
        self.e_alt = l_p2.add_param("Alt:", "0.0")
        self.e_laser_offset_2 = l_p2.add_param("Laser Time Offset (s):", "0")
        self.e_height_min = l_p2.add_param("Laser Height Min (m):", "")
        self.e_height_max = l_p2.add_param("Laser Height Max (m):", "")

        self.cb_correct_inc = QCheckBox("Apply Correction & Fill")
        self.cb_correct_inc.setChecked(True)
        l_p2.layout.addWidget(self.cb_correct_inc, l_p2.row + 1, 0, 1, 4)

        gb_params_sync.setLayout(l_p2.widget.layout())
        l_sync.addRow(gb_params_sync)

        self.e_out_inc_folder = self.create_folder_input(l_sync, "Output Folder:")
        self.e_out_inc_name = self.create_text_input(l_sync, "Output Filename:", "merged_output_inc.csv")

        self.list_meas_extra = QListWidget(); self.list_meas_extra.setFixedHeight(60)
        l_sync.addRow("Measurements:", self.list_meas_extra)
        hb = QHBoxLayout()
        b1 = QPushButton("Add"); b1.clicked.connect(lambda: self.add_measurement_logic(self.measurements_extra, self.list_meas_extra))
        b_del = QPushButton("Delete Selected"); b_del.clicked.connect(lambda: self.delete_selected_measurement(self.measurements_extra, self.list_meas_extra))
        b2 = QPushButton("Clear"); b2.clicked.connect(lambda: self.clear_measurements(self.measurements_extra, self.list_meas_extra))
        b_csv = QPushButton("Load from CSV"); b_csv.clicked.connect(lambda: self.load_measurements_from_csv(self.measurements_extra, self.list_meas_extra))
        hb.addWidget(b1); hb.addWidget(b_del); hb.addWidget(b2); hb.addWidget(b_csv)
        l_sync.addRow(hb)

        btn_sync = QPushButton("Run Sync NO SRT")
        btn_sync.clicked.connect(self.run_pipeline_inc_sync)
        l_sync.addRow(btn_sync)
        gb_sync.setLayout(l_sync); self.form_extra.addWidget(gb_sync)
        self.lbl_results_sync = QLabel("")
        self.lbl_results_sync.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_sync.setWordWrap(True)
        self.form_extra.addWidget(self.lbl_results_sync)
        self.results_container_sync = QVBoxLayout(); self.form_extra.addLayout(self.results_container_sync)
        self.form_extra.addWidget(self.create_separator())

        gb_fin = QGroupBox("Phase 3: Final Sync")
        l_fin = QFormLayout()
        self.e_merged_corr = self.create_file_input(l_fin, "Merged Corrected:", "CSV (*.csv)")
        self.e_sr_c = self.create_text_input(l_fin, "Sec Relative:", "0")
        self.e_sv_c = self.create_text_input(l_fin, "Sec Video:", "0")
        self.e_out_fin_folder = self.create_folder_input(l_fin, "Output Folder:")
        self.e_out_fin_name = self.create_text_input(l_fin, "Output Filename:", "merged_output_animal_inc.csv")
        btn_fin = QPushButton("Run Final Sync")
        btn_fin.clicked.connect(self.run_pipeline_inc_final)
        l_fin.addRow(btn_fin)
        gb_fin.setLayout(l_fin); self.form_extra.addWidget(gb_fin)

   
    # TAB 3: SYNC SRT
    # This whole tab is disabled (kept only as reference/backup code): the SRT/DAT sync it used to do is now handled elsewhere in the main flow, via merge_DAT_SRT_standard.

    # def setup_tab_sync(self):
    #     outer = QVBoxLayout(self.tab_sync)
    #     scroll = QScrollArea(); scroll.setWidgetResizable(True)
    #     content = QWidget(); scroll.setWidget(content)
    #     outer.addWidget(scroll)
    #     self.form_sync = QVBoxLayout(content)
    #
    #     gb = QGroupBox("Sync SRT <-> DAT")
    #     form = QFormLayout()
    #     self.s_vid = self.create_file_input(form, "Video:", "Video (*.mp4 *.mov *.MP4 *.MOV)")
    #     self.s_apr = self.create_file_input(form, "DAT CSV:", "CSV (*.csv)")
    #     self.s_rp = self.create_text_input(form, "Rows Before:", "0")
    #     self.s_rd = self.create_text_input(form, "Rows After:", "0")
    #     self.s_out_folder = self.create_folder_input(form, "Output Folder:")
    #     self.s_out_name = self.create_text_input(form, "Output Filename:", "merged_output_sync_srt.csv")
    #     self.s_height_min = self.create_text_input(form, "Laser Height Min (m):", "")
    #     self.s_height_max = self.create_text_input(form, "Laser Height Max (m):", "")
    #     self.s_hour_offset = self.create_text_input(form, "Hour Offset:", "0")
    #     self.cb_dat_feet_sync = QCheckBox("DAT Altitude in Feet (convert to meters)")
    #     self.cb_dat_feet_sync.setChecked(False)
    #     form.addRow(self.cb_dat_feet_sync)
    #     btn = QPushButton("Run Sync SRT")
    #     btn.clicked.connect(self.run_sync_srt)
    #     form.addRow(btn)
    #     gb.setLayout(form); self.form_sync.addWidget(gb)
    #     self.results_container_srt = QVBoxLayout(); self.form_sync.addLayout(self.results_container_srt); self.form_sync.addStretch()


    # TAB 4: MAVIC SYNC
    # Builds the "Mavic Sync" tab, for footage from a DJI Mavic drone. Its SRT subtitle
    # Its SRT subtitle file already contains GPS/telemetry, so this pipeline reads that directly instead of relying on a separate DAT log.

    def setup_tab_mavic(self):
        outer = QVBoxLayout(self.tab_mavic)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); scroll.setWidget(content)
        outer.addWidget(scroll)
        self.form_mavic = QVBoxLayout(content)

        # MAVIC SYNC (SRT + LOG)

        # Files
        gb_files = QGroupBox("Input Files")
        l_files = QFormLayout()
        self.mav_srt = self.create_file_input(l_files, "SRT File:", "SRT (*.srt)")
        self.mav_las = self.create_file_input(l_files, "Laser Log:", "Txt (*.txt *.csv *.log)")
        self.mav_out_folder = self.create_folder_input(l_files, "Output Folder:")
        self.mav_out_name = self.create_text_input(l_files, "Output Filename:", "mavic_sync_output.csv")
        gb_files.setLayout(l_files); self.form_mavic.addWidget(gb_files)

        # Params
        gb_params = QGroupBox("Camera / Sensor Parameters")
        l_params = QGridLayout_Helper()
        self.mav_wimg = l_params.add_param("W Image (px):", "3840.0")
        self.mav_wsens = l_params.add_param("W Sensor (mm):", "17.29")
        self.mav_foc = l_params.add_param("Focal Length:", "12.1086484375 ")
        self.mav_true = l_params.add_param("True Length (m):", "1.95")
        self.mav_alt = l_params.add_param("Altitude (m):", "0.0")
        self.mav_offset = l_params.add_param("Hour Offset:", "-1")
        self.mav_laser_offset = l_params.add_param("Laser Time Offset (s):", "0")
        self.mav_rp = l_params.add_param("Rows Before:", "0")
        self.mav_rd = l_params.add_param("Rows After:", "0")
        self.mav_height_min = l_params.add_param("Laser Height Min (m):", "")
        self.mav_height_max = l_params.add_param("Laser Height Max (m):", "")

        self.cb_correct_mavic = QCheckBox("Apply Correction & Fill (Laser only)")
        self.cb_correct_mavic.setChecked(True)
        l_params.layout.addWidget(self.cb_correct_mavic, l_params.row + 1, 0, 1, 4)

        gb_params.setLayout(l_params.widget.layout()); self.form_mavic.addWidget(gb_params)

        # Measurements
        gb_meas = QGroupBox("Measurements for Correction & Fill")
        l_meas = QVBoxLayout()
        self.list_meas_mavic = QListWidget(); self.list_meas_mavic.setFixedHeight(80)
        l_meas.addWidget(self.list_meas_mavic)
        h_btns = QHBoxLayout()
        btn_add = QPushButton("Add Measurement"); btn_add.clicked.connect(lambda: self.add_measurement_logic(self.measurements_mavic, self.list_meas_mavic))
        btn_del = QPushButton("Delete Selected"); btn_del.clicked.connect(lambda: self.delete_selected_measurement(self.measurements_mavic, self.list_meas_mavic))
        btn_clr = QPushButton("Clear All"); btn_clr.clicked.connect(lambda: self.clear_measurements(self.measurements_mavic, self.list_meas_mavic))
        btn_csv = QPushButton("Load from CSV"); btn_csv.clicked.connect(lambda: self.load_measurements_from_csv(self.measurements_mavic, self.list_meas_mavic))
        h_btns.addWidget(btn_add); h_btns.addWidget(btn_del); h_btns.addWidget(btn_clr); h_btns.addWidget(btn_csv)
        l_meas.addLayout(h_btns)
        gb_meas.setLayout(l_meas); self.form_mavic.addWidget(gb_meas)

        # Run 1
        btn_run = QPushButton("RUN MAVIC SYNC (SRT + LOG)")
        btn_run.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")
        btn_run.clicked.connect(self.run_pipeline_mavic)
        self.form_mavic.addWidget(btn_run)

        self.form_mavic.addWidget(self.create_separator())
        self.lbl_results_mavic = QLabel("Results will appear here...")
        self.lbl_results_mavic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_mavic.setWordWrap(True)
        self.form_mavic.addWidget(self.lbl_results_mavic)
        self.results_container_mavic = QVBoxLayout(); self.form_mavic.addLayout(self.results_container_mavic)

        # SECTION 2: OTHER VIDEOS SYNC
        # Same idea as Section 1, but for a video that was already synced to a log elsewhere,using a plain meter offset instead of recomputing distance from scratch.

        gb_files_2 = QGroupBox("Other videos sync")
        l_files_2 = QFormLayout()
        self.mav_srt_new = self.create_file_input(l_files_2, "SRT File:", "SRT (*.srt)")
        self.mav_synched = self.create_file_input(l_files_2, "LOG File:", "Txt (*.txt *.csv *.log)")
        self.mav_out_folder_new = self.create_folder_input(l_files_2, "Output Folder:")
        self.mav_out_name_new = self.create_text_input(l_files_2, "Output Filename:", "mavic_sync_output_animal.csv")
        gb_files_2.setLayout(l_files_2); self.form_mavic.addWidget(gb_files_2)

        gb_params_2 = QGroupBox("Info:")
        l_params_2 = QGridLayout_Helper()
        self.mav_offset_2 = l_params_2.add_param("Hour Offset:", "-1")
        self.offset_m = l_params_2.add_param("OFFSET in meters:", "0.0")
        self.mav_laser_offset_2 = l_params_2.add_param("Laser Time Offset (s):", "0")
        self.mav_rp_2 = l_params_2.add_param("Rows Before:", "0")
        self.mav_rd_2 = l_params_2.add_param("Rows After:", "0")
        self.mav_height_min_2 = l_params_2.add_param("Laser Height Min (m):", "")
        self.mav_height_max_2 = l_params_2.add_param("Laser Height Max (m):", "")
        self.cb_correct_mavic_2 = QCheckBox("Apply Correction & Fill (Laser only)")
        self.cb_correct_mavic_2.setChecked(True)
        l_params_2.layout.addWidget(self.cb_correct_mavic_2, l_params_2.row + 1, 0, 1, 4)

        gb_params_2.setLayout(l_params_2.widget.layout()); self.form_mavic.addWidget(gb_params_2)

        # Run 2 (button variable renamed for clarity)
        btn_run_2 = QPushButton("RUN NEW SYNC")
        btn_run_2.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")
        btn_run_2.clicked.connect(self.run_pipeline_mavic_new)
        self.form_mavic.addWidget(btn_run_2)

        self.form_mavic.addWidget(self.create_separator())

        self.lbl_results_mavic_new = QLabel("Results will appear here...")
        self.lbl_results_mavic_new.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_results_mavic_new.setWordWrap(True)
        self.form_mavic.addWidget(self.lbl_results_mavic_new)

        self.results_container_mavic_2 = QVBoxLayout()
        self.form_mavic.addLayout(self.results_container_mavic_2)


    # TAB 5: NATIVE IMAGE ANALYSIS (OPENCV + PYQT)

    # Builds the (currently hidden, see __init__) tab where a single still image can be loaded, brightness/contrast adjusted, and measured directly with the polyline tool.

    def setup_tab_analysis(self):
        # Main horizontal layout: controls on the left, image viewer on the right
        layout = QHBoxLayout(self.tab_analysis)


        sidebar = QVBoxLayout()
        sidebar_widget = QWidget(); sidebar_widget.setLayout(sidebar); sidebar_widget.setFixedWidth(320)
        layout.addWidget(sidebar_widget)

        # Load Image
        gb_load = QGroupBox("1. Load Image")
        l_load = QHBoxLayout()
        self.btn_load_native = QPushButton("Open Image")
        self.btn_load_native.clicked.connect(self.load_image_native)
        l_load.addWidget(self.btn_load_native)
        gb_load.setLayout(l_load)
        sidebar.addWidget(gb_load)

        # 2. Image Adjustments (Window/Level)
        gb_adj = QGroupBox("2. Adjust (Contrast/Brightness)")
        l_adj = QGridLayout()

        l_adj.addWidget(QLabel("Contrast:"), 0, 0)
        self.sl_contrast = QSlider(Qt.Orientation.Horizontal)
        self.sl_contrast.setRange(1, 30); self.sl_contrast.setValue(10) # slider units map to a 1.0-3.0 multiplier
        self.sl_contrast.valueChanged.connect(self.update_image_native)
        l_adj.addWidget(self.sl_contrast, 0, 1)

        l_adj.addWidget(QLabel("Brightness:"), 1, 0)
        self.sl_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sl_brightness.setRange(-100, 100); self.sl_brightness.setValue(0)
        self.sl_brightness.valueChanged.connect(self.update_image_native)
        l_adj.addWidget(self.sl_brightness, 1, 1)

        btn_reset_adj = QPushButton("Reset"); btn_reset_adj.clicked.connect(self.reset_adjustments)
        l_adj.addWidget(btn_reset_adj, 2, 0, 1, 2)

        gb_adj.setLayout(l_adj)
        sidebar.addWidget(gb_adj)

        # Measurements
        gb_meas = QGroupBox("3. Polyline Measure Tool")
        l_meas = QVBoxLayout()
        # Short instructions shown to the user (kept in Italian here since it's UI text, not code)

        self.table_native = QTableWidget()
        self.table_native.setColumnCount(2)
        self.table_native.setHorizontalHeaderLabels(["Label", "Length (px)"])
        self.table_native.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        l_meas.addWidget(self.table_native)

        btn_clear_lines = QPushButton("Clear Graphics")
        btn_clear_lines.clicked.connect(self.clear_all_graphics)
        l_meas.addWidget(btn_clear_lines)

        # Actions on the currently selected measurement row
        btn_use_native = QPushButton("Use Selected Measurement")
        btn_use_native.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_use_native.clicked.connect(self.use_native_measurement)
        l_meas.addWidget(btn_use_native)

        btn_del_row = QPushButton("Delete Selected Row")
        btn_del_row.clicked.connect(lambda: self.table_native.removeRow(self.table_native.currentRow()))
        l_meas.addWidget(btn_del_row)

        gb_meas.setLayout(l_meas)
        sidebar.addWidget(gb_meas)
        sidebar.addStretch()

        # RIGHT SIDE (viewer) 
        self.viewer_native = NativeImageViewer()
        self.viewer_native.measurement_made.connect(self.add_native_result)

        # Wrap the viewer in a framed container just to give it a visible border
        viewer_container = QFrame()
        viewer_container.setFrameShape(QFrame.Shape.StyledPanel)
        l_cont = QVBoxLayout(viewer_container)
        l_cont.setContentsMargins(0,0,0,0)
        l_cont.addWidget(self.viewer_native)

        layout.addWidget(viewer_container)

    # Opens a file picker, loads the chosen image as RGB, and shows it in the viewer.
    def load_image_native(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif *.bmp)")
        if f:
            try:
                pil_img = Image.open(f).convert("RGB")
                img = np.array(pil_img)
            except Exception:
                QMessageBox.warning(self, "Error", "Could not load image.")
                return

            self.viewer_native.set_image(img)
            self.reset_adjustments()

    # Puts the contrast/brightness sliders back to their default (no change) position.
    def reset_adjustments(self):
        self.sl_contrast.setValue(10)
        self.sl_brightness.setValue(0)
        if self.viewer_native.image_cv is not None:
            self.viewer_native.update_display(self.viewer_native.image_cv)

    # Re-draws the image using the current contrast (multiply) and brightness (add) sliders.
    def update_image_native(self):
        if self.viewer_native.image_cv is None: return

        alpha = self.sl_contrast.value() / 10.0
        beta = self.sl_brightness.value()

        adjusted = np.clip(self.viewer_native.image_cv.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        self.viewer_native.update_display(adjusted)

    # Called when the image viewer reports a finished measurement; adds it as a new table row.
    def add_native_result(self, length_px):
        row = self.table_native.rowCount()
        self.table_native.insertRow(row)
        self.table_native.setItem(row, 0, QTableWidgetItem(f"Polyline {row+1}"))
        self.table_native.setItem(row, 1, QTableWidgetItem(f"{length_px:.3f}"))

    # If read_log had to patch up gaps or clock-jumps in the laser log, this shows the
    # full repair details in a popup: the packaged executable has no visible console,
    # so a simple print() would never reach the user.
    def show_log_repair_detail(self, merged):
        detail = merged.attrs.get("log_repair_detail")
        if detail:
            dlg = QDialog(self)
            dlg.setWindowTitle("Log laser: timestamp irregolari")
            dlg.resize(600, 400)
            layout = QVBoxLayout(dlg)
            text = QTextEdit(dlg)
            text.setReadOnly(True)
            text.setPlainText(detail)
            layout.addWidget(text)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(dlg.accept)
            layout.addWidget(buttons)
            dlg.exec()

    # Takes the currently selected row of the measurement table and adds it to the
    # "Complete Pipeline" measurement list, asking the user for the matching video second.
    def use_native_measurement(self):
        r = self.table_native.currentRow()
        if r < 0:
            QMessageBox.warning(self, "Warning", "Select a row first.")
            return

        val_px = float(self.table_native.item(r, 1).text())

        dlg = AddMeasurementDialog(self)
        dlg.px_input.setValue(val_px)
        if dlg.exec():
            px, sec = dlg.get_data()
            self.measurements_main.append({'len_pitrinu_px': px, 'sec_vid': int(sec)})
            self.list_meas_main.addItem(f"{int(sec)} s | {px:.3f} px")
            QMessageBox.information(self, "Success", "Added to Complete Pipeline measurements.")

    def clear_all_graphics(self):
        self.viewer_native.clear_temp_graphics()


    # Builds a "text box + Browse (...) button" row for picking a file, and adds it to layout.
    def create_file_input(self, layout, label, filter_ext):
        line = QLineEdit(); btn = QPushButton("..."); btn.setFixedWidth(40)
        btn.clicked.connect(lambda: self.pick_file(line, filter_ext))
        row = QHBoxLayout(); row.addWidget(line); row.addWidget(btn)
        if isinstance(layout, QFormLayout): layout.addRow(label, row)
        else: layout.addLayout(row)
        return line

    # Same as create_file_input, but opens a folder picker instead of a file picker.
    def create_folder_input(self, layout, label):
        line = QLineEdit(); btn = QPushButton("..."); btn.setFixedWidth(40)
        btn.clicked.connect(lambda: self.pick_folder(line))
        row = QHBoxLayout(); row.addWidget(line); row.addWidget(btn)
        if isinstance(layout, QFormLayout): layout.addRow(label, row)
        else: layout.addLayout(row)
        return line

    # Builds a plain text box (no browse button) pre-filled with a default value.
    def create_text_input(self, layout, label, default):
        line = QLineEdit(default)
        if isinstance(layout, QFormLayout): layout.addRow(label, line)
        return line

    # A thin horizontal line used to visually separate sections in the form.
    def create_separator(self):
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def parse_height_filter(self, height_min_widget, height_max_widget):
        """Parse height filter values. Returns (min_val, max_val) or (None, None) if empty."""
        min_val = None
        max_val = None
        try:
            min_text = height_min_widget.text().strip()
            max_text = height_max_widget.text().strip()
            if min_text:
                min_val = float(min_text)
            if max_text:
                max_val = float(max_text)
        except ValueError:
            pass
        return min_val, max_val

    # Joins the output folder and filename the user typed into one path (folder is optional).
    def get_full_output_path(self, folder_widget, name_widget):
        folder = folder_widget.text(); name = name_widget.text()
        return os.path.join(folder, name) if folder else name

    # If out_path already exists on disk, ask the user to confirm before overwriting it.
    def confirm_overwrite(self, out_path: str) -> bool:
        if out_path and os.path.exists(out_path):
            reply = QMessageBox.question(
                self, "Conferma sovrascrittura",
                f"Il file esiste gia':\n{out_path}\n\nVuoi sovrascriverlo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def pick_file(self, line_edit, filters):
        f, _ = QFileDialog.getOpenFileName(self, "Select File", "", filters)
        if f: line_edit.setText(f)

    def pick_folder(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d: line_edit.setText(d)

    # Opens the "add one measurement" dialog and, if confirmed, appends it to list_data
    # and shows it in list_widget.
    def add_measurement_logic(self, list_data, list_widget):
        dlg = AddMeasurementDialog(self)
        if dlg.exec():
            px, sec = dlg.get_data()
            list_data.append({'len_pitrinu_px': px, 'sec_vid': int(sec)})
            list_widget.addItem(f"{int(sec)} s | {px:.3f} px")

    # Opens the "load measurements from CSV" dialog and, if any valid rows were found,
    # replaces list_data/list_widget with them.
    def load_measurements_from_csv(self, list_data, list_widget):
        dlg = LoadMeasurementCSVDialog(self)
        if dlg.exec():
            measurements = dlg.get_measurements()
            if not measurements:
                QMessageBox.warning(self, "Warning", "No valid measurements found in CSV.")
                return

            list_data.clear()
            list_widget.clear()
            for m in measurements:
                list_data.append(m)
                list_widget.addItem(f"{m['sec_vid']} s | {m['len_pitrinu_px']:.3f} px")

            QMessageBox.information(self, "Success", f"Loaded {len(measurements)} measurements from CSV.")


    # Removes the currently selected measurement, both from the on-screen list and from
    # the underlying data list (the two must always be removed together, in step).
    def delete_selected_measurement(self, list_data, list_widget):
        row = list_widget.currentRow()
        if row >= 0:
            list_widget.takeItem(row) # Remove from the visible list
            list_data.pop(row)        # Remove from the actual data stored in memory

    def clear_measurements(self, list_data, list_widget):
        list_data.clear(); list_widget.clear()

    PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def build_plot_widget(self, spec, height=400):
        """Builds a NoWheelPlotWidget from a fig-spec {'title', 'series'}. Reused both for
        the pipeline result plots and for the alignment confirmation dialog."""
        pw = NoWheelPlotWidget(title=spec['title'])
        pw.setFixedHeight(height)
        pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pw.addLegend(offset=(10, 10))
        pw.setBackground('w')

        pw.getAxis('left').setPen('k')
        pw.getAxis('bottom').setPen('k')
        pw.getAxis('left').setTextPen('k')
        pw.getAxis('bottom').setTextPen('k')
        pw.getAxis('left').tickFont = pg.Qt.QtGui.QFont()
        pw.getAxis('left').tickFont.setPointSize(11)
        pw.getAxis('bottom').tickFont = pg.Qt.QtGui.QFont()
        pw.getAxis('bottom').tickFont.setPointSize(11)
        pw.setTitle(spec['title'], size='14pt')

        for j, s in enumerate(spec['series']):
            color = self.PLOT_COLORS[j % len(self.PLOT_COLORS)]
            pen = pg.mkPen(color=color, width=3)
            if s.get('dash'):
                pen.setStyle(Qt.PenStyle.DashLine)
            x = s['x'].astype(float)
            y = s['y'].astype(float)
            pw.plot(x, y, pen=pen, name=s['label'], connect='finite')
        return pw

    # Clears out any previously shown plots/labels from layout_container, then draws one
    # plot per fig-spec in figs and saves each one as a PNG next to the output CSV.
    def process_and_show_results(self, figs, csv_path, layout_container):
        # Empty the container first (remove old widgets from a previous run)
        while layout_container.count():
            item = layout_container.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

        base_name = os.path.splitext(csv_path)[0]

        for i, spec in enumerate(figs, start=1):
            label = QLabel(f"<b>Graph {i}</b> — {spec['title']}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout_container.addWidget(label)

            pw = self.build_plot_widget(spec)

            # Add the plot widget to the layout before saving it as an image
            layout_container.addWidget(pw)
            layout_container.addWidget(self.create_separator())
            QApplication.instance().processEvents()

            # Save as PNG - captures exactly what the user sees on screen
            save_path = f"{base_name}_fig{i}.png"
            try:
                # grab() takes a screenshot of the widget as currently rendered
                pixmap = pw.grab()
                pixmap.save(save_path)
            except (IOError, OSError) as e:
                print(f"Save err fig {i}: Cannot write to file - {e}")
            except Exception as e:
                print(f"Save err fig {i}: {e}")


    # EXECUTION

    # Runs the full pipeline for tab 1 (video/laser/DAT -> merged CSV + plots), then shows
    # the result and any warning messages the pipeline attached to the merged dataframe.
    def run_pipeline_main(self):
        video = self.m_vid.text(); laser = self.m_las.text(); apr = self.m_apr.text()
        out_path = self.get_full_output_path(self.m_out_folder, self.m_out_name)
        # The DAT log is optional: without it the pipeline matches the laser log straight
        # against the SRT timestamps, like the Mavic pipeline does.
        if not (video and laser):
            QMessageBox.warning(self, "Error", "Select video and laser log!"); return
        if not self.confirm_overwrite(out_path): return

        try:
            self.lbl_results_main.setText("Running..."); QApplication.processEvents()
            h_min, h_max = self.parse_height_filter(self.m_height_min, self.m_height_max)
            merged, figs, csv_path = pipeline.run_pipeline(
                video_file=video, laser_csv=laser, apr_csv=apr, output_csv=out_path,
                righe_prima=int(self.m_rp.text()), righe_dopo=int(self.m_rd.text()),
                measurements=self.measurements_main,
                w_img=float(self.m_wimg.text()), w_sens=float(self.m_wsens.text()),
                true_length_m=float(self.m_true.text()), focal_lenght=float(self.m_foc.text()),
                altitude_pitrinu=float(self.m_alt.text()),
                apply_correction=self.cb_correct_main.isChecked(),
                laser_time_offset_s=float(self.m_laser_offset.text()),
                height_min=h_min, height_max=h_max,
                srt_time_offset_s=float(self.m_srt_offset.text()),
                dat_offset_ore=int(self.m_dat_hour_offset.text()),
                srt_offset_ore=int(self.m_srt_hour_offset.text()),
                dat_in_feet=self.cb_dat_feet_main.isChecked()
            )
            # merged.attrs holds extra info the pipeline attached to the dataframe (not real columns), used here to surface warnings/status messages to the user.
            repair_msg = merged.attrs.get("log_repair_message")
            score_msg = merged.attrs.get("alignment_score_message")
            match_msg = merged.attrs.get("offset_match_message")
            srt_match_msg = merged.attrs.get("srt_match_message")
            srt_offset_match_msg = merged.attrs.get("srt_offset_match_message")
            self.show_log_repair_detail(merged)
            status = f"Done! {csv_path}"
            if repair_msg:
                status += f"\n{repair_msg}"
            sync_line = " | ".join(m for m in (match_msg, score_msg, srt_match_msg, srt_offset_match_msg) if m)
            if sync_line:
                status += f"\n{sync_line}"
            self.lbl_results_main.setText(status)
            self.process_and_show_results(figs, csv_path, self.results_container_main)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
            self.lbl_results_main.setText("Error: Invalid input")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
            self.lbl_results_main.setText("Error: File not found")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
            self.lbl_results_main.setText("Error: Processing failed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            self.lbl_results_main.setText("Error.")

    # Runs the optional "animal video" pipeline: syncs a second video against an already merged CSV, so it shares the same laser/DAT timeline as the main pipeline.
    def run_pipeline_animal_logic(self):
        out_path = self.get_full_output_path(self.ma_out_folder, self.ma_out_name)
        if not self.confirm_overwrite(out_path): return
        try:
            merged, figs, csv_path = pipeline.run_pipeline_video_sync(
                video_file_a=self.ma_vid.text(), merged_file=self.ma_csv.text(),
                output_csv_a=out_path,
                righe_prima_animal=int(self.ma_rp.text()), righe_dopo_animal=int(self.ma_rd.text()),
                offset_ore=int(self.ma_hour_offset.text()),
                srt_time_offset_s=float(self.ma_srt_offset.text())
            )
            srt_match_msg = merged.attrs.get("srt_match_message")
            srt_offset_match_msg = merged.attrs.get("srt_offset_match_message")
            sync_line = " | ".join(m for m in (srt_match_msg, srt_offset_match_msg) if m)
            self.lbl_results_animal.setText(sync_line)
            self.process_and_show_results(figs, csv_path, self.results_container_animal)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    # Phase 1 of the incomplete pipeline: merges just the laser log and the DAT log.
    def run_pipeline_inc_base(self):
        out_path = self.get_full_output_path(self.e_out_folder, self.e_out_name)
        if not self.confirm_overwrite(out_path): return
        try:
            h_min, h_max = self.parse_height_filter(self.e_base_height_min, self.e_base_height_max)
            merged, figs, csv_path = pipeline.run_pipeline_no_srt(
                laser_csv=self.e_las.text(), apr_csv=self.e_apr.text(), output_csv=out_path,
                laser_time_offset_s=float(self.e_laser_offset.text()),
                offset_ore=int(self.e_hour_offset.text()),
                dat_in_feet=self.cb_dat_feet_inc.isChecked(),
                height_min=h_min, height_max=h_max
            )
            repair_msg = merged.attrs.get("log_repair_message")
            score_msg = merged.attrs.get("alignment_score_message")
            self.show_log_repair_detail(merged)
            base_status = " | ".join(m for m in (repair_msg, score_msg) if m)
            self.lbl_results_base.setText(base_status)
            self.process_and_show_results(figs, csv_path, self.results_container_base)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    # Phase 2 of the incomplete pipeline: aligns the Phase-1 merged file to a manually chosen reference point (target_sec / target_val) since there is no SRT to sync from.
    def run_pipeline_inc_sync(self):
        out_path = self.get_full_output_path(self.e_out_inc_folder, self.e_out_inc_name)
        if not self.confirm_overwrite(out_path): return
        try:
            h_min, h_max = self.parse_height_filter(self.e_height_min, self.e_height_max)
            merged, figs, csv_path = pipeline.run_pipeline_no_srt_correction(
                merged_path=self.e_merged.text(),
                target_sec=int(self.e_secrel.text()), target_val=int(self.e_secvid.text()),
                output_csv=out_path, measurements=self.measurements_extra,
                w_img=float(self.e_wimg.text()), w_sens=float(self.e_wsens.text()),
                true_length_m=float(self.e_true.text()), focal_lenght=float(self.e_foc.text()),
                altitude_pitrinu=float(self.e_alt.text()),
                apply_correction=self.cb_correct_inc.isChecked(),
                laser_time_offset_s=float(self.e_laser_offset_2.text()),
                height_min=h_min, height_max=h_max
            )
            score_msg = merged.attrs.get("alignment_score_message")
            match_msg = merged.attrs.get("offset_match_message")
            sync_line = " | ".join(m for m in (match_msg, score_msg) if m)
            self.lbl_results_sync.setText(sync_line)
            self.process_and_show_results(figs, csv_path, self.results_container_sync)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    # Phase 3 of the incomplete pipeline: applies a second, final time alignment on top of the corrected Phase-2 file and writes the final output CSV.
    def run_pipeline_inc_final(self):
        out_path = self.get_full_output_path(self.e_out_fin_folder, self.e_out_fin_name)
        if not self.confirm_overwrite(out_path): return
        try:
            csv_path = pipeline.run_pipeline_no_srt_final_sync(
                merged_path=self.e_merged_corr.text(),
                target_sec=int(self.e_sr_c.text()), target_val=int(self.e_sv_c.text()),
                output_csv=out_path
            )
            QMessageBox.information(self, "Success", f"Done.\n{csv_path}")
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    # Disabled together with tab 3 above: kept only as reference for the old SRT/DAT sync flow.
    # def run_sync_srt(self):
    #     out_path = self.get_full_output_path(self.s_out_folder, self.s_out_name)
    #     if not self.confirm_overwrite(out_path): return
    #     try:
    #         h_min, h_max = self.parse_height_filter(self.s_height_min, self.s_height_max)
    #         csv_path, figs = pipeline.sync_srt_with_dat(
    #             video_file=self.s_vid.text(), apr_csv=self.s_apr.text(),
    #             output_csv=out_path,
    #             righe_prima=int(self.s_rp.text()), righe_dopo=int(self.s_rd.text()),
    #             height_min=h_min, height_max=h_max,
    #             offset_ore=int(self.s_hour_offset.text()),
    #             dat_in_feet=self.cb_dat_feet_sync.isChecked()
    #         )
    #         self.process_and_show_results(figs, csv_path, self.results_container_srt)
    #     except (ValueError, TypeError) as e:
    #         QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
    #     except FileNotFoundError as e:
    #         QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
    #     except RuntimeError as e:
    #         QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    # Runs the Mavic sync pipeline: SRT telemetry + laser log TO merged CSV + plots.
    def run_pipeline_mavic(self):
        srt = self.mav_srt.text(); laser = self.mav_las.text()
        out_path = self.get_full_output_path(self.mav_out_folder, self.mav_out_name)
        if not (srt):
            QMessageBox.warning(self, "Error", "Select SRT!"); return
        if not self.confirm_overwrite(out_path): return

        try:
            self.lbl_results_mavic.setText("Running..."); QApplication.processEvents()
            h_min, h_max = self.parse_height_filter(self.mav_height_min, self.mav_height_max)
            merged, figs, csv_path = pipeline.sync_mavik_srt_log(
                srt_file=srt, output_csv=out_path, laser_csv=laser,
                measurements=self.measurements_mavic,
                w_img=float(self.mav_wimg.text()), w_sens=float(self.mav_wsens.text()),
                true_length_m=float(self.mav_true.text()), focal_lenght=float(self.mav_foc.text()),
                altitude_pitrinu=float(self.mav_alt.text()),
                offset_ore=int(self.mav_offset.text()),
                apply_correction=self.cb_correct_mavic.isChecked(),
                laser_time_offset_s=float(self.mav_laser_offset.text()),
                height_min=h_min, height_max=h_max,
                righe_prima=int(self.mav_rp.text()), righe_dopo=int(self.mav_rd.text()),
            )
            score_msg = merged.attrs.get("alignment_score_message")
            repair_msg = merged.attrs.get("log_repair_message")
            ts_repair_msg = merged.attrs.get("timestamp_repair_message")
            self.show_log_repair_detail(merged)
            match_msg = merged.attrs.get("offset_match_message")
            status = f"Done! {csv_path}"
            if repair_msg:
                status += f"\n{repair_msg}"
            if ts_repair_msg:
                status += f"\n{ts_repair_msg}"
            sync_line = " | ".join(m for m in (match_msg, score_msg) if m)
            if sync_line:
                status += f"\n{sync_line}"
            self.lbl_results_mavic.setText(status)
            self.process_and_show_results(figs, csv_path, self.results_container_mavic)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
            self.lbl_results_mavic.setText("Error: Invalid input")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
            self.lbl_results_mavic.setText("Error: File not found")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
            self.lbl_results_mavic.setText("Error: Processing failed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            self.lbl_results_mavic.setText("Error.")

    # Runs the Mavic sync pipeline (Section 2): same idea, but for a video already synced to a log elsewhere, so it uses a known meter offset instead of laser distance.
    def run_pipeline_mavic_new(self):
        srt = self.mav_srt_new.text(); csv_synched = self.mav_synched.text()
        out_path = self.get_full_output_path(self.mav_out_folder_new, self.mav_out_name_new)
        if not (srt):
            QMessageBox.warning(self, "Error", "Select SRT!"); return
        if not self.confirm_overwrite(out_path): return

        try:
            self.lbl_results_mavic_new.setText("Running..."); QApplication.processEvents()
            h_min, h_max = self.parse_height_filter(self.mav_height_min_2, self.mav_height_max_2)
            merged, figs, csv_path = pipeline.sync_mavik_srt_log_2(
                srt_file=srt, output_csv=out_path, laser_csv=csv_synched,
                offset_metri=float(self.offset_m.text()),
                offset_ore=int(self.mav_offset_2.text()),
                apply_correction=self.cb_correct_mavic_2.isChecked(),
                laser_time_offset_s=float(self.mav_laser_offset_2.text()),
                height_min=h_min, height_max=h_max,
                righe_prima=int(self.mav_rp_2.text()), righe_dopo=int(self.mav_rd_2.text()),
            )
            score_msg = merged.attrs.get("alignment_score_message")
            repair_msg = merged.attrs.get("log_repair_message")
            ts_repair_msg = merged.attrs.get("timestamp_repair_message")
            self.show_log_repair_detail(merged)
            match_msg = merged.attrs.get("offset_match_message")
            status = f"Done! {csv_path}"
            if repair_msg:
                status += f"\n{repair_msg}"
            if ts_repair_msg:
                status += f"\n{ts_repair_msg}"
            sync_line = " | ".join(m for m in (match_msg, score_msg) if m)
            if sync_line:
                status += f"\n{sync_line}"
            self.lbl_results_mavic_new.setText(status)
            self.process_and_show_results(figs, csv_path, self.results_container_mavic_2)
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {str(e)}")
            self.lbl_results_mavic_new.setText("Error: Invalid input")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
            self.lbl_results_mavic_new.setText("Error: File not found")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
            self.lbl_results_mavic_new.setText("Error: Processing failed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            self.lbl_results_mavic_new.setText("Error.")

#launches the Qt application and shows the main window.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WhaleSyncApp()
    window.show()
    sys.exit(app.exec())
