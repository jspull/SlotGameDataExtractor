import sys
import os
import cv2
# Import EasyOCR (torch) BEFORE PyQt5 to prevent WinError 1114 DLL initialization failure
from extractor import ExtractorThread, DEFAULT_EVENT_ENTRIES
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QTableWidget, QTableWidgetItem, QMessageBox,
                             QHeaderView, QSplitter, QTabWidget, QComboBox, QSizePolicy,
                             QLineEdit, QSlider, QStyle, QDialog, QDoubleSpinBox,
                             QCheckBox, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import json
import time

# matplotlib for Graph Tab
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Dark Theme Stylesheet ───────────────────────────────────────────
DARK_STYLE = """
QMainWindow { background-color: #1a1a2e; }
QWidget { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QLabel { color: #c0c0c0; }
QLabel#titleLabel { font-size: 18px; font-weight: bold; color: #00d4aa; letter-spacing: 1px; }
QLabel#statusLabel { font-size: 13px; font-weight: bold; color: #ffb347; padding: 4px 0px; }
QLabel#videoDisplay { background-color: #0f0f23; border: 2px solid #16213e; border-radius: 8px; }
QLabel#timeLabel { color: #aaa; font-size: 12px; }
QPushButton {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 12px;
}
QPushButton:hover { background-color: #0f3460; border-color: #00d4aa; color: #00d4aa; }
QPushButton:pressed { background-color: #00d4aa; color: #0f0f23; }
QPushButton:disabled { background-color: #111; color: #555; border-color: #222; }
QPushButton#btnAccent {
    background-color: #00d4aa; color: #0f0f23; border: none; font-weight: bold;
}
QPushButton#btnAccent:hover { background-color: #00f0c0; }
QPushButton#btnAccent:disabled { background-color: #333; color: #666; }
QPushButton#btnDanger { background-color: #e74c3c; color: #fff; border: none; }
QPushButton#btnDanger:hover { background-color: #ff6b6b; }
QPushButton#btnDanger:disabled { background-color: #333; color: #666; }
QPushButton#btnSmall { padding: 2px 8px; font-size: 11px; border-radius: 4px; min-width: 40px; }
QProgressBar {
    border: 1px solid #16213e; border-radius: 6px; text-align: center;
    background-color: #0f0f23; color: #00d4aa; font-weight: bold; height: 22px;
}
QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4aa, stop:1 #0f3460); border-radius: 5px; }
QTableWidget {
    background-color: #0f0f23; alternate-background-color: #16213e; gridline-color: #1a1a2e;
    border: 1px solid #16213e; border-radius: 6px; font-size: 12px; color: #ddd;
}
QHeaderView::section {
    background-color: #16213e; color: #00d4aa; border: none;
    padding: 6px; font-weight: bold; font-size: 12px;
}
QTableWidget::item:selected { background-color: #0f3460; }
QTabWidget::pane { border: 1px solid #16213e; border-radius: 6px; background: #1a1a2e; }
QTabBar::tab {
    background: #16213e; color: #aaa; padding: 8px 20px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold;
}
QTabBar::tab:selected { background: #0f3460; color: #00d4aa; }
QTabBar::tab:hover { background: #0f3460; color: #fff; }
QComboBox {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 4px; padding: 4px 8px;
}
QComboBox QAbstractItemView { background-color: #16213e; color: #e0e0e0; selection-background-color: #0f3460; }
QSlider::groove:horizontal { height: 6px; background: #16213e; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -4px 0;
    background: #00d4aa; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #00d4aa; border-radius: 3px; }
QSplitter::handle { background-color: #16213e; width: 3px; }
"""


# ── Click-to-Seek Slider ────────────────────────────────────────────
class ClickableSlider(QSlider):
    """QSlider 서브클래스: 아무 곳이나 클릭하면 그 위치로 즉시 이동 + 드래그 지원"""
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)  # 드래그 상태 유지를 위해 항상 호출

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mouseMoveEvent(event)


# ── ROI Filter Tuning Dialog ────────────────────────────────────────
class RoiFilterDialog(QDialog):
    """ROI별 밝기/대비/이진화 필터를 슬라이더로 조정하고 실시간 미리보기하는 다이얼로그 (v44 포팅)"""
    def __init__(self, roi_type_label, roi_gray_img, current_filter=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"ROI Filter Tuning ({roi_type_label})")
        self.setMinimumSize(520, 450)
        self.roi_gray = roi_gray_img
        self.result_filter = None  # None = 취소, dict = 확정

        ref = {"brightness": 0, "contrast": 100, "threshold_on": 0, "block_size": 11}
        if current_filter:
            ref.update(current_filter)

        layout = QVBoxLayout(self)

        # 미리보기 영역
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 130)
        self.preview_label.setStyleSheet("background-color: #222; border: 1px solid #333; border-radius: 4px;")
        layout.addWidget(self.preview_label)

        # 밝기 슬라이더
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("밝기:"))
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(ref["brightness"])
        self.lbl_brightness_val = QLabel(str(ref["brightness"]))
        self.lbl_brightness_val.setFixedWidth(35)
        self.slider_brightness.valueChanged.connect(lambda v: (self.lbl_brightness_val.setText(str(v)), self._update_preview()))
        row1.addWidget(self.slider_brightness, 1)
        row1.addWidget(self.lbl_brightness_val)
        layout.addLayout(row1)

        # 대비 슬라이더
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("대비%:"))
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(50, 200)
        self.slider_contrast.setValue(ref["contrast"])
        self.lbl_contrast_val = QLabel(str(ref["contrast"]))
        self.lbl_contrast_val.setFixedWidth(35)
        self.slider_contrast.valueChanged.connect(lambda v: (self.lbl_contrast_val.setText(str(v)), self._update_preview()))
        row2.addWidget(self.slider_contrast, 1)
        row2.addWidget(self.lbl_contrast_val)
        layout.addLayout(row2)

        # 이진화 체크 + 블록크기
        row3 = QHBoxLayout()
        self.chk_threshold = QCheckBox("이진화")
        self.chk_threshold.setChecked(bool(ref["threshold_on"]))
        self.chk_threshold.stateChanged.connect(lambda: self._update_preview())
        row3.addWidget(self.chk_threshold)
        row3.addWidget(QLabel("블록크기:"))
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 31)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(ref["block_size"])
        self.spin_block.valueChanged.connect(lambda: self._update_preview())
        row3.addWidget(self.spin_block)
        row3.addStretch()
        layout.addLayout(row3)

        # 버튼
        btn_row = QHBoxLayout()
        btn_confirm = QPushButton("✅ 확정 (이 필터 저장)")
        btn_confirm.setStyleSheet("background-color: #0f3460; color: #00d4aa; font-weight: bold; padding: 8px;")
        btn_confirm.clicked.connect(self._on_confirm)
        btn_skip = QPushButton("⏭ 스킵 (필터 없음)")
        btn_skip.setStyleSheet("padding: 8px;")
        btn_skip.clicked.connect(self.reject)
        btn_row.addWidget(btn_confirm)
        btn_row.addWidget(btn_skip)
        layout.addLayout(btn_row)

        self._update_preview()

    def _build_filter(self):
        bs = self.spin_block.value()
        if bs % 2 == 0:
            bs += 1
        return {
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value(),
            "threshold_on": 1 if self.chk_threshold.isChecked() else 0,
            "block_size": max(3, bs)
        }

    @staticmethod
    def apply_roi_filter(gray_img, filter_dict):
        """필터를 적용한 이미지 반환 (extractor에서도 동일 로직 사용)"""
        if filter_dict is None or not filter_dict:
            return gray_img
        img = gray_img.astype(np.float64)
        contrast = filter_dict.get("contrast", 100) / 100.0
        brightness = filter_dict.get("brightness", 0)
        img = img * contrast + brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        if filter_dict.get("threshold_on", 0):
            bs = filter_dict.get("block_size", 11)
            if bs % 2 == 0:
                bs += 1
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 2)
        return img

    def _update_preview(self):
        fd = self._build_filter()
        out = RoiFilterDialog.apply_roi_filter(self.roi_gray.copy(), fd)
        h, w = out.shape[:2]
        canvas_w, canvas_h = 480, 120
        scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1), 5.0)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if len(resized.shape) == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = resized
        h2, w2, ch = rgb.shape
        qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _on_confirm(self):
        self.result_filter = self._build_filter()
        self.accept()


# ── Time Input with Auto-Formatting ─────────────────────────────────
class TimeLineEdit(QLineEdit):
    """숫자만 입력하면 자동으로 hh:mm:ss 형태로 변환하는 입력칸"""
    def __init__(self, parent=None):
        super().__init__("00:00:00", parent)
        self.setFixedWidth(72)
        self.setAlignment(Qt.AlignCenter)
        self.setPlaceholderText("hh:mm:ss")
        self.setStyleSheet("background:#16213e; color:#e0e0e0; border:1px solid #0f3460; border-radius:4px; padding:4px;")
        self.editingFinished.connect(self._auto_format)

    def _auto_format(self):
        text = self.text().strip()
        # 이미 올바른 형식이면 무시
        if len(text) == 8 and text[2] == ':' and text[5] == ':':
            return
        # 숫자만 추출
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            self.setText("00:00:00")
            return
        # 6자리까지 왼쪽 0으로 채움 (예: "10" -> "000010", "012530" -> "012530")
        digits = digits.zfill(6)
        # 6자리 초과 시 앞부분만 사용
        if len(digits) > 6:
            digits = digits[:6]
        hh = digits[0:2]
        mm = digits[2:4]
        ss = digits[4:6]
        # 범위 보정
        hh = str(min(int(hh), 99)).zfill(2)
        mm = str(min(int(mm), 59)).zfill(2)
        ss = str(min(int(ss), 59)).zfill(2)
        self.setText(f"{hh}:{mm}:{ss}")

    def reset(self):
        self.setText("00:00:00")


class VideoPlayer(QWidget):
    """Left panel: embedded video player with seek bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("No Video Loaded")
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 270)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # OCR 미리보기 레이블
        self.lbl_ocr_preview = QLabel("")
        self.lbl_ocr_preview.setObjectName("ocrPreview")
        self.lbl_ocr_preview.setStyleSheet("color: #00e0ff; font-size: 11px; padding: 2px 6px; background: rgba(0,0,0,0.5); border-radius: 4px;")
        self.lbl_ocr_preview.setFixedHeight(22)
        self.lbl_ocr_preview.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_ocr_preview.hide()

        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle_play)

        self.slider = ClickableSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.seek)

        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        self.lbl_time.setObjectName("timeLabel")
        self.lbl_time.setFixedWidth(130)

        self.btn_step_back = QPushButton("◀")
        self.btn_step_back.setFixedSize(28, 24)
        self.btn_step_back.setStyleSheet("font-size: 9px; padding: 0px;")
        self.btn_step_back.setToolTip("1 Frame Back")
        self.btn_step_back.setAutoRepeat(True)
        self.btn_step_back.setAutoRepeatDelay(300)
        self.btn_step_back.setAutoRepeatInterval(100)
        self.btn_step_back.clicked.connect(self.step_backward)

        self.btn_step_fwd = QPushButton("▶")
        self.btn_step_fwd.setFixedSize(28, 24)
        self.btn_step_fwd.setStyleSheet("font-size: 9px; padding: 0px;")
        self.btn_step_fwd.setToolTip("1 Frame Forward")
        self.btn_step_fwd.setAutoRepeat(True)
        self.btn_step_fwd.setAutoRepeatDelay(300)
        self.btn_step_fwd.setAutoRepeatInterval(100)
        self.btn_step_fwd.clicked.connect(self.step_forward)

        # 프레임/초 단위 전환 토글
        self.step_mode = "frame"  # "frame" or "second"
        self.btn_step_mode = QPushButton("F")
        self.btn_step_mode.setFixedSize(28, 24)
        self.btn_step_mode.setStyleSheet("font-size: 10px; font-weight: bold; padding: 0px; color: #00d4aa;")
        self.btn_step_mode.setToolTip("Step Mode: Frame (click to toggle)")
        self.btn_step_mode.clicked.connect(self._toggle_step_mode)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_step_back)
        ctrl_layout.addWidget(self.btn_step_fwd)
        ctrl_layout.addWidget(self.btn_step_mode)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.lbl_time)

        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.lbl_ocr_preview)
        layout.addLayout(ctrl_layout)

        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

        # OCR 미리보기용 속성 (MainWindow에서 설정)
        self.ocr_reader = None
        self._ocr_roi_bal = None
        self._ocr_roi_win = None
        self._ocr_enabled = False  # step/seek 시에만 True

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.video_label.setText("Failed to open video")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.show_frame(0)

    def show_frame(self, frame_idx):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_cv_frame(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
            self.update_time_label(frame_idx)
            # OCR 미리보기
            if self._ocr_enabled and self._ocr_roi_bal is not None:
                if self.ocr_reader is None:
                    self._lazy_init_ocr_reader()
                if self.ocr_reader is not None:
                    self._run_ocr_preview(frame)
                self._ocr_enabled = False

    def _lazy_init_ocr_reader(self):
        """EasyOCR reader lazy-init (최초 프레임 이동 시)"""
        try:
            import easyocr, torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.lbl_ocr_preview.setText("  OCR ▸ Initializing reader...")
            self.lbl_ocr_preview.show()
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            self.ocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
        except Exception:
            self.ocr_reader = None

    def display_cv_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def update_time_label(self, frame_idx):
        cur = frame_idx / self.fps
        tot = self.total_frames / self.fps
        self.lbl_time.setText(f"{self.fmt(cur)} / {self.fmt(tot)}")

    @staticmethod
    def fmt(sec):
        return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"

    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶")
        else:
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.timer.start(interval)
            self.playing = True
            self.btn_play.setText("⏸")

    def next_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶")
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.display_cv_frame(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.update_time_label(pos)

    def seek(self, pos):
        self._ocr_enabled = True
        self.show_frame(pos)

    def _toggle_step_mode(self):
        """프레임/초 단위 전환"""
        if self.step_mode == "frame":
            self.step_mode = "second"
            self.btn_step_mode.setText("S")
            self.btn_step_mode.setToolTip("Step Mode: Second (click to toggle)")
            self.btn_step_mode.setStyleSheet("font-size: 10px; font-weight: bold; padding: 0px; color: #ffb347;")
            self.btn_step_back.setToolTip("1 Second Back")
            self.btn_step_fwd.setToolTip("1 Second Forward")
        else:
            self.step_mode = "frame"
            self.btn_step_mode.setText("F")
            self.btn_step_mode.setToolTip("Step Mode: Frame (click to toggle)")
            self.btn_step_mode.setStyleSheet("font-size: 10px; font-weight: bold; padding: 0px; color: #00d4aa;")
            self.btn_step_back.setToolTip("1 Frame Back")
            self.btn_step_fwd.setToolTip("1 Frame Forward")

    def step_backward(self):
        """뒤로 이동 (프레임 또는 초 단위)"""
        if not self.cap: return
        self._ocr_enabled = True
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self.step_mode == "second":
            step = max(1, int(self.fps))
        else:
            step = 2  # read() advances by 1, so go back 2 for 1-frame step
        target = max(0, cur - step)
        self.show_frame(target)

    def step_forward(self):
        """앞으로 이동 (프레임 또는 초 단위)"""
        if not self.cap: return
        self._ocr_enabled = True
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self.step_mode == "second":
            step = max(1, int(self.fps))
        else:
            step = 1  # read() will advance, so cur is already next frame
        target = min(self.total_frames - 1, cur - 1 + step)
        self.show_frame(target)

    def _ocr_read_value(self, roi, frame):
        """ROI 영역에서 숫자 OCR 수행"""
        import re as _re
        if roi is None or self.ocr_reader is None:
            return None
        try:
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            res = self.ocr_reader.readtext(thresh, detail=0)
            text = " ".join(res).strip() if res else ""
            if not text:
                return None
            char_map = {'O':'0', 'o':'0', 'l':'1', 'i':'1', 'I':'1',
                        'S':'5', 's':'5', 'B':'8', 'Z':'2', 'z':'2', 'G':'6', 'g':'9'}
            text = "".join([char_map.get(c, c) for c in text])
            clean = _re.sub(r'[^\d.]+', '', text)
            parts = clean.split('.')
            if len(parts) > 1:
                last_part = parts[-1]
                if len(last_part) == 3:
                    clean = "".join(parts)
                else:
                    clean = "".join(parts[:-1]) + '.' + parts[-1]
            if not clean or clean == '.':
                return None
            return float(clean)
        except Exception:
            return None

    def _run_ocr_preview(self, frame):
        """현재 프레임에서 Balance/Win OCR 미리보기 실행 및 추출된 이벤트 연동"""
        bal = self._ocr_read_value(self._ocr_roi_bal, frame)
        win = self._ocr_read_value(self._ocr_roi_win, frame)
        bal_str = f"{bal:,.2f}" if bal is not None else "N/A"
        win_str = f"{win:,.2f}" if win is not None else "N/A"
        
        evt_str = ""
        if hasattr(self, 'get_event_log') and self.get_event_log:
            cur_sec = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) / self.fps if self.cap else 0
            evt = self.get_event_log(cur_sec)
            if evt:
                evt_str = f"  |  Event(Log): {evt}"
                
        self.lbl_ocr_preview.setText(f"  OCR ▸ Balance: {bal_str}  |  Win: {win_str}{evt_str}")
        self.lbl_ocr_preview.show()

    def seek_to_time(self, time_str):
        """Jump to hh:mm:ss position."""
        parts = time_str.split(':')
        if len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            frame = int(secs * self.fps)
            self.show_frame(min(frame, self.total_frames - 1))

    def cleanup(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.video_label.setText("No Video Loaded")
        self.slider.setValue(0)
        self.slider.setRange(0, 0)
        self.lbl_time.setText("00:00:00 / 00:00:00")
        self.playing = False
        self.btn_play.setText("▶")


class GraphTab(QWidget):
    """Graph tab with selectable X/Y axes from table data."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X Axis:"))
        self.combo_x = QComboBox()
        self.combo_x.addItems(["Spin #", "Bet", "Balance", "Win"])
        ctrl.addWidget(self.combo_x)
        ctrl.addWidget(QLabel("Y Axis:"))
        self.combo_y = QComboBox()
        self.combo_y.addItems(["Balance", "Win", "Bet", "Spin #"])
        ctrl.addWidget(self.combo_y)
        self.btn_refresh = QPushButton("Refresh Chart")
        self.btn_refresh.setObjectName("btnAccent")
        ctrl.addWidget(self.btn_refresh)
        ctrl.addStretch()

        self.figure = Figure(facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(ctrl)
        layout.addWidget(self.canvas, 1)
        self.data_rows = []

        self.btn_refresh.clicked.connect(self.refresh_chart)
        self.combo_x.currentIndexChanged.connect(self.refresh_chart)
        self.combo_y.currentIndexChanged.connect(self.refresh_chart)

    def set_data(self, rows):
        self.data_rows = rows
        self.refresh_chart()

    def refresh_chart(self):
        self.figure.clear()
        if not self.data_rows:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='#ccc')
        ax.xaxis.label.set_color('#ccc')
        ax.yaxis.label.set_color('#ccc')
        for spine in ax.spines.values():
            spine.set_color('#16213e')

        col_map = {"Spin #": 0, "Bet": 2, "Balance": 3, "Win": 4}
        x_key = self.combo_x.currentText()
        y_key = self.combo_y.currentText()
        x_idx = col_map.get(x_key, 0)
        y_idx = col_map.get(y_key, 4)

        xs, ys = [], []
        for r in self.data_rows:
            try:
                xv = float(str(r[x_idx]).replace(',', ''))
                yv = float(str(r[y_idx]).replace(',', ''))
                xs.append(xv)
                ys.append(yv)
            except (ValueError, IndexError):
                continue

        ax.plot(xs, ys, color='#00d4aa', linewidth=1.5, marker='o', markersize=3, alpha=0.8)
        ax.set_xlabel(x_key, fontsize=11)
        ax.set_ylabel(y_key, fontsize=11)
        ax.set_title(f"{y_key} vs {x_key}", color='#00d4aa', fontsize=13, fontweight='bold')
        ax.grid(True, color='#16213e', alpha=0.5)

        if len(xs) < 50:
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', color='#aaa', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎰 Slot Game Data Extractor")
        self.resize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 8, 12, 8)

        # ROI / state
        self.roi_event = None  # Event detection ROI
        self.roi_bal_filter = None  # Balance OCR 전처리 필터 dict
        self.roi_win_filter = None  # Win OCR 전처리 필터 dict
        root_layout.setSpacing(8)

        # ── Title ──
        title = QLabel("🎰 Slot Game Data Extractor")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        # ── Toolbar (프로젝트/영상/감지 관련만) ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_select = QPushButton("📂 Select Video")
        self.btn_select.clicked.connect(self.select_file)
        self.btn_roi = QPushButton("🎯 Select ROI")
        self.btn_roi.clicked.connect(self.select_roi)
        self.btn_roi.setEnabled(False)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btnAccent")
        self.btn_start.clicked.connect(self.start_extraction)
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.clicked.connect(self.stop_extraction)
        self.btn_stop.setEnabled(False)
        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_reset.clicked.connect(self.reset_all)

        # Project Save/Load
        self.btn_save_proj = QPushButton("💾 Save Project")
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_proj = QPushButton("📂 Load Project")
        self.btn_load_proj.clicked.connect(self.load_project)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setStyleSheet("color:#888; font-style:italic;")

        # Start time input (자동 포맷팅 적용)
        lbl_st = QLabel("Start:")
        lbl_st.setFixedWidth(36)
        self.txt_start_time = TimeLineEdit()

        # CLIP Threshold input
        lbl_th = QLabel("CLIP Thresh:")
        self.spin_clip_th = QDoubleSpinBox()
        self.spin_clip_th.setRange(0.1, 0.99)
        self.spin_clip_th.setSingleStep(0.05)
        self.spin_clip_th.setValue(0.5)
        self.spin_clip_th.setFixedWidth(55)

        # Balance Filter input
        lbl_bal_filter = QLabel("Bal. Filter:")
        self.txt_bal_filter = QLineEdit("1,000,000")
        self.txt_bal_filter.setFixedWidth(80)
        self.txt_bal_filter.setToolTip("Ignore abrupt balance drops/gains >= this amount")

        # Fixed Bet input
        lbl_fixed_bet = QLabel("Fixed Bet:")
        self.txt_fixed_bet = QLineEdit("")
        self.txt_fixed_bet.setFixedWidth(80)
        self.txt_fixed_bet.setPlaceholderText("e.g. 10,000")
        self.txt_fixed_bet.setToolTip("고정 배팅액. 입력 시 모든 스핀의 Bet을 이 값으로 강제 적용")

        # Stability Threshold input (안정 감지 %, 낮을수록 민감)
        lbl_stab = QLabel("Stab%:")
        self.spin_stability = QDoubleSpinBox()
        self.spin_stability.setRange(0.0, 50.0)
        self.spin_stability.setSingleStep(0.1)
        self.spin_stability.setValue(0.5)
        self.spin_stability.setDecimals(1)
        self.spin_stability.setFixedWidth(55)
        self.spin_stability.setToolTip("ROI 픽셀 변화율(%). 0=변화감지 OFF(매 프레임 OCR). 낮을수록 민감")

        for b in [self.btn_select, self.btn_roi, self.btn_start, self.btn_stop,
                  self.btn_save_proj, self.btn_load_proj, self.btn_reset]:
            b.setMinimumHeight(36)
            toolbar.addWidget(b)
        toolbar.addWidget(lbl_st)
        toolbar.addWidget(self.txt_start_time)
        toolbar.addWidget(lbl_th)
        toolbar.addWidget(self.spin_clip_th)
        toolbar.addWidget(lbl_bal_filter)
        toolbar.addWidget(self.txt_bal_filter)
        toolbar.addWidget(lbl_fixed_bet)
        toolbar.addWidget(self.txt_fixed_bet)
        toolbar.addWidget(lbl_stab)
        toolbar.addWidget(self.spin_stability)

        # Drop Only Spin 옵션
        self.chk_drop_only = QCheckBox("Drop Only")
        self.chk_drop_only.setToolTip("ON: Balance 하락 시에만 스핀 카운트 (롤업 안착은 baseline 갱신만)")
        toolbar.addWidget(self.chk_drop_only)

        toolbar.addWidget(self.lbl_file, 1)
        root_layout.addLayout(toolbar)

        # ── Status + Progress ──
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setObjectName("statusLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        status_row.addWidget(self.lbl_status, 1)
        status_row.addWidget(self.progress_bar, 2)
        root_layout.addLayout(status_row)

        # ── Main Content: Splitter (Video | Data) ──
        splitter = QSplitter(Qt.Horizontal)

        # Left: Video Player
        self.player = VideoPlayer()
        self.player.get_event_log = self.get_event_at_time
        splitter.addWidget(self.player)

        # Right: Tabs (Data / Graph)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Data Tab
        data_tab = QWidget()
        dt_layout = QVBoxLayout(data_tab)
        dt_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(["Spin #", "Time", "Duration", "Bet", "Balance", "Win", "Δ Balance", "Δ Bal+Bet", "Event Type", ""])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self.on_item_changed)
        dt_layout.addWidget(self.table)

        # ── Data 하단 버튼 바 (Import/Export/Row관리/리셋) ──
        data_btn_bar = QHBoxLayout()
        data_btn_bar.setSpacing(6)

        self.btn_add_row = QPushButton("➕ Add Row")
        self.btn_add_row.clicked.connect(self.add_empty_row)
        self.btn_del_row = QPushButton("➖ Delete Row")
        self.btn_del_row.clicked.connect(self.delete_selected_row)
        self.btn_clear_data = QPushButton("🗑 Clear Data")
        self.btn_clear_data.setObjectName("btnDanger")
        self.btn_clear_data.clicked.connect(self.clear_table_data)
        self.btn_import = QPushButton("📥 Import")
        self.btn_import.clicked.connect(self.import_excel)
        self.btn_export = QPushButton("💾 Export")
        self.btn_export.clicked.connect(self.export_excel)
        self.btn_export.setEnabled(False)

        for b in [self.btn_add_row, self.btn_del_row, self.btn_clear_data,
                  self.btn_import, self.btn_export]:
            b.setMinimumHeight(30)
            data_btn_bar.addWidget(b)
        data_btn_bar.addStretch()

        dt_layout.addLayout(data_btn_bar)

        # ── 경과시간 표시 ──
        elapsed_bar = QHBoxLayout()
        elapsed_bar.addStretch()
        self.lbl_elapsed = QLabel("⏱ 00:00:00")
        self.lbl_elapsed.setStyleSheet("color: #aaa; font-size: 13px; padding-right: 8px;")
        elapsed_bar.addWidget(self.lbl_elapsed)
        dt_layout.addLayout(elapsed_bar)

        self.tabs.addTab(data_tab, "📊 Data")

        # Graph Tab
        self.graph_tab = GraphTab()
        self.tabs.addTab(self.graph_tab, "📈 Graph")

        # ── Event Labels Tab ──
        event_tab = QWidget()
        event_layout = QVBoxLayout(event_tab)
        event_layout.setContentsMargins(6, 6, 6, 6)
        event_layout.setSpacing(6)

        event_layout.addWidget(QLabel("CLIP 감지 이벤트 목록  (Name = 결과에 표시될 이름,  CLIP Prompt = 영어 설명)"))

        self.event_table = QTableWidget()
        self.event_table.setColumnCount(2)
        self.event_table.setHorizontalHeaderLabels(["Name", "CLIP Prompt"])
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.event_table.setAlternatingRowColors(True)
        event_layout.addWidget(self.event_table)

        # 기본 이벤트 데이터 로드
        self._load_default_events()

        evt_btn_bar = QHBoxLayout()
        evt_btn_bar.setSpacing(6)
        btn_add_evt = QPushButton("➕ Add")
        btn_add_evt.clicked.connect(self._add_event_row)
        btn_del_evt = QPushButton("➖ Delete")
        btn_del_evt.clicked.connect(self._del_event_row)
        btn_reset_evt = QPushButton("🔄 Reset Default")
        btn_reset_evt.clicked.connect(self._load_default_events)
        for b in [btn_add_evt, btn_del_evt, btn_reset_evt]:
            b.setMinimumHeight(28)
            evt_btn_bar.addWidget(b)
        evt_btn_bar.addStretch()
        event_layout.addLayout(evt_btn_bar)

        self.tabs.addTab(event_tab, "🎯 Events")
        
        # ── Raw Data Tab ──
        raw_tab = QWidget()
        raw_layout = QVBoxLayout(raw_tab)
        raw_layout.setContentsMargins(6, 6, 6, 6)
        
        from PyQt5.QtWidgets import QPlainTextEdit
        self.txt_raw_log = QPlainTextEdit()
        self.txt_raw_log.setReadOnly(True)
        self.txt_raw_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #0b1120; color: #a5b4fc;")
        
        btn_clear_raw = QPushButton("🗑 Clear Log")
        btn_clear_raw.setObjectName("btnSmall")
        btn_clear_raw.setFixedWidth(80)
        btn_clear_raw.clicked.connect(self.txt_raw_log.clear)
        
        raw_layout.addWidget(btn_clear_raw, alignment=Qt.AlignRight)
        raw_layout.addWidget(self.txt_raw_log)
        
        self.tabs.addTab(raw_tab, "🧾 RawData")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)

        root_layout.addWidget(splitter, 1)

        # ── State ──
        self.video_path = None
        self.roi_bal = None
        self.roi_win = None
        self.extractor_thread = None
        self.elapsed_time = ""
        self.prev_balance = None
        self.data_rows = []
        self._preview_reader = None  # EasyOCR reader for video preview (lazy-init)

    # ── File Selection ──
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_path = path
            self.lbl_file.setText(os.path.basename(path))
            self.btn_roi.setEnabled(True)
            self.btn_start.setEnabled(False)
            self.roi_bal = None
            self.roi_win = None
            self.lbl_status.setText("Video loaded → Select ROI")
            self.player.load_video(path)

    # ── ROI Selection ──
    def select_roi(self):
        if not self.video_path:
            return
        self.lbl_status.setText("Opening ROI selector...")
        QApplication.processEvents()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # ── ROI 프레임 선택 다이얼로그 ──
        frame_dialog = QDialog(self)
        frame_dialog.setWindowTitle("Select Frame for ROI")
        frame_dialog.setMinimumSize(700, 500)
        dlg_layout = QVBoxLayout(frame_dialog)

        # 프레임 표시 레이블
        frame_label = QLabel("Loading...")
        frame_label.setAlignment(Qt.AlignCenter)
        frame_label.setMinimumSize(640, 360)
        frame_label.setStyleSheet("background-color: #0f0f23; border: 2px solid #16213e; border-radius: 8px;")
        dlg_layout.addWidget(frame_label, 1)

        # Seek Bar + 시간 레이블
        seek_layout = QHBoxLayout()
        frame_slider = ClickableSlider(Qt.Horizontal)
        frame_slider.setRange(0, max(0, total_frames - 1))
        frame_slider.setValue(0)
        time_lbl = QLabel(f"00:00:00 / {VideoPlayer.fmt(total_frames / fps)}")
        time_lbl.setFixedWidth(130)
        time_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        seek_layout.addWidget(frame_slider)
        seek_layout.addWidget(time_lbl)
        dlg_layout.addLayout(seek_layout)

        # 확인 버튼
        btn_confirm = QPushButton("Use this frame for ROI selection")
        btn_confirm.setMinimumHeight(32)
        btn_confirm.setStyleSheet("font-weight: bold; background-color: #0f3460; color: #00d4aa;")
        btn_confirm.clicked.connect(frame_dialog.accept)
        dlg_layout.addWidget(btn_confirm)

        # 프레임 표시 함수
        selected_frame = [None]
        def show_dialog_frame(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                selected_frame[0] = f.copy()
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                frame_label.setPixmap(pixmap)
                sec = idx / fps
                tot = total_frames / fps
                time_lbl.setText(f"{VideoPlayer.fmt(sec)} / {VideoPlayer.fmt(tot)}")

        frame_slider.valueChanged.connect(show_dialog_frame)
        show_dialog_frame(0)  # 첫 프레임 표시

        result = frame_dialog.exec_()
        if result != QDialog.Accepted or selected_frame[0] is None:
            self.lbl_status.setText("ROI cancelled.")
            cap.release()
            return

        frame = selected_frame[0]
        cap.release()

        # ── 2단계 줌(확대) ROI 선택 헬퍼 함수 ──
        def select_roi_with_zoom(frame, window_title="ROI Selection", use_zoom=True) -> tuple[int, int, int, int] | None:
            h, w = frame.shape[:2]
            max_h = 750 
            scale = 1.0
            
            if h > max_h:
                scale = max_h / h
                f_display = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                r1 = cv2.selectROI(f"{window_title} - Step 1", f_display, False)
                cv2.destroyWindow(f"{window_title} - Step 1")
                if r1[2] <= 0: return None
                # 원본 좌표로 역산
                r1 = (int(r1[0] / scale), int(r1[1] / scale), int(r1[2] / scale), int(r1[3] / scale))
            else:
                r1 = cv2.selectROI(f"{window_title} - Step 1", frame, False)
                cv2.destroyWindow(f"{window_title} - Step 1")
                if r1[2] <= 0: return None

            x1, y1, w1, h1 = r1
            if not use_zoom:
                return (x1, y1, w1, h1)

            # 2단계: 선택 영역을 5배 확대하여 정밀 선택
            x1, y1, w1, h1 = r1
            roi_crop = frame[y1:y1 + h1, x1:x1 + w1]
            if roi_crop.size == 0: return None
            
            # 5배 확대 (Lanczos4 보간법 사용으로 계단현상 방지)
            zoom_img = cv2.resize(roi_crop, None, fx=5, fy=5, interpolation=cv2.INTER_LANCZOS4)
            
            # CLAHE 적용을 통한 빛 번짐 개선 및 국소 대비 향상
            lab = cv2.cvtColor(zoom_img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            zoom_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # 약한 Sharpening(언샤프 마스킹)으로 경계선 뚜렷하게
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            zoom_img = cv2.filter2D(zoom_img, -1, kernel)
                         
            r2 = cv2.selectROI(f"{window_title} - Zoom (5x)", zoom_img, False)
            cv2.destroyWindow(f"{window_title} - Zoom (5x)")
            
            if r2[2] <= 0: return None
            
            # 최종 좌표 계산 (원본 프레임 기준)
            final_x = x1 + int(r2[0] / 5)
            final_y = y1 + int(r2[1] / 5)
            final_w = int(r2[2] / 5)
            final_h = int(r2[3] / 5)
            
            return (final_x, final_y, final_w, final_h)

        # ── cv2 ROI 선택 (선택된 프레임 사용) ──
        roi_bal = select_roi_with_zoom(frame, "Select BALANCE Region")
        if not roi_bal:
            self.lbl_status.setText("ROI cancelled.")
            return

        self.roi_bal = roi_bal

        roi_win = select_roi_with_zoom(frame, "Select WIN Region (or ESC to skip)", use_zoom=True)
        if not roi_win:
            self.roi_win = None
        else:
            self.roi_win = roi_win
            
        roi_event = select_roi_with_zoom(frame, "Select EVENT Region (or ESC to skip)", use_zoom=False)
        if not roi_event:
            self.roi_event = None
        else:
            self.roi_event = roi_event

        # ── Balance ROI 필터 튜닝 다이얼로그 ──
        bx, by, bw, bh = roi_bal
        bal_crop_gray = cv2.cvtColor(frame[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
        bal_dlg = RoiFilterDialog("Balance", bal_crop_gray, self.roi_bal_filter, self)
        if bal_dlg.exec_() == QDialog.Accepted:
            self.roi_bal_filter = bal_dlg.result_filter
        else:
            self.roi_bal_filter = None

        if self.roi_win is None:
            self.roi_win_filter = None
            self.btn_start.setEnabled(True)
            self.lbl_status.setText(f"ROI set ✓ (Balance{', Event' if self.roi_event else ''}) → Press Start")
        else:
            # ── Win ROI 필터 튜닝 다이얼로그 ──
            wx2, wy2, ww2, wh2 = self.roi_win
            win_crop_gray = cv2.cvtColor(frame[wy2:wy2+wh2, wx2:wx2+ww2], cv2.COLOR_BGR2GRAY)
            win_dlg = RoiFilterDialog("Win", win_crop_gray, self.roi_win_filter, self)
            if win_dlg.exec_() == QDialog.Accepted:
                self.roi_win_filter = win_dlg.result_filter
            else:
                self.roi_win_filter = None
            self.btn_start.setEnabled(True)
            self.lbl_status.setText(f"ROI set ✓ (Bal, Win{', Event' if self.roi_event else ''}) → Press Start")

        # VideoPlayer에 ROI 전달 및 EasyOCR lazy-init
        self.player._ocr_roi_bal = self.roi_bal
        self.player._ocr_roi_win = self.roi_win
        if self._preview_reader is None:
            try:
                import easyocr, torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.lbl_status.setText("Initializing OCR preview reader...")
                QApplication.processEvents()
                self._preview_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
                self.lbl_status.setText("ROI set ✓ + OCR Preview ready → Press Start")
            except Exception:
                self._preview_reader = None
        self.player.ocr_reader = self._preview_reader

    # ── Extraction (이어서 감지 지원) ──
    def start_extraction(self):
        if not self.video_path or not self.roi_bal:
            return

        self.btn_select.setEnabled(False)
        self.btn_roi.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)

        # ★ 기존 데이터를 지우지 않고 이어서 감지 (테이블/data_rows 유지)
        # prev_balance는 마지막 행의 Balance 값으로 유지
        self.progress_bar.setValue(0)

        # Parse start time
        start_frame = 0
        st_text = self.txt_start_time.text().strip()
        if st_text and st_text != "00:00:00":
            try:
                parts = st_text.split(':')
                secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                cap_tmp = cv2.VideoCapture(self.video_path)
                fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
                cap_tmp.release()
                start_frame = int(secs * fps_tmp)
            except Exception:
                start_frame = 0

        # 이어서 감지: 현재 spin 번호를 extractor에 전달
        current_spin = self.table.rowCount()

        # Parse Balance Filter
        try:
            bf_str = self.txt_bal_filter.text().replace(',', '').strip()
            bal_filter_val = float(bf_str) if bf_str else None
            if bal_filter_val == 0.0:
                bal_filter_val = None
        except ValueError:
            bal_filter_val = None

        # Parse Fixed Bet
        try:
            fb_str = self.txt_fixed_bet.text().replace(',', '').strip()
            fixed_bet_val = float(fb_str) if fb_str else None
        except ValueError:
            fixed_bet_val = None

        self.extractor_thread = ExtractorThread(
            self.video_path, self.roi_bal, self.roi_win, self.roi_event, start_frame,
            clip_threshold=self.spin_clip_th.value(),
            event_entries=self._get_event_entries(),
            bal_filter=bal_filter_val,
            fixed_bet=fixed_bet_val,
            roi_bal_filter=self.roi_bal_filter,
            roi_win_filter=self.roi_win_filter,
            stability_pct=self.spin_stability.value(),
            drop_only_spin=self.chk_drop_only.isChecked()
        )
        self.extractor_thread.progress_signal.connect(self.update_progress)
        self.extractor_thread.data_signal.connect(
            lambda spin, t, b, w, bal, evt: self.add_row(spin + current_spin, t, b, w, bal, evt))
        self.extractor_thread.status_signal.connect(self.update_status)
        self.extractor_thread.finished_signal.connect(self.extraction_finished)
        self.extractor_thread.error_signal.connect(self.extraction_error)
        self.extractor_thread.elapsed_signal.connect(self.on_elapsed)
        self.extractor_thread.raw_log_signal.connect(self.on_raw_log)
        self.extractor_thread.start()

        # 경과시간 타이머 시작
        self._elapsed_start = time.time()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)
        self._elapsed_timer.start(1000)
        self.lbl_elapsed.setText("⏱ 00:00:00")

    def stop_extraction(self):
        if self.extractor_thread:
            self.extractor_thread.stop()
            self.lbl_status.setText("Stopping...")
            # 경과시간 타이머 정지
            if hasattr(self, '_elapsed_timer') and self._elapsed_timer:
                self._elapsed_timer.stop()
            # 정지 후 UI 복원
            self.btn_select.setEnabled(True)
            self.btn_roi.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if self.table.rowCount() > 0:
                self.btn_export.setEnabled(True)

    # ── Item Changed (Auto Formatting & Data Sync) ──
    def on_item_changed(self, item):
        self.table.blockSignals(True)
        col = item.column()
        text = item.text().replace(',', '')

        # Auto comma formatting for numeric columns (Bet, Balance, Win)
        if col in [3, 4, 5]:
            try:
                val = float(text)
                item.setText(f"{val:,.0f}")
            except ValueError:
                pass

        # Sync with data_rows for graph
        self.sync_data_from_table()
        self.table.blockSignals(False)

    def sync_data_from_table(self):
        was_blocked = self.table.signalsBlocked()
        if not was_blocked:
            self.table.blockSignals(True)

        self.data_rows = []
        prev_bal = None
        for row in range(self.table.rowCount()):
            # Event Type 행과 무관하게 Spin#가 존재하면 스핀 행으로 취급
            evt_item = self.table.item(row, 8)
            event_type = evt_item.text().strip() if evt_item else ""
            spin_item = self.table.item(row, 0)
            spin_text = spin_item.text().strip() if spin_item else ""
            is_event_row = not bool(spin_text)

            row_data = []
            for col in range(6):  # Spin#, Time, Duration, Bet, Balance, Win
                it = self.table.item(row, col)
                val_str = it.text().replace(',', '') if it else "0"
                if col in (1, 2):  # Time and Duration are strings
                    row_data.append(val_str)
                else:
                    try: row_data.append(float(val_str))
                    except: row_data.append(0.0)

            # Event 전용 행은 Delta 계산에서 제외
            if is_event_row:
                delta = 0.0
                delta_plus_bet = 0.0
                # Delta/Bal+Bet 셀 비우기
                d_item = self.table.item(row, 6)
                if d_item: d_item.setText("")
                dbp_item = self.table.item(row, 7)
                if dbp_item: dbp_item.setText("")
            else:
                # 스핀 행: Delta 계산 및 Duration(경과시간) 계산 자동 복원
                current_bal = row_data[4]  # Balance
                current_bet = row_data[3]  # Bet
                delta = current_bal - prev_bal if prev_bal is not None else 0.0
                prev_bal = current_bal
                delta_plus_bet = delta + current_bet  # Δ Bal+Bet = 실질 수익

                # "이전" 스핀의 행 번호를 찾아 Duration 업데이트
                if row > 0:
                    for prev_row in range(row - 1, -1, -1):
                        prev_spin_item = self.table.item(prev_row, 0)
                        if prev_spin_item and prev_spin_item.text().strip():
                            try:
                                p_time = self.table.item(prev_row, 1).text()
                                c_time = row_data[1]  # current time
                                p_parts = p_time.split(':')
                                c_parts = c_time.split(':')
                                p_sec = int(p_parts[0])*3600 + int(p_parts[1])*60 + int(p_parts[2])
                                c_sec = int(c_parts[0])*3600 + int(c_parts[1])*60 + int(c_parts[2])
                                e_sec = c_sec - p_sec
                                if e_sec < 0: e_sec = 0
                                dur_str = f"{e_sec // 60:02d}:{e_sec % 60:02d}"
                                dur_item = self.table.item(prev_row, 2)
                                if dur_item: dur_item.setText(dur_str)
                                # data_rows가 있다면 업데이트
                                if len(self.data_rows) > prev_row and len(self.data_rows[prev_row]) > 2:
                                    self.data_rows[prev_row][2] = dur_str
                            except Exception:
                                pass
                            break

                # Update Delta Balance UI
                delta_item = self.table.item(row, 6)
                if delta_item:
                    delta_item.setText(f"{delta:+,.0f}")
                    if delta > 0: delta_item.setForeground(Qt.green)
                    elif delta < 0: delta_item.setForeground(Qt.red)
                    else: delta_item.setForeground(Qt.gray)

                # Δ Bal+Bet → 이전 스핀 행에 기록
                # 현재 행의 Bal+Bet 셀은 비움
                dbp_item = self.table.item(row, 7)
                if dbp_item:
                    dbp_item.setText("")
                if row > 0:
                    for prev_row in range(row - 1, -1, -1):
                        prev_spin_item = self.table.item(prev_row, 0)
                        if prev_spin_item and prev_spin_item.text().strip():
                            prev_dbp = self.table.item(prev_row, 7)
                            if prev_dbp:
                                prev_dbp.setText(f"{delta_plus_bet:+,.0f}")
                                if delta_plus_bet > 0: prev_dbp.setForeground(Qt.green)
                                elif delta_plus_bet < 0: prev_dbp.setForeground(Qt.red)
                                else: prev_dbp.setForeground(Qt.gray)
                            break

            # data_rows: [Spin#, Time, Duration, Bet, Balance, Win, Delta, DeltaPlusBet, EventType]
            row_data.append(delta)
            row_data.append(delta_plus_bet)
            row_data.append(event_type)
            self.data_rows.append(row_data)

        if not was_blocked:
            self.table.blockSignals(False)

        self.graph_tab.set_data(self.data_rows)

    # ── Row Management ──
    def add_empty_row(self):
        """선택된 행 아래에 빈 행 추가. 선택 없으면 맨 아래."""
        self.table.blockSignals(True)
        selected = self.table.currentRow()
        if selected < 0:
            insert_pos = self.table.rowCount()
        else:
            insert_pos = selected + 1

        self.table.insertRow(insert_pos)
        self.table.setItem(insert_pos, 0, QTableWidgetItem("0"))
        self.table.setItem(insert_pos, 1, QTableWidgetItem("00:00:00"))
        self.table.setItem(insert_pos, 2, QTableWidgetItem(""))
        self.table.setItem(insert_pos, 3, QTableWidgetItem("0"))
        self.table.setItem(insert_pos, 4, QTableWidgetItem("0"))
        self.table.setItem(insert_pos, 5, QTableWidgetItem("0"))
        self.table.setItem(insert_pos, 6, QTableWidgetItem("+0"))
        self.table.setItem(insert_pos, 7, QTableWidgetItem("+0"))
        self.table.setItem(insert_pos, 8, QTableWidgetItem(""))

        btn = QPushButton("Go")
        btn.setObjectName("btnSmall")
        btn.clicked.connect(lambda checked, ts="00:00:00": self.player.seek_to_time(ts))
        self.table.setCellWidget(insert_pos, 9, btn)

        self._renumber_spins()
        self.table.blockSignals(False)
        self.sync_data_from_table()

    def delete_selected_row(self):
        """선택된 행 삭제 후 Spin 번호 재정렬."""
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "삭제할 행을 먼저 선택해 주세요.")
            return

        self.table.blockSignals(True)
        self.table.removeRow(selected)
        self._renumber_spins()
        self.table.blockSignals(False)
        self.sync_data_from_table()

    def _renumber_spins(self):
        """모든 행의 Spin 번호를 1부터 순서대로 재설정. Event 전용 행(Spin이 기존에 없던 행)은 건너뜀."""
        spin_num = 0
        for row in range(self.table.rowCount()):
            # Spin # 열 (column 0) 이 비어있다면 이벤트 전용 행으로 간주
            spin_item = self.table.item(row, 0)
            if not spin_item or not spin_item.text().strip():
                if spin_item:
                    spin_item.setText("")
                else:
                    self.table.setItem(row, 0, QTableWidgetItem(""))
            else:
                spin_num += 1
                item = self.table.item(row, 0)
                if item:
                    item.setText(str(spin_num))
                else:
                    self.table.setItem(row, 0, QTableWidgetItem(str(spin_num)))

    def clear_table_data(self):
        """데이터 테이블 전체를 리셋."""
        reply = QMessageBox.question(self, "Confirm", "모든 데이터를 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            self.data_rows = []
            self.prev_balance = None
            self.table.blockSignals(False)
            self.graph_tab.set_data([])
            self.btn_export.setEnabled(False)
            self.lbl_status.setText("Data cleared.")

    # ── Slots ──
    def update_progress(self, v):
        self.progress_bar.setValue(v)

    def update_status(self, msg):
        self.lbl_status.setText(msg)

    def on_elapsed(self, t):
        self.elapsed_time = t

    def on_raw_log(self, msg):
        try:
            self.txt_raw_log.appendPlainText(str(msg))
            # Scroll to bottom automatically
            scrollbar = self.txt_raw_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            print(f"RawData append error: {e}")

    def _update_elapsed_label(self):
        if hasattr(self, '_elapsed_start') and self._elapsed_start:
            elapsed = time.time() - self._elapsed_start
            eh = int(elapsed // 3600)
            em = int((elapsed % 3600) // 60)
            es = int(elapsed % 60)
            self.lbl_elapsed.setText(f"⏱ {eh:02d}:{em:02d}:{es:02d}")

    def add_row(self, spin, time_str, bet, win, balance, event_type=""):
        # Check if scrollbar is at the bottom before inserting
        sb = self.table.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 1

        is_event_only = (spin == -1)

        # Δ Balance & Δ Bal+Bet calculation
        if is_event_only:
            delta = 0.0
            delta_str = ""
            dbp = 0.0
            dbp_str = ""
        else:
            delta = balance - self.prev_balance if self.prev_balance is not None else 0.0
            self.prev_balance = balance
            delta_str = f"{delta:+,.0f}"  # +1,000 or -500
            dbp = delta + bet  # Δ Balance + Bet = 실질 수익
            dbp_str = f"{dbp:+,.0f}"

        self.table.blockSignals(True)

        row = self.table.rowCount()
        self.table.insertRow(row)

        # Calculate Duration (elapsed time to next spin) and update the previous spin row
        if time_str:
            try:
                parts = time_str.split(':')
                current_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                if spin != -1:
                    if hasattr(self, 'prev_spin_sec') and self.prev_spin_sec is not None:
                        e_sec = current_sec - self.prev_spin_sec
                        if e_sec < 0: e_sec = 0
                        duration_str = f"{e_sec // 60:02d}:{e_sec % 60:02d}"
                        
                        # Find the previous spin row to update its Duration (column 2)
                        for prev_row in range(row - 1, -1, -1):
                            prev_spin = self.table.item(prev_row, 0)
                            if prev_spin and prev_spin.text().strip():
                                self.table.setItem(prev_row, 2, QTableWidgetItem(duration_str))
                                self.data_rows[prev_row][2] = duration_str
                                break
                    self.prev_spin_sec = current_sec
            except Exception:
                pass
        
        if is_event_only:
            self.table.setItem(row, 0, QTableWidgetItem(""))
            self.table.setItem(row, 1, QTableWidgetItem(time_str))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            self.table.setItem(row, 4, QTableWidgetItem(""))
            self.table.setItem(row, 5, QTableWidgetItem(""))
            self.table.setItem(row, 6, QTableWidgetItem(""))
            self.table.setItem(row, 7, QTableWidgetItem(""))
        else:
            self.table.setItem(row, 0, QTableWidgetItem(str(spin)))
            self.table.setItem(row, 1, QTableWidgetItem(time_str))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(f"{bet:,.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{balance:,.0f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{win:,.0f}"))

            # Δ Balance cell with color
            delta_item = QTableWidgetItem(delta_str)
            if delta > 0:
                delta_item.setForeground(Qt.green)
            elif delta < 0:
                delta_item.setForeground(Qt.red)
            self.table.setItem(row, 6, delta_item)

            # Δ Bal+Bet → 이전 스핀 행에 기록 (현재 스핀이 아니라 이전 스핀의 "결과 수익")
            # 현재 행은 빈 셀로 유지
            self.table.setItem(row, 7, QTableWidgetItem(""))
            if dbp_str and row > 0:
                # 이전 스핀 행 찾기 (이벤트 전용 행 건너뛰기)
                for prev_row in range(row - 1, -1, -1):
                    prev_spin = self.table.item(prev_row, 0)
                    # 이벤트 전용 행이 아닌 일반 스핀 행 찾기
                    if prev_spin and prev_spin.text().strip():
                        dbp_item = QTableWidgetItem(dbp_str)
                        if dbp > 0:
                            dbp_item.setForeground(Qt.green)
                        elif dbp < 0:
                            dbp_item.setForeground(Qt.red)
                        self.table.setItem(prev_row, 7, dbp_item)
                        break

        # Event Type cell with highlight (column 8)
        event_item = QTableWidgetItem(event_type if event_type else "")
        if event_type:
            event_item.setForeground(Qt.yellow)
        self.table.setItem(row, 8, event_item)

        # "Go" button to jump video to this time (column 9)
        btn = QPushButton("Go")
        btn.setObjectName("btnSmall")
        btn.clicked.connect(lambda checked, ts=time_str: self.player.seek_to_time(ts))
        self.table.setCellWidget(row, 9, btn)

        # Only auto-scroll if user was already at the bottom
        if at_bottom:
            self.table.scrollToBottom()

        # data_rows: [Spin#, Time, Duration, Bet, Balance, Win, Delta, DeltaPlusBet, EventType]
        self.data_rows.append([spin if not is_event_only else "", time_str, "",
                               bet if not is_event_only else "", 
                               balance if not is_event_only else "", 
                               win if not is_event_only else "", 
                               delta if not is_event_only else "", 
                               dbp if not is_event_only else "",
                               event_type])
        self.table.blockSignals(False)

    def get_event_at_time(self, frame_time_sec):
        """저장된 테이블 데이터 중 현재 시간과 1.5초 이내에 발생한 이벤트 문구를 반환합니다."""
        event_texts = []
        for r in self.data_rows:
            if len(r) > 8 and r[8]:  # event_type 존재 (index 8)
                parts = str(r[1]).split(':')
                if len(parts) == 3:
                    try:
                        event_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        if abs(event_sec - frame_time_sec) <= 1.5:
                            if r[8] not in event_texts:
                                event_texts.append(r[8])
                    except ValueError:
                        pass
        return " / ".join(event_texts) if event_texts else ""

    def extraction_finished(self):
        # 경과시간 타이머 정지
        if hasattr(self, '_elapsed_timer') and self._elapsed_timer:
            self._elapsed_timer.stop()
        self.btn_select.setEnabled(True)
        self.btn_roi.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(100)
        elapsed_msg = f" (Time: {self.elapsed_time})" if self.elapsed_time else ""
        self.lbl_status.setText(f"✅ Extraction Complete!{elapsed_msg}")
        self.graph_tab.set_data(self.data_rows)

    def extraction_error(self, msg):
        self.btn_select.setEnabled(True)
        self.btn_roi.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(self.table.rowCount() > 0)
        self.lbl_status.setText("❌ Error occurred.")
        QMessageBox.critical(self, "Error", msg)

    def _get_event_entries(self):
        """이벤트 테이블의 모든 엔트리를 리스트로 반환합니다."""
        entries = []
        for row in range(self.event_table.rowCount()):
            name_item = self.event_table.item(row, 0)
            prompt_item = self.event_table.item(row, 1)
            if name_item and prompt_item:
                name = name_item.text().strip()
                prompt = prompt_item.text().strip()
                if name and prompt:
                    entries.append({"name": name, "prompt": prompt})
        return entries

    def _load_events_from_list(self, event_list):
        """저장된 이벤트 리스트를 이벤트 테이블에 로드합니다."""
        self.event_table.blockSignals(True)
        self.event_table.setRowCount(0)
        for i, entry in enumerate(event_list):
            self.event_table.insertRow(i)
            self.event_table.setItem(i, 0, QTableWidgetItem(entry.get("name", "")))
            self.event_table.setItem(i, 1, QTableWidgetItem(entry.get("prompt", "")))
        self.event_table.blockSignals(False)

    def _load_default_events(self):
        """기본 이벤트 목록(DEFAULT_EVENT_ENTRIES)을 로드합니다."""
        self._load_events_from_list(DEFAULT_EVENT_ENTRIES)

    def _add_event_row(self):
        """이벤트 테이블에 빈 행을 추가합니다."""
        row = self.event_table.rowCount()
        self.event_table.insertRow(row)
        self.event_table.setItem(row, 0, QTableWidgetItem(""))
        self.event_table.setItem(row, 1, QTableWidgetItem(""))

    def _del_event_row(self):
        """선택된 이벤트 행을 삭제합니다."""
        selected = self.event_table.currentRow()
        if selected >= 0:
            self.event_table.removeRow(selected)

    def reset_all(self):
        """전체 리셋: 파일·ROI·데이터·프리뷰·이벤트 테이블 초기화."""
        # Stop thread if running
        if self.extractor_thread and self.extractor_thread.isRunning():
            self.extractor_thread.stop()
            self.extractor_thread.wait(3000)
        self.player.cleanup()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        self.data_rows = []
        self.progress_bar.setValue(0)
        self.video_path = None
        self.roi_bal = None
        self.roi_win = None
        self.roi_event = None
        self.roi_bal_filter = None
        self.roi_win_filter = None
        self.elapsed_time = ""
        self.prev_balance = None
        self.prev_spin_sec = None
        self.lbl_file.setText("No file selected")
        self.lbl_status.setText("Ready")
        self.txt_start_time.reset()  # ★ 시작시간도 리셋
        self.spin_clip_th.setValue(0.5)
        self.btn_select.setEnabled(True)
        self.btn_roi.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.graph_tab.set_data([])

    # ── Export ──
    def export_excel(self):
        if not self.data_rows:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "slot_extracted_data.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        headers = ["Spin #", "Time", "Duration", "Bet", "Balance", "Win", "Δ Balance", "Δ Bal+Bet", "Event Type"]

        rows = []
        for i, r in enumerate(self.data_rows):
            try:
                spin_val = str(int(r[0])) if r[0] != "" and r[0] != 0.0 else ""
            except (ValueError, TypeError):
                spin_val = str(r[0]) if r[0] else ""
            time_val = str(r[1]) if len(r) > 1 else ""
            elapsed_val = str(r[2]) if len(r) > 2 else ""
            event = str(r[8]) if len(r) > 8 else ""

            # 숫자 컨럼은 float 그대로 (Excel 숫자 형식)
            try:
                bet_val = float(r[3]) if r[3] != "" else None
                bal_val = float(r[4]) if r[4] != "" else None
                win_val = float(r[5]) if r[5] != "" else None
                delta_val = float(r[6]) if r[6] != "" else None
                dbp_val = float(r[7]) if r[7] != "" else None
            except (ValueError, TypeError):
                bet_val = r[3] if len(r) > 3 else None
                bal_val = r[4] if len(r) > 4 else None
                win_val = r[5] if len(r) > 5 else None
                delta_val = r[6] if len(r) > 6 else None
                dbp_val = r[7] if len(r) > 7 else None

            rows.append([spin_val, time_val, elapsed_val, bet_val, bal_val, win_val, delta_val, dbp_val, event])

        df = pd.DataFrame(rows, columns=headers)

        # Append elapsed time row at the very bottom
        if self.elapsed_time:
            # Shift Total Time text column appropriately
            elapsed_row = pd.DataFrame([["", "", "", "", "Total Time:", self.elapsed_time, "", "", ""]], columns=headers)
            df = pd.concat([df, elapsed_row], ignore_index=True)

        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
                ws = writer.sheets['Data']
                # 숫자 컨럼에 쉼표 포맷 적용 (Bet~Δ Bal+Bet = col D~H)
                for col_letter in ['D', 'E', 'F', 'G', 'H']:
                    for cell in ws[col_letter][1:]:  # 헤더 제외
                        if cell.value is not None and isinstance(cell.value, (int, float)):
                            cell.number_format = '#,##0'
            QMessageBox.information(self, "Export", f"Data exported successfully.\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Excel", "", "Excel Files (*.xlsx)")
        if not path:
            return

        try:
            df = pd.read_excel(path)
            # Basic validation
            expected_cols = ["Spin #", "Time", "Bet", "Balance", "Win"]
            for col in expected_cols:
                if col not in df.columns:
                    QMessageBox.warning(self, "Warning", f"Missing column: {col}")
                    return

            self.table.blockSignals(True)
            self.table.setRowCount(0)
            self.data_rows = []
            self.prev_spin_sec = None

            for _, row in df.iterrows():
                # Skip total time row if exists
                if "Total Time" in str(row.values) or "Total Time:" in str(row.values):
                    continue

                spin_val = row.get("Spin #", "")
                if pd.isna(spin_val) or str(spin_val).strip() == "":
                    spin = -1
                else:
                    try:
                        spin = int(spin_val)
                    except ValueError:
                        spin = -1

                time_str = str(row.get("Time", ""))
                if time_str == "nan": time_str = ""
                
                def safe_float(val):
                    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "nan": return 0.0
                    try: return float(str(val).replace(',', ''))
                    except ValueError: return 0.0

                bet = safe_float(row.get("Bet"))
                balance = safe_float(row.get("Balance"))
                win = safe_float(row.get("Win"))
                event_type = str(row.get("Event Type", ""))
                if event_type == "nan": event_type = ""

                self.add_row(spin, time_str, bet, win, balance, event_type)

            self.table.blockSignals(False)
            self.sync_data_from_table()
            self.btn_export.setEnabled(True)
            self.lbl_status.setText(f"✅ Imported {self.table.rowCount()} rows.")

        except Exception as e:
            self.table.blockSignals(False)
            QMessageBox.critical(self, "Import Error", f"Failed to import excel: {str(e)}")

    # ── Project Save/Load ──
    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.sge", "Slot Game Project (*.sge)")
        if not path:
            return

        data = {
            "video_path": self.video_path,
            "roi_bal": list(self.roi_bal) if self.roi_bal else None,
            "roi_win": list(self.roi_win) if self.roi_win else None,
            "roi_event": list(self.roi_event) if self.roi_event else None,
            "roi_bal_filter": self.roi_bal_filter,
            "roi_win_filter": self.roi_win_filter,
            "data_rows": self.data_rows,
            "start_time": self.txt_start_time.text(),
            "clip_threshold": self.spin_clip_th.value(),
            "event_entries": self._get_event_entries()
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Save", "Project saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Slot Game Project (*.sge)")
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.reset_all()

            self.video_path = data.get("video_path")
            if self.video_path and os.path.exists(self.video_path):
                self.player.load_video(self.video_path)
                self.lbl_file.setText(os.path.basename(self.video_path))
                self.btn_roi.setEnabled(True)

            rb = data.get("roi_bal")
            rw = data.get("roi_win")
            re_ = data.get("roi_event")
            if rb: self.roi_bal = tuple(rb)
            if rw: self.roi_win = tuple(rw)
            if re_: self.roi_event = tuple(re_)
            self.roi_bal_filter = data.get("roi_bal_filter")
            self.roi_win_filter = data.get("roi_win_filter")

            if self.roi_bal:
                self.btn_start.setEnabled(True)
                self.player._ocr_roi_bal = self.roi_bal
                self.player._ocr_roi_win = self.roi_win

            st = data.get("start_time", "00:00:00")
            self.txt_start_time.setText(st)
            
            ct = data.get("clip_threshold", 0.5)
            self.spin_clip_th.setValue(ct)

            # 이벤트 엔트리 복원
            saved_events = data.get("event_entries", None)
            if saved_events:
                self._load_events_from_list(saved_events)
            else:
                self._load_default_events()

            # ★ Load table data — 빠른 일괄 삽입 (UI 업데이트 억제)
            rows = data.get("data_rows", [])
            self.table.blockSignals(True)
            self.table.setUpdatesEnabled(False)  # 화면 갱신 억제
            self.prev_balance = None

            for r in rows:
                if len(r) >= 5:
                    spin_val = r[0]
                    spin = int(spin_val) if spin_val != "" else -1
                    time_str = str(r[1])
                    elapsed_str = ""

                    if len(r) >= 9:
                        # 새 구조 (9개 요소)
                        elapsed_str = str(r[2])
                        bet = float(r[3]) if r[3] != "" else 0.0
                        bal = float(r[4]) if r[4] != "" else 0.0
                        win = float(r[5]) if r[5] != "" else 0.0
                        event_type = str(r[8]) if len(r) > 8 else ""
                    elif len(r) == 8:
                        # 과도기 구조 (8개 요소)
                        bet = float(r[2]) if r[2] != "" else 0.0
                        bal = float(r[3]) if r[3] != "" else 0.0
                        win = float(r[4]) if r[4] != "" else 0.0
                        event_type = str(r[7]) if len(r) > 7 else ""
                    else:
                        # 기존 레거시 구조 (7개 요소, win/balance 뒤바뀜)
                        bet = float(r[2]) if r[2] != "" else 0.0
                        win = float(r[3]) if r[3] != "" else 0.0
                        bal = float(r[4]) if r[4] != "" else 0.0
                        event_type = str(r[6]) if len(r) > 6 else ""

                    is_evt = (spin == -1)
                    row_idx = self.table.rowCount()
                    self.table.insertRow(row_idx)

                    if is_evt:
                        self.table.setItem(row_idx, 0, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 1, QTableWidgetItem(time_str))
                        self.table.setItem(row_idx, 2, QTableWidgetItem(elapsed_str))
                        self.table.setItem(row_idx, 3, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 4, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 5, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 6, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 7, QTableWidgetItem(""))
                    else:
                        self.table.setItem(row_idx, 0, QTableWidgetItem(str(spin)))
                        self.table.setItem(row_idx, 1, QTableWidgetItem(time_str))
                        self.table.setItem(row_idx, 2, QTableWidgetItem(elapsed_str))
                        self.table.setItem(row_idx, 3, QTableWidgetItem(f"{bet:,.0f}"))
                        self.table.setItem(row_idx, 4, QTableWidgetItem(f"{bal:,.0f}"))
                        self.table.setItem(row_idx, 5, QTableWidgetItem(f"{win:,.0f}"))
                        self.table.setItem(row_idx, 6, QTableWidgetItem(""))
                        self.table.setItem(row_idx, 7, QTableWidgetItem(""))

                    event_item = QTableWidgetItem(event_type)
                    if event_type:
                        event_item.setForeground(Qt.yellow)
                    self.table.setItem(row_idx, 8, event_item)

                    btn = QPushButton("Go")
                    btn.setObjectName("btnSmall")
                    btn.clicked.connect(lambda checked, ts=time_str: self.player.seek_to_time(ts))
                    self.table.setCellWidget(row_idx, 9, btn)

            self.table.setUpdatesEnabled(True)  # 화면 갱신 재개
            self.table.blockSignals(False)
            
            # 동기화하면서 Duration, Delta 등이 옛날 포맷에서 로드된 것까지 완벽하게 재계산됨.
            self.sync_data_from_table()

            if self.table.rowCount() > 0:
                self.btn_export.setEnabled(True)

            self.lbl_status.setText("✅ Project loaded successfully.")

        except Exception as e:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
