import sys
import os
import json
import re
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, filtfilt, resample
import subprocess
from pathlib import Path
import multiprocessing
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush


# ==========================================
# 1. 백엔드 분석 및 실시간 스레드
# ==========================================
class DemucsAnalysisThread(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, audio_path, sr=22050):
        super().__init__()
        self.audio_path = audio_path
        self.sr = sr

    def load_audio(self, file_path):
        data, sr = sf.read(file_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != self.sr:
            num_samples = int(len(data) * self.sr / sr)
            data = resample(data, num_samples)
        return data.astype(np.float32)

    def _lowpass_filter(self, data, cutoff=10000, fs=22050, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def run(self):
        try:
            self.status_signal.emit("ANALYZING...")
            output_dir = Path.home() / "Documents" / "GuitarToneTuner_separated"
            output_dir.mkdir(parents=True, exist_ok=True)

            from demucs.separate import main as demucs_main

            demucs_args = [
                "-n", "htdemucs_6s",
                "--two-stems", "guitar",
                "-o", str(output_dir),
                self.audio_path
            ]

            demucs_main(demucs_args)

            all_files = list(output_dir.rglob("*.*"))
            guitar_files = [
                p for p in all_files 
                if not p.name.startswith("no_") and p.suffix.lower() in [".wav", ".flac", ".mp3"]
            ]

            if not guitar_files:
                raise FileNotFoundError(f"음원 분리는 수행되었으나 결과 파일이 없습니다.\n경로: {output_dir}")

            guitar_track_path = max(guitar_files, key=lambda p: p.stat().st_mtime)

            y = self.load_audio(str(guitar_track_path))
            y_clean = self._lowpass_filter(y, cutoff=10000, fs=self.sr)

            # Drive
            frame_size, hop_size = 2048, 512
            frames = [y_clean[i:i+frame_size] for i in range(0, len(y_clean)-frame_size, hop_size)]
            rms_vals = [np.sqrt(np.mean(f**2)) for f in frames]
            mean_rms = np.mean(rms_vals) if rms_vals else 1e-6
            peak = np.max(np.abs(y_clean))
            crest_factor = 20 * np.log10(peak / (mean_rms + 1e-6))

            # EQ
            fft_vals = np.abs(np.fft.rfft(y_clean))
            freqs = np.fft.rfftfreq(len(y_clean), 1.0 / self.sr)
            tot = np.sum(fft_vals) + 1e-6
            low_r = np.sum(fft_vals[(freqs >= 100) & (freqs < 350)]) / tot
            mid_r = np.sum(fft_vals[(freqs >= 350) & (freqs < 3000)]) / tot
            high_r = np.sum(fft_vals[freqs >= 3000]) / tot

            # Modulation
            zcrs = [np.sum(np.diff(np.signbit(y_clean[i:i+2048]))) / 2048 for i in range(0, len(y_clean)-2048, 512)]
            zcr_std = np.std(zcrs) * 1000 if zcrs else 0

            target_json = {
                "drive": {"crest_factor_db": round(float(crest_factor), 2)},
                "eq": {
                    "low_ratio": round(float(low_r), 3),
                    "mid_ratio": round(float(mid_r), 3),
                    "high_ratio": round(float(high_r), 3)
                },
                "modulation": {"modulation_instability_score": round(float(zcr_std), 2)}
            }

            self.finished_signal.emit(target_json)

        except Exception as e:
            self.error_signal.emit(str(e))


class RealtimeAudioWorker(QThread):
    update_signal = pyqtSignal(float, dict)

    def __init__(self, target_json, sr=22050, buffer_duration=1.0):
        super().__init__()
        self.target = target_json
        self.sr = sr
        self.buffer_size = int(sr * buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.is_running = True

    def _lowpass_filter(self, data, cutoff=10000, fs=22050, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def extract_features(self, y):
        if np.max(np.abs(y)) < 0.01:
            return None

        y_clean = self._lowpass_filter(y, cutoff=10000, fs=self.sr)
        rms = np.sqrt(np.mean(y_clean**2)) + 1e-6
        peak = np.max(np.abs(y_clean)) + 1e-6
        crest_factor = 20 * np.log10(peak / rms)

        fft_vals = np.abs(np.fft.rfft(y_clean))
        freqs = np.fft.rfftfreq(len(y_clean), 1.0 / self.sr)
        tot = np.sum(fft_vals) + 1e-6
        low_r = np.sum(fft_vals[(freqs >= 100) & (freqs < 350)]) / tot
        mid_r = np.sum(fft_vals[(freqs >= 350) & (freqs < 3000)]) / tot
        high_r = np.sum(fft_vals[freqs >= 3000]) / tot

        return {
            "crest_factor_db": crest_factor,
            "low_ratio": low_r,
            "mid_ratio": mid_r,
            "high_ratio": high_r
        }

    def calculate_offsets(self, current):
        if current is None:
            return 0.0, {"DRIVE": 0, "BASS": 0, "MID": 0, "TREBLE": 0}

        t_drive = self.target["drive"]["crest_factor_db"]
        t_eq = self.target["eq"]

        drive_diff = np.clip((current["crest_factor_db"] - t_drive) / 10.0, -2.0, 2.0)
        low_diff = np.clip((current["low_ratio"] - t_eq["low_ratio"]) / 0.2, -2.0, 2.0)
        mid_diff = np.clip((current["mid_ratio"] - t_eq["mid_ratio"]) / 0.2, -2.0, 2.0)
        high_diff = np.clip((current["high_ratio"] - t_eq["high_ratio"]) / 0.2, -2.0, 2.0)

        offsets = {
            "DRIVE": drive_diff,
            "BASS": low_diff,
            "MID": mid_diff,
            "TREBLE": high_diff
        }

        total_err = (abs(drive_diff)*0.3) + (abs(low_diff)*0.2 + abs(mid_diff)*0.3 + abs(high_diff)*0.2)
        similarity = max(0.0, min(100.0, (1.0 - (total_err / 2.0)) * 100))

        return round(similarity, 1), offsets

    def run(self):
        def callback(indata, frames, time_info, status):
            if self.is_running:
                shift = len(indata)
                self.audio_buffer = np.roll(self.audio_buffer, -shift)
                self.audio_buffer[-shift:] = indata[:, 0]

        with sd.InputStream(samplerate=self.sr, channels=1, callback=callback):
            while self.is_running:
                self.msleep(150)
                curr_features = self.extract_features(self.audio_buffer)
                score, offsets = self.calculate_offsets(curr_features)
                self.update_signal.emit(score, offsets)

    def stop(self):
        self.is_running = False
        self.wait()


# ==========================================
# 2. 커스텀 UI 위젯 (GuitarTunerApp보다 위에 선언)
# ==========================================
class CircleGaugeButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.percentage = 0.0
        self.state_text = "FILE UPLOAD"
        self.is_analyzing = False
        self.is_tuning = False

    def set_analyzing(self, flag):
        self.is_analyzing = flag
        self.state_text = "ANALYZING..." if flag else "FILE UPLOAD"
        self.update()

    def set_score(self, score):
        self.percentage = score
        self.is_tuning = True
        self.update()

    def reset_ui(self):
        self.percentage = 0.0
        self.is_tuning = False
        self.is_analyzing = False
        self.state_text = "FILE UPLOAD"
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        size = min(width, height) - 20
        center_x, center_y = width / 2, height / 2
        radius = size / 2

        pen_bg = QPen(QColor(60, 60, 65), 8)
        painter.setPen(pen_bg)
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

        if self.is_tuning:
            pen_gauge = QPen(QColor("#9D65FC"), 8)
            painter.setPen(pen_gauge)
            angle = int((self.percentage / 100.0) * 360 * 16)
            painter.drawArc(
                int(center_x - radius), int(center_y - radius), 
                int(size), int(size), 
                90 * 16, -angle
            )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#9D65FC")))
        painter.drawEllipse(QPointF(center_x, center_y), radius - 12, radius - 12)

        painter.setPen(QPen(QColor(255, 255, 255)))
        if self.is_tuning:
            font = QFont("Arial", 36, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(self.percentage)}%")
        else:
            font = QFont("Arial", 14, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.state_text)


class MultiTunerScaleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.offsets = {"DRIVE": 0.0, "BASS": 0.0, "MID": 0.0, "TREBLE": 0.0}

    def set_offsets(self, offsets):
        self.offsets = offsets
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center_x = w / 2
        line_y = h / 2

        painter.setPen(QPen(QColor(80, 80, 90), 2))
        painter.drawLine(40, int(line_y), w - 40, int(line_y))

        painter.setPen(QPen(QColor("#9D65FC"), 3))
        painter.drawLine(int(center_x), int(line_y - 25), int(center_x), int(line_y + 25))

        max_dist = (w / 2) - 50
        colors = {
            "DRIVE": QColor("#FF5252"),
            "BASS": QColor("#4CAF50"),
            "MID": QColor("#FFC107"),
            "TREBLE": QColor("#00E5FF")
        }

        font_item = QFont("Arial", 10, QFont.Weight.Bold)
        font_arrow = QFont("Arial", 11, QFont.Weight.Bold)

        for name, val in self.offsets.items():
            x_pos = center_x + (val * max_dist)
            color = colors.get(name, QColor(255, 255, 255))

            if x_pos < 40:
                painter.setFont(font_arrow)
                painter.setPen(QPen(color))
                painter.drawText(10, int(line_y + 5), f"◀ {name}")
            elif x_pos > w - 40:
                painter.setFont(font_arrow)
                painter.setPen(QPen(color))
                painter.drawText(w - 75, int(line_y + 5), f"{name} ▶")
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x_pos, line_y), 7, 7)

                painter.setFont(font_item)
                painter.setPen(QPen(color))
                painter.drawText(int(x_pos - 25), int(line_y - 12), 50, 20, Qt.AlignmentFlag.AlignCenter, name)


# ==========================================
# 3. 메인 앱 GUI Window
# ==========================================
class GuitarTunerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tone Tuner")
        self.setFixedSize(400, 650)
        self.setStyleSheet("background-color: #000000;")

        self.target_json = None
        self.analysis_thread = None
        self.audio_worker = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 60, 20, 20)

        self.circle_btn = CircleGaugeButton()
        self.circle_btn.setFixedSize(260, 260)
        self.circle_btn.clicked.connect(self.handle_circle_click)

        circle_container = QHBoxLayout()
        circle_container.addWidget(self.circle_btn)
        layout.addLayout(circle_container)

        layout.addSpacing(40)

        self.tuner_scale = MultiTunerScaleWidget()
        self.tuner_scale.setFixedHeight(180)
        layout.addWidget(self.tuner_scale)
        
        # 초기 숨김 처리
        self.tuner_scale.hide()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def handle_circle_click(self):
        if self.circle_btn.is_tuning:
            self.stop_tuning()
            self.circle_btn.reset_ui()
            self.tuner_scale.set_offsets({"DRIVE": 0.0, "BASS": 0.0, "MID": 0.0, "TREBLE": 0.0})
            self.tuner_scale.hide()
            return

        if self.circle_btn.is_analyzing:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "레퍼런스 음원 선택", "", "Audio Files (*.mp3 *.wav *.flac *.m4a)"
        )
        if file_path:
            self.circle_btn.set_analyzing(True)
            self.tuner_scale.hide()
            
            self.analysis_thread = DemucsAnalysisThread(audio_path=file_path)
            self.analysis_thread.finished_signal.connect(self.on_analysis_finished)
            self.analysis_thread.error_signal.connect(self.on_analysis_error)
            self.analysis_thread.start()

    def on_analysis_finished(self, target_json):
        self.target_json = target_json
        self.circle_btn.is_analyzing = False
        
        # 분석 완료 시에만 하단 스케일 위젯 표출
        self.tuner_scale.show()

        self.audio_worker = RealtimeAudioWorker(target_json=self.target_json)
        self.audio_worker.update_signal.connect(self.update_gui_feedback)
        self.audio_worker.start()

    def on_analysis_error(self, err_msg):
        QMessageBox.critical(self, "분석 오류", f"음원 분석 중 오류가 발생했습니다:\n{err_msg}")
        self.circle_btn.reset_ui()
        self.tuner_scale.hide()

    def update_gui_feedback(self, score, offsets):
        self.circle_btn.set_score(score)
        self.tuner_scale.set_offsets(offsets)

    def stop_tuning(self):
        if self.audio_worker and self.audio_worker.isRunning():
            self.audio_worker.stop()
            self.audio_worker = None

    def closeEvent(self, event):
        self.stop_tuning()
        event.accept()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = GuitarTunerApp()
    window.show()
    sys.exit(app.exec())
