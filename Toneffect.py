import sys
import os
import json
import re
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, filtfilt, resample, medfilt
import subprocess
from pathlib import Path
import multiprocessing
import platform
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush


# ==========================================
# 0. 보정된 드라이브 '게인 량(THD 고조파 왜곡률)' 연산 함수
# ==========================================
def _calculate_drive_score(y, sr=22050):
    rms = np.sqrt(np.mean(y**2)) + 1e-6
    if rms < 0.005:  # 노이즈 게이트 임계값
        return 0.0

    # 1. FFT 분석
    fft_vals = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    valid_mask = (freqs >= 80) & (freqs <= 8000)
    if not np.any(valid_mask):
        return 0.0

    valid_freqs = freqs[valid_mask]
    valid_fft = fft_vals[valid_mask]

    # 2. Peak(기본음 f0 추정) 찾기
    peak_idx = np.argmax(valid_fft)
    f0 = valid_freqs[peak_idx]
    f0_amp = valid_fft[peak_idx] + 1e-6

    if f0 < 80:
        return 0.0

    # 3. 주요 고조파(2f0 ~ 6f0) 에너지 측정 (오차 대역 폭 10Hz로 압축)
    harmonics_energy = 0.0
    for h in range(2, 7):
        h_freq = f0 * h
        if h_freq > 8000:
            break
        h_mask = (freqs >= h_freq - 10) & (freqs <= h_freq + 10)
        if np.any(h_mask):
            harmonics_energy += np.max(fft_vals[h_mask])**2

    # 4. THD 산출 및 압축(Compression) 적용
    raw_thd = np.sqrt(harmonics_energy) / f0_amp
    
    # Log scale 압축을 적용하여 급격한 피킹 순간의 스파이크 수치 감쇄
    compressed_thd = np.log1p(raw_thd * 2.0)

    # Clean < 0.25, Overdrive ~ 0.6, Dist/High-gain > 1.0
    drive_score = (compressed_thd / 0.9) * 100.0

    return float(np.clip(drive_score, 0.0, 100.0))


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

    def _highpass_filter(self, data, cutoff=50, fs=22050, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def _lowpass_filter(self, data, cutoff=6000, fs=22050, order=5):
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
                raise FileNotFoundError(f"음원 분리는 완료되었으나 결과 파일이 없습니다.\n경로: {output_dir}")

            guitar_track_path = max(guitar_files, key=lambda p: p.stat().st_mtime)

            y = self.load_audio(str(guitar_track_path))
            y_hp = self._highpass_filter(y, cutoff=50, fs=self.sr)
            y_clean = self._lowpass_filter(y_hp, cutoff=6000, fs=self.sr)

            frame_length = int(self.sr * 1.0)
            hop_length = int(self.sr * 0.25)

            drive_scores = []
            low_ratios = []
            mid_ratios = []
            high_ratios = []

            for start in range(0, len(y_clean) - frame_length, hop_length):
                frame = y_clean[start:start + frame_length]
                rms = np.sqrt(np.mean(frame**2))

                if rms < 0.008:
                    continue

                d_score = _calculate_drive_score(frame, sr=self.sr)
                
                fft_vals = np.abs(np.fft.rfft(frame))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sr)
                valid_idx = (freqs >= 80) & (freqs <= 6000)
                tot = np.sum(fft_vals[valid_idx]) + 1e-6

                low_r = (np.sum(fft_vals[(freqs >= 80) & (freqs < 300)]) * 0.85) / tot
                mid_r = np.sum(fft_vals[(freqs >= 300) & (freqs < 3000)]) / tot
                high_r = (np.sum(fft_vals[(freqs >= 3000) & (freqs <= 6000)]) * 0.75) / tot

                drive_scores.append(d_score)
                low_ratios.append(low_r)
                mid_ratios.append(mid_r)
                high_ratios.append(high_r)

            if not drive_scores:
                raise ValueError("분석할 수 있는 유효한 기타 연주 구간이 음원에 존재하지 않습니다.")

            # 유효 연주 구간 중 중간~상위 구간(40%~80%)의 안정된 드라이브 평균값 사용
            sorted_indices = np.argsort(drive_scores)
            idx_start = int(len(sorted_indices) * 0.4)
            idx_end = int(len(sorted_indices) * 0.85)
            target_indices = sorted_indices[idx_start:idx_end]

            if len(target_indices) == 0:
                target_indices = sorted_indices

            target_drive = np.mean([drive_scores[i] for i in target_indices])
            target_low = np.mean([low_ratios[i] for i in target_indices])
            target_mid = np.mean([mid_ratios[i] for i in target_indices])
            target_high = np.mean([high_ratios[i] for i in target_indices])

            target_json = {
                "drive": {"drive_score": round(float(target_drive), 2)},
                "eq": {
                    "low_ratio": round(float(target_low), 3),
                    "mid_ratio": round(float(target_mid), 3),
                    "high_ratio": round(float(target_high), 3)
                },
                "modulation": {"modulation_instability_score": 0.0}
            }

            self.finished_signal.emit(target_json)

        except Exception as e:
            err_details = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(err_details)


class RealtimeAudioWorker(QThread):
    update_signal = pyqtSignal(float, dict)

    def __init__(self, target_json, sr=22050, buffer_duration=1.0, gate_threshold_db=-38.0):
        super().__init__()
        self.target = target_json
        self.sr = sr
        self.buffer_size = int(sr * buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.is_running = True
        self.gate_threshold_db = gate_threshold_db
        
        # 지수 이동 평균(EMA) 및 링 버퍼 관련 변수
        self.smoothed_drive = None
        self.smoothed_low = None
        self.smoothed_mid = None
        self.smoothed_high = None
        self.alpha = 0.25  # Smooth 계수 (낮을수록 보정이 강해짐)

    def _highpass_filter(self, data, cutoff=50, fs=22050, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def _lowpass_filter(self, data, cutoff=6000, fs=22050, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def _apply_noise_gate(self, y):
        rms = np.sqrt(np.mean(y**2)) + 1e-6
        db = 20 * np.log10(rms)
        if db < self.gate_threshold_db:
            return np.zeros_like(y)
        return y

    def extract_features(self, y):
        y_gated = self._apply_noise_gate(y)

        if np.max(np.abs(y_gated)) < 0.002:
            # 연주를 멈추면 필터 리셋
            self.smoothed_drive = None
            return None

        y_hp = self._highpass_filter(y_gated, cutoff=50, fs=self.sr)
        y_clean = self._lowpass_filter(y_hp, cutoff=6000, fs=self.sr)
        y_clean = medfilt(y_clean, kernel_size=3)

        curr_drive = _calculate_drive_score(y_clean, sr=self.sr)

        fft_vals = np.abs(np.fft.rfft(y_clean))
        freqs = np.fft.rfftfreq(len(y_clean), 1.0 / self.sr)
        
        valid_idx = (freqs >= 80) & (freqs <= 6000)
        tot = np.sum(fft_vals[valid_idx]) + 1e-6

        curr_low = (np.sum(fft_vals[(freqs >= 80) & (freqs < 300)]) * 0.85) / tot
        curr_mid = np.sum(fft_vals[(freqs >= 300) & (freqs < 3000)]) / tot
        curr_high = (np.sum(fft_vals[(freqs >= 3000) & (freqs <= 6000)]) * 0.75) / tot

        # EMA(지수 이동 평균) 필터링으로 스파이크 수치 억제
        if self.smoothed_drive is None:
            self.smoothed_drive = curr_drive
            self.smoothed_low = curr_low
            self.smoothed_mid = curr_mid
            self.smoothed_high = curr_high
        else:
            self.smoothed_drive = (self.alpha * curr_drive) + ((1 - self.alpha) * self.smoothed_drive)
            self.smoothed_low = (self.alpha * curr_low) + ((1 - self.alpha) * self.smoothed_low)
            self.smoothed_mid = (self.alpha * curr_mid) + ((1 - self.alpha) * self.smoothed_mid)
            self.smoothed_high = (self.alpha * curr_high) + ((1 - self.alpha) * self.smoothed_high)

        return {
            "drive_score": self.smoothed_drive,
            "low_ratio": self.smoothed_low,
            "mid_ratio": self.smoothed_mid,
            "high_ratio": self.smoothed_high
        }

    def calculate_offsets(self, current):
        if current is None:
            return 0.0, {"DRIVE": 0, "BASS": 0, "MID": 0, "TREBLE": 0}

        t_drive = self.target["drive"].get("drive_score", 0.0)
        t_eq = self.target["eq"]

        # 드라이브 편차 허용 오차 보정
        drive_diff = np.clip((current["drive_score"] - t_drive) / 35.0, -2.0, 2.0)
        low_diff = np.clip((current["low_ratio"] - t_eq["low_ratio"]) / 0.30, -2.0, 2.0)
        mid_diff = np.clip((current["mid_ratio"] - t_eq["mid_ratio"]) / 0.30, -2.0, 2.0)
        high_diff = np.clip((current["high_ratio"] - t_eq["high_ratio"]) / 0.30, -2.0, 2.0)

        offsets = {
            "DRIVE": round(float(drive_diff), 2),
            "BASS": round(float(low_diff), 2),
            "MID": round(float(mid_diff), 2),
            "TREBLE": round(float(high_diff), 2)
        }

        total_err = (abs(drive_diff) * 0.4) + (abs(low_diff) * 0.2 + abs(mid_diff) * 0.2 + abs(high_diff) * 0.2)
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
# 2. 커스텀 UI 위젯
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
        
        self.tuner_scale.show()

        self.audio_worker = RealtimeAudioWorker(target_json=self.target_json)
        self.audio_worker.update_signal.connect(self.update_gui_feedback)
        self.audio_worker.start()

    def on_analysis_error(self, err_msg):
        self.circle_btn.reset_ui()
        self.tuner_scale.hide()

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("분석 오류 발생")
        msg.setText("음원 분석 프로세스 중 오류가 발생했습니다.")
        msg.setDetailedText(err_msg)
        msg.exec()

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


def force_mac_microphone_permission():
    if platform.system() == "Darwin":
        cmd = """
        osascript -e 'tell application "System Events" to do shell script "echo request_mic"'
        """
        try:
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    force_mac_microphone_permission()
    window = GuitarTunerApp()
    window.show()
    sys.exit(app.exec())
