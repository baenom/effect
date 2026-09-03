import sys
import os
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, filtfilt, resample
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# ==========================================
# 1. 레퍼런스 음원 분석 스레드 (Demucs + DSP)
# ==========================================
class DemucsAnalysisThread(QThread):
    # 분석 상태 메시지, 완료된 Target JSON 전달 시그널
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
            data = data.mean(axis=1)  # 모노 변환
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
            self.status_signal.emit("Demucs로 기타 트랙을 분리하는 중입니다...")
            
            output_dir = Path("./separated").resolve()
            
            # 1. Demucs 명령 실행
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", "htdemucs_6s",
                "--two-stems", "guitar",
                "-o", str(output_dir),
                self.audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 2. Demucs 결과 폴더 내부에서 'guitar.wav' 파일 자동 검색
            # (폴더명이 길거나 특수문자가 있어도 감지하여 정확한 경로를 찾아냅니다)
            guitar_files = list(output_dir.rglob("guitar.wav"))

            if not guitar_files:
                raise FileNotFoundError("Demucs 분리 결과물에서 'guitar.wav' 파일을 찾을 수 없습니다.")

            # 가장 최근에 생성된 guitar.wav 파일 선택
            guitar_track_path = max(guitar_files, key=lambda p: p.stat().st_mtime)

            # 3. DSP 톤 특징 분석
            self.status_signal.emit("기타 트랙 톤 데이터 분석 중...")
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


# ==========================================
# 2. 실시간 오디오 입력 & 피드백 스레드
# ==========================================
class RealtimeAudioWorker(QThread):
    update_signal = pyqtSignal(float, list)

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

        zcrs = [np.sum(np.diff(np.signbit(y_clean[i:i+512]))) / 512 for i in range(0, len(y_clean)-512, 256)]
        zcr_std = np.std(zcrs) * 1000 if len(zcrs) > 0 else 0

        return {
            "crest_factor_db": crest_factor,
            "low_ratio": low_r,
            "mid_ratio": mid_r,
            "high_ratio": high_r,
            "modulation_score": zcr_std
        }

    def calculate_feedback(self, current):
        if current is None:
            return 0.0, ["기타를 연주하세요... (신호 대기 중)"]

        t_drive = self.target["drive"]["crest_factor_db"]
        t_eq = self.target["eq"]
        t_mod = self.target["modulation"]["modulation_instability_score"]

        drive_err = abs(current["crest_factor_db"] - t_drive) / 15.0
        low_err = abs(current["low_ratio"] - t_eq["low_ratio"]) / 0.5
        mid_err = abs(current["mid_ratio"] - t_eq["mid_ratio"]) / 0.5
        high_err = abs(current["high_ratio"] - t_eq["high_ratio"]) / 0.5
        mod_err = abs(current["modulation_score"] - t_mod) / 10.0

        total_error = (drive_err * 0.3) + (low_err * 0.15 + mid_err * 0.2 + high_err * 0.15) + (mod_err * 0.2)
        similarity = max(0.0, min(100.0, (1.0 - total_error) * 100))

        hints = []
        if current["crest_factor_db"] > t_drive + 2.5:
            hints.append("GAIN / DRIVE 올리세요 ▲")
        elif current["crest_factor_db"] < t_drive - 2.5:
            hints.append("GAIN / DRIVE 낮추세요 ▼")

        if current["low_ratio"] < t_eq["low_ratio"] - 0.08:
            hints.append("BASS(Low) 올리세요 ▲")
        elif current["low_ratio"] > t_eq["low_ratio"] + 0.08:
            hints.append("BASS(Low) 낮추세요 ▼")

        if current["mid_ratio"] < t_eq["mid_ratio"] - 0.08:
            hints.append("MIDDLE 올리세요 ▲")
        elif current["mid_ratio"] > t_eq["mid_ratio"] + 0.08:
            hints.append("MIDDLE 낮추세요 ▼")

        if current["high_ratio"] < t_eq["high_ratio"] - 0.08:
            hints.append("TREBLE(High) 올리세요 ▲")
        elif current["high_ratio"] > t_eq["high_ratio"] + 0.08:
            hints.append("TREBLE(High) 낮추세요 ▼")

        if not hints:
            hints.append("완벽합니다! 목표 톤 영역에 도달했습니다.")

        return round(similarity, 1), hints

    def run(self):
        def callback(indata, frames, time_info, status):
            if self.is_running:
                shift = len(indata)
                self.audio_buffer = np.roll(self.audio_buffer, -shift)
                self.audio_buffer[-shift:] = indata[:, 0]

        with sd.InputStream(samplerate=self.sr, channels=1, callback=callback):
            while self.is_running:
                self.msleep(200)
                curr_features = self.extract_features(self.audio_buffer)
                score, hints = self.calculate_feedback(curr_features)
                self.update_signal.emit(score, hints)

    def stop(self):
        self.is_running = False
        self.wait()


# ==========================================
# 3. 메인 GUI 앱
# ==========================================
class GuitarTunerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Tone Tuner - Audio Upload Edition")
        self.setFixedSize(520, 620)
        self.target_json = None
        self.analysis_thread = None
        self.audio_worker = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # 1. 음원 업로드 섹션
        top_layout = QHBoxLayout()
        self.btn_upload_audio = QPushButton("레퍼런스 음원 업로드")
        self.btn_upload_audio.setStyleSheet("padding: 12px; font-weight: bold; background-color: #2b5c8f; color: white; border-radius: 5px;")
        self.btn_upload_audio.clicked.connect(self.upload_reference_audio)
        
        self.lbl_file_status = QLabel("선택된 음원 없음")
        self.lbl_file_status.setStyleSheet("color: #666;")
        top_layout.addWidget(self.btn_upload_audio)
        top_layout.addWidget(self.lbl_file_status)
        main_layout.addLayout(top_layout)

        # 상태 안내 메시지 레이블
        self.lbl_analysis_status = QLabel("")
        self.lbl_analysis_status.setStyleSheet("color: #0066cc; font-weight: bold;")
        self.lbl_analysis_status.setWordWrap(True)
        main_layout.addWidget(self.lbl_analysis_status)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(line)

        # 2. 실시간 일치도 게이지
        main_layout.addWidget(QLabel("실시간 톤 일치도:"))
        self.progress_score = QProgressBar()
        self.progress_score.setRange(0, 100)
        self.progress_score.setValue(0)
        self.progress_score.setFixedHeight(35)
        self.progress_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_score.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        main_layout.addWidget(self.progress_score)

        # 3. 실시간 노브 조절 가이드
        main_layout.addWidget(QLabel("노브 조절 정도:"))
        self.lbl_hints = QLabel("레퍼런스 음원을 업로드하면 톤 분석 후 튜닝이 준비됩니다.")
        self.lbl_hints.setStyleSheet("""
            background-color: #f0f4f8; 
            border: 1px solid #d0d7de; 
            border-radius: 8px; 
            padding: 15px; 
            font-size: 14px;
            line-height: 1.5;
        """)
        self.lbl_hints.setWordWrap(True)
        main_layout.addWidget(self.lbl_hints)

        # 4. 실시간 튜닝 제어 버튼
        self.btn_toggle = QPushButton("실시간 톤 튜닝 시작")
        self.btn_toggle.setEnabled(False)
        self.btn_toggle.setFixedHeight(45)
        self.btn_toggle.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.btn_toggle.clicked.connect(self.toggle_tuning)
        main_layout.addWidget(self.btn_toggle)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def upload_reference_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "레퍼런스 음원 선택", "", "Audio Files (*.mp3 *.wav *.flac *.m4a)"
        )
        if file_path:
            filename = os.path.basename(file_path)
            self.lbl_file_status.setText(filename)
            self.btn_upload_audio.setEnabled(False)
            self.btn_toggle.setEnabled(False)

            # 분석 스레드 실행
            self.analysis_thread = DemucsAnalysisThread(audio_path=file_path)
            self.analysis_thread.status_signal.connect(self.lbl_analysis_status.setText)
            self.analysis_thread.finished_signal.connect(self.on_analysis_finished)
            self.analysis_thread.error_signal.connect(self.on_analysis_error)
            self.analysis_thread.start()

    def on_analysis_finished(self, target_json):
        self.target_json = target_json
        self.lbl_analysis_status.setText("음원 분석 완료! 튜닝을 시작할 수 있습니다.")
        self.btn_upload_audio.setEnabled(True)
        self.btn_toggle.setEnabled(True)
        self.lbl_hints.setText("기타를 오디오 인터페이스에 연결하고 '실시간 톤 튜닝 시작' 버튼을 누르세요.")

    def on_analysis_error(self, err_msg):
        QMessageBox.critical(self, "분석 오류", f"음원 분석 중 오류가 발생했습니다:\n{err_msg}")
        self.lbl_analysis_status.setText("분석 실패")
        self.btn_upload_audio.setEnabled(True)

    def toggle_tuning(self):
        if self.audio_worker is None or not self.audio_worker.isRunning():
            self.audio_worker = RealtimeAudioWorker(target_json=self.target_json)
            self.audio_worker.update_signal.connect(self.update_gui_feedback)
            self.audio_worker.start()

            self.btn_toggle.setText("튜닝 중지")
            self.btn_toggle.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #f44336; color: white; border-radius: 5px;")
            self.btn_upload_audio.setEnabled(False)
        else:
            self.audio_worker.stop()
            self.audio_worker = None

            self.btn_toggle.setText("실시간 톤 튜닝 시작")
            self.btn_toggle.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")
            self.btn_upload_audio.setEnabled(True)

    def update_gui_feedback(self, score, hints):
        self.progress_score.setValue(int(score))
        hint_text = "\n".join([f"• {h}" for h in hints])
        self.lbl_hints.setText(hint_text)

    def closeEvent(self, event):
        if self.audio_worker and self.audio_worker.isRunning():
            self.audio_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuitarTunerApp()
    window.show()
    sys.exit(app.exec())