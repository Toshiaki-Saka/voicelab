"""
apps/transcribe_demo.py

Python GUI: file picker → STFT spectrogram → Whisper transcription → WAV playback
Requirements: pip install PyQt6 openai-whisper scipy numpy matplotlib
Run:          python apps/transcribe_demo.py [--autorun]
"""

import sys
import struct
import wave as wav_mod
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft, resample_poly

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QFileDialog,
    QComboBox, QSizePolicy, QFrame,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── WAV I/O ─────────────────────────────────────────────────────────────────

def load_wav_float(path: str) -> tuple[np.ndarray, int, int]:
    """Return (float32 samples, sample_rate, num_channels)."""
    with wav_mod.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    return arr, sr, channels


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels == 2:
        return (samples[0::2] + samples[1::2]) * 0.5
    return samples


def resample_to_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16_000:
        return mono
    g = gcd(16_000, sr)
    return resample_poly(mono, 16_000 // g, sr // g).astype(np.float32)


# ── Spectrogram ──────────────────────────────────────────────────────────────

def compute_spectrogram(wav_path: str):
    """Return (mag_db [freq×time], freqs_hz, times_s) limited to 8 kHz."""
    samples, sr, channels = load_wav_float(wav_path)
    mono = to_mono(samples, channels)

    N, HOP = 1024, 256
    freqs, times, Zxx = scipy_stft(mono, fs=sr, nperseg=N, noverlap=N - HOP,
                                    window="hann")
    mag = np.abs(Zxx)

    mask = freqs <= 8_000.0
    mag_db = 20.0 * np.log10(mag[mask] + 1e-8)
    return mag_db, freqs[mask], times


# ── Background worker ─────────────────────────────────────────────────────────

class TranscribeWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, wav_path: str, model_name: str):
        super().__init__()
        self.wav_path = wav_path
        self.model_name = model_name

    def run(self):
        try:
            import whisper  # openai-whisper

            samples, sr, channels = load_wav_float(self.wav_path)
            mono = to_mono(samples, channels)
            pcm16k = resample_to_16k(mono, sr)

            model = whisper.load_model(self.model_name)
            result = model.transcribe(pcm16k, language="ja", fp16=False)
            text = result.get("text", "").strip()
            self.finished.emit(text or "(認識結果なし)")
        except Exception as exc:
            self.error.emit(str(exc))


# ── Spectrogram canvas ────────────────────────────────────────────────────────

class SpectrogramCanvas(FigureCanvas):
    _BG = "#0d0d0d"
    _FG = "#777777"

    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 2.2), dpi=100)
        fig.patch.set_facecolor(self._BG)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._style()
        self._placeholder()

    def _style(self):
        self.ax.set_facecolor(self._BG)
        self.ax.tick_params(colors=self._FG, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#2a2a2a")

    def _placeholder(self, msg="(ファイル読込後に表示)"):
        self.ax.clear()
        self._style()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center",
                     color="#444444", fontsize=10,
                     transform=self.ax.transAxes)
        self.draw_idle()

    def show(self, mag_db: np.ndarray, freqs: np.ndarray, times: np.ndarray):
        self.ax.clear()
        self._style()
        vmax = float(mag_db.max())
        self.ax.pcolormesh(times, freqs / 1000.0, mag_db,
                           cmap="viridis", vmin=vmax - 80.0, vmax=vmax,
                           shading="auto")
        self.ax.set_ylabel("周波数 (kHz)", color=self._FG, fontsize=8)
        self.ax.set_xlabel("時間 (s)", color=self._FG, fontsize=8)
        self.figure.tight_layout(pad=0.4)
        self.draw_idle()

    def error(self, msg: str):
        self._placeholder(f"エラー: {msg}")


# ── Main window ───────────────────────────────────────────────────────────────

_RESULT_STYLE = "background:#1a1a1a; color:#e0e0e0; font-size:15px; border:1px solid #333;"
_ERROR_STYLE  = "background:#1a1a1a; color:#ff6666; font-size:13px; border:1px solid #333;"

class MainWindow(QMainWindow):
    def __init__(self, autorun: bool = False):
        super().__init__()
        self.setWindowTitle("Voicelab Transcribe Demo")
        self.setMinimumSize(900, 660)
        self._worker: TranscribeWorker | None = None
        self._build_ui()
        if autorun:
            self._run()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(8)
        vbox.setContentsMargins(12, 12, 12, 12)

        # Header
        hdr = QLabel("Voicelab Transcribe Demo")
        hdr.setStyleSheet("color:#66ccff; font-size:16px; font-weight:bold;")
        vbox.addWidget(hdr)
        vbox.addWidget(_hline())

        # WAV row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("音声ファイル:"))
        self._wav = QLineEdit("simulation_output/my_voice.wav")
        self._wav.returnPressed.connect(self._reload)
        row1.addWidget(self._wav)
        btn_pick = QPushButton("ファイル選択")
        btn_pick.setFixedWidth(90)
        btn_pick.clicked.connect(self._pick)
        row1.addWidget(btn_pick)
        vbox.addLayout(row1)

        # Model row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Whisper モデル:"))
        self._model = QComboBox()
        self._model.addItems(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
        self._model.setCurrentText("small")
        row2.addWidget(self._model)
        row2.addStretch()
        vbox.addLayout(row2)
        vbox.addWidget(_hline())

        # Run button
        row3 = QHBoxLayout()
        self._btn = QPushButton("  文字起こし実行  ")
        self._btn.setStyleSheet("font-size:14px; padding:6px 16px;")
        self._btn.clicked.connect(self._run)
        row3.addWidget(self._btn)
        self._status = QLabel("")
        self._status.setStyleSheet("color:#888;")
        row3.addWidget(self._status)
        row3.addStretch()
        vbox.addLayout(row3)
        vbox.addWidget(_hline())

        # Result
        vbox.addWidget(QLabel("認識結果:"))
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setFixedHeight(120)
        self._result.setStyleSheet(_RESULT_STYLE)
        self._result.setPlaceholderText("(ここに認識結果が表示されます)")
        vbox.addWidget(self._result)

        # Spectrogram
        vbox.addWidget(QLabel("スペクトログラム:"))
        self._spec = SpectrogramCanvas()
        vbox.addWidget(self._spec, stretch=1)

    # ── slots ──

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "WAVファイルを選択", "",
            "WAV Files (*.wav);;All Files (*.*)")
        if path:
            self._wav.setText(path)
            self._reload()

    def _reload(self):
        self._result.clear()
        self._refresh_spec()

    def _refresh_spec(self):
        path = self._wav.text().strip()
        if not Path(path).exists():
            self._spec.error("ファイルが見つかりません")
            return
        try:
            mag_db, freqs, times = compute_spectrogram(path)
            self._spec.show(mag_db, freqs, times)
        except Exception as exc:
            self._spec.error(str(exc))

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        path = self._wav.text().strip()
        model = self._model.currentText()

        self._result.clear()
        self._result.setStyleSheet(_RESULT_STYLE)
        self._btn.setEnabled(False)
        self._status.setText("処理中...")
        self._refresh_spec()

        self._worker = TranscribeWorker(path, model)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, text: str):
        self._result.setStyleSheet(_RESULT_STYLE)
        self._result.setPlainText(text)
        self._btn.setEnabled(True)
        self._status.setText("")
        _play_wav(self._wav.text().strip())

    def _on_error(self, msg: str):
        self._result.setStyleSheet(_ERROR_STYLE)
        self._result.setPlainText(f"エラー:\n{msg}")
        self._btn.setEnabled(True)
        self._status.setText("")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#2a2a2a;")
    return f


def _play_wav(path: str):
    if not Path(path).exists():
        return
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(28, 28, 28))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(18, 18, 18))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(50, 50, 50))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 50))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(200, 200, 200))
    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    autorun = "--autorun" in sys.argv
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    win = MainWindow(autorun=autorun)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
