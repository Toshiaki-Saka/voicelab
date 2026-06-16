"""
apps/transcribe_demo.py

Python GUI: file picker → waveform + spectrogram + MFCC → Whisper transcription → WAV playback
Requirements: pip install PyQt6 openai-whisper scipy numpy matplotlib
Run:          python apps/transcribe_demo.py [--autorun]
"""

import sys
import wave as wav_mod
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft, resample_poly
from scipy.fftpack import dct as scipy_dct

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

matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]


# ── WAV I/O ──────────────────────────────────────────────────────────────────

def load_wav_float(path: str) -> tuple[np.ndarray, int, int]:
    with wav_mod.open(path, "rb") as wf:
        channels  = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr        = wf.getframerate()
        raw       = wf.readframes(wf.getnframes())
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


# ── Analysis (spectrogram + MFCC) ────────────────────────────────────────────

def compute_analysis(wav_path: str, n_mfcc: int = 13, n_mels: int = 40):
    """Return (mono, sr, mag_db, freqs_hz, times_s, mfcc [n_mfcc × n_frames])."""
    samples, sr, channels = load_wav_float(wav_path)
    mono = to_mono(samples, channels)

    N, HOP = 1024, 256
    freqs, times, Zxx = scipy_stft(mono, fs=sr, nperseg=N, noverlap=N - HOP,
                                    window="hann")
    power = np.abs(Zxx) ** 2

    # Spectrogram limited to 8 kHz
    mask   = freqs <= 8_000.0
    mag_db = 20.0 * np.log10(np.sqrt(power[mask]) + 1e-8)

    # Mel filterbank
    fmin, fmax = 0.0, float(sr) / 2.0
    mel_pts = np.linspace(2595.0 * np.log10(1.0 + fmin / 700.0),
                          2595.0 * np.log10(1.0 + fmax / 700.0),
                          n_mels + 2)
    hz_pts  = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins    = np.clip(np.floor((N + 1) * hz_pts / sr).astype(int), 0, N // 2)

    fb = np.zeros((n_mels, N // 2 + 1))
    for m in range(1, n_mels + 1):
        fs, fc, fe = bins[m - 1], bins[m], bins[m + 1]
        if fc > fs:
            fb[m - 1, fs:fc] = (np.arange(fs, fc) - fs) / (fc - fs)
        if fe > fc:
            fb[m - 1, fc:fe] = (fe - np.arange(fc, fe)) / (fe - fc)

    mfcc = scipy_dct(np.log(fb @ power + 1e-8), type=2, axis=0, norm="ortho")[:n_mfcc]

    return mono, sr, mag_db, freqs[mask], times, mfcc


# ── Background worker ─────────────────────────────────────────────────────────

class TranscribeWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, wav_path: str, model_name: str):
        super().__init__()
        self.wav_path   = wav_path
        self.model_name = model_name

    def run(self):
        try:
            import whisper
            if not Path(self.wav_path).exists():
                raise FileNotFoundError(f"ファイルが見つかりません: {self.wav_path}")
            self.status.emit(f"モデル読込中 ({self.model_name})…")
            model = whisper.load_model(self.model_name)
            self.status.emit("音声解析中…")
            samples, sr, channels = load_wav_float(self.wav_path)
            pcm16k = resample_to_16k(to_mono(samples, channels), sr)
            self.status.emit("文字起こし中…")
            result = model.transcribe(pcm16k, language="ja", fp16=False)
            self.finished.emit(result.get("text", "").strip() or "(認識結果なし)")
        except Exception as exc:
            self.error.emit(str(exc))


# ── Analysis canvas (4 panels) ────────────────────────────────────────────────

_BG   = "#0d0d0d"
_BG2  = "#111111"
_FG   = "#666666"
_EDGE = "#2a2a2a"


def _style_ax(ax, title: str):
    ax.set_facecolor(_BG2)
    ax.set_title(title, color=_FG, fontsize=8, pad=4)
    ax.tick_params(colors=_FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_EDGE)
    ax.xaxis.label.set_color(_FG)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_color(_FG)
    ax.yaxis.label.set_fontsize(7)


class AnalysisCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 6.5), dpi=100)
        fig.patch.set_facecolor(_BG)
        gs = fig.add_gridspec(3, 2,
                               hspace=0.58, wspace=0.32,
                               left=0.07, right=0.96,
                               top=0.97, bottom=0.06)
        self.ax_wave = fig.add_subplot(gs[0, :])
        self.ax_spec = fig.add_subplot(gs[1, :])
        self.ax_heat = fig.add_subplot(gs[2, 0])
        self.ax_bar  = fig.add_subplot(gs[2, 1])
        self._cbar   = None

        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._placeholder()

    def _placeholder(self, msg="(ファイル読込後に表示)"):
        for ax, title in (
            (self.ax_wave, "Waveform"),
            (self.ax_spec, "Spectrogram"),
            (self.ax_heat, "MFCC Heatmap"),
            (self.ax_bar,  "MFCC Mean ± Std"),
        ):
            ax.clear()
            _style_ax(ax, title)
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="#333333", fontsize=9, transform=ax.transAxes)
        self.draw_idle()

    def show_analysis(self, mono: np.ndarray, sr: int,
                      mag_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                      mfcc: np.ndarray):
        # Waveform
        ax = self.ax_wave
        ax.clear()
        t = np.arange(len(mono)) / sr
        ax.plot(t, mono, color="#89dceb", linewidth=0.3, alpha=0.9)
        _style_ax(ax, "Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, float(t[-1]))

        # Spectrogram
        ax = self.ax_spec
        ax.clear()
        vmax = float(mag_db.max())
        ax.pcolormesh(times, freqs / 1000.0, mag_db,
                      cmap="inferno", vmin=vmax - 80.0, vmax=vmax, shading="auto")
        _style_ax(ax, "Spectrogram  (NFFT=1024, hop=256)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")

        # MFCC heatmap (remove old colorbar first)
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        ax = self.ax_heat
        ax.clear()
        n_coeff, n_frames = mfcc.shape
        im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm",
                       extent=[0, n_frames, -0.5, n_coeff - 0.5])
        _style_ax(ax, f"MFCC Heatmap  ({n_coeff} coeff × {n_frames} frames)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("MFCC index")
        self._cbar = self.figure.colorbar(im, ax=ax, pad=0.02)
        self._cbar.ax.tick_params(labelsize=6, colors=_FG)

        # MFCC mean ± std
        ax = self.ax_bar
        ax.clear()
        means = mfcc.mean(axis=1)
        stds  = mfcc.std(axis=1)
        x = np.arange(n_coeff)
        ax.bar(x, means, color="#a6e3a1", alpha=0.8, zorder=2, label="mean")
        ax.errorbar(x, means, yerr=stds,
                    fmt="none", color="#f38ba8", capsize=3,
                    linewidth=1.2, zorder=3, label="±std")
        ax.axhline(0, color=_EDGE, linewidth=0.7)
        _style_ax(ax, "MFCC  Mean ± Std  (all frames)")
        ax.set_xlabel("MFCC index")
        ax.set_ylabel("Value")
        leg = ax.legend(fontsize=6, facecolor="#1a1a1a",
                        labelcolor=_FG, edgecolor=_EDGE)

        self.draw_idle()

    def show_error(self, msg: str):
        self._placeholder(f"エラー: {msg}")


# ── Main window ───────────────────────────────────────────────────────────────

_RESULT_STYLE = "background:#1a1a1a; color:#e0e0e0; font-size:15px; border:1px solid #333;"
_ERROR_STYLE  = "background:#1a1a1a; color:#ff6666; font-size:13px; border:1px solid #333;"


class MainWindow(QMainWindow):
    def __init__(self, autorun: bool = False):
        super().__init__()
        self.setWindowTitle("Voicelab Transcribe Demo")
        self.setMinimumSize(960, 860)
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

        hdr = QLabel("Voicelab Transcribe Demo")
        hdr.setStyleSheet("color:#66ccff; font-size:16px; font-weight:bold;")
        vbox.addWidget(hdr)
        vbox.addWidget(_hline())

        # WAV picker
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

        # Model selector
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Whisper モデル:"))
        self._model = QComboBox()
        self._model.addItems(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
        self._model.setCurrentText("base")
        row2.addWidget(self._model)
        row2.addStretch()
        vbox.addLayout(row2)
        vbox.addWidget(_hline())

        # Run button + status
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

        # Transcription result
        vbox.addWidget(QLabel("認識結果:"))
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setFixedHeight(80)
        self._result.setStyleSheet(_RESULT_STYLE)
        self._result.setPlaceholderText("(ここに認識結果が表示されます)")
        vbox.addWidget(self._result)

        # 4-panel analysis canvas
        self._canvas = AnalysisCanvas()
        vbox.addWidget(self._canvas, stretch=1)

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
        self._refresh_canvas()

    def _refresh_canvas(self):
        path = self._wav.text().strip()
        if not Path(path).exists():
            self._canvas.show_error("ファイルが見つかりません")
            return
        try:
            mono, sr, mag_db, freqs, times, mfcc = compute_analysis(path)
            self._canvas.show_analysis(mono, sr, mag_db, freqs, times, mfcc)
        except Exception as exc:
            self._canvas.show_error(str(exc))

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        path  = self._wav.text().strip()
        model = self._model.currentText()
        self._result.clear()
        self._result.setStyleSheet(_RESULT_STYLE)
        self._btn.setEnabled(False)
        self._status.setText("処理中...")
        self._refresh_canvas()

        self._worker = TranscribeWorker(path, model)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._status.setText)
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
