"""
apps/transcribe_demo.py

Python GUI: file picker → waveform + spectrogram + MFCC → Whisper transcription → WAV playback
The spectral analysis is NOT computed here: the STFT, mel filterbank and DCT run
in the C++ core via the `05_analyze_wav` example, which this GUI runs and parses.
Build it with `cmake --build build --config Release` (or set VOICELAB_ANALYZER).
scipy is used only for resample_poly on the way to Whisper.

Requirements: pip install PyQt6 openai-whisper scipy numpy matplotlib
Run:          python apps/transcribe_demo.py [--autorun]
"""

import os
import shutil
import subprocess
import sys
import wave as wav_mod
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

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

ANALYZER = "05_analyze_wav"


def find_analyzer() -> str:
    """Locate the 05_analyze_wav executable built from examples/.

    Set VOICELAB_ANALYZER to override; otherwise the usual single- and
    multi-config build layouts are searched, then PATH.
    """
    override = os.environ.get("VOICELAB_ANALYZER")
    if override:
        if Path(override).is_file():
            return override
        raise FileNotFoundError(
            f"VOICELAB_ANALYZER points at {override}, which does not exist")

    root = Path(__file__).resolve().parent.parent
    exe = ANALYZER + (".exe" if os.name == "nt" else "")
    candidates = [
        root / "build" / "examples" / exe,
        root / "build" / "examples" / "Release" / exe,
        root / "build" / "examples" / "Debug" / exe,
        root / "build" / exe,
        root / "build" / "Release" / exe,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    found = shutil.which(ANALYZER)
    if found:
        return found

    raise FileNotFoundError(
        f"{ANALYZER} not found. Build it first:\n"
        f"    cmake -S . -B build && cmake --build build --config Release\n"
        f"or point VOICELAB_ANALYZER at the executable.")


def _parse_analysis(text: str):
    """Parse the 05_analyze_wav stream into (meta, freqs, times, mag_db, mfcc)."""
    meta: dict[str, int] = {}
    freqs = times = None
    mag_rows: list[list[float]] = []
    mfcc_rows: list[list[float]] = []
    section = None

    def row(line: str) -> list[float]:
        return [float(v) for v in line.split(",")]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            section = line
            continue
        if section is None:
            key, _, value = line.partition(",")
            meta[key] = int(value)
        elif section == "[freqs]":
            freqs = row(line)
        elif section == "[times]":
            times = row(line)
        elif section == "[mag_db]":
            mag_rows.append(row(line))
        elif section == "[mfcc]":
            mfcc_rows.append(row(line))

    for key in ("sample_rate", "num_frames", "num_bins", "mfcc_count"):
        if key not in meta:
            raise ValueError(f"{ANALYZER} output is missing '{key}'")
    if freqs is None or times is None:
        raise ValueError(f"{ANALYZER} output is missing the freqs/times axes")
    if len(mag_rows) != meta["num_frames"] or len(mfcc_rows) != meta["num_frames"]:
        raise ValueError(
            f"{ANALYZER} announced {meta['num_frames']} frames but emitted "
            f"{len(mag_rows)} spectrogram and {len(mfcc_rows)} MFCC rows")

    # The tool emits one row per frame; the plots want [bin, frame] / [coef, frame].
    return (meta,
            np.asarray(freqs, dtype=float),
            np.asarray(times, dtype=float),
            np.asarray(mag_rows, dtype=float).T,
            np.asarray(mfcc_rows, dtype=float).T)


def compute_analysis(wav_path: str, n_mfcc: int = 13, n_mels: int = 40):
    """Return (mono, sr, mag_db, freqs_hz, times_s, mfcc [n_mfcc × n_frames]).

    The STFT, mel filterbank and DCT all run in the C++ core via the
    ``05_analyze_wav`` example; this function only reads the numbers back and
    reshapes them for matplotlib. Recomputing any of it in scipy here would
    fork the DSP, which is exactly what voicelab exists to provide.
    """
    # The waveform panel and the Whisper hand-off need the samples themselves.
    # That is plain file I/O, not signal processing, so it stays in Python.
    samples, sr, channels = load_wav_float(wav_path)
    mono = to_mono(samples, channels)

    proc = subprocess.run(
        [find_analyzer(), wav_path, str(n_mfcc), str(n_mels), "8000"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{ANALYZER} failed ({proc.returncode}): {proc.stderr.strip()}")

    meta, freqs, times, mag_db, mfcc = _parse_analysis(proc.stdout)
    if meta["sample_rate"] != sr:
        raise RuntimeError(
            f"{ANALYZER} read {meta['sample_rate']} Hz but Python read {sr} Hz")

    return mono, sr, mag_db, freqs, times, mfcc


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
                raise FileNotFoundError(f"File not found: {self.wav_path}")
            self.status.emit(f"Loading model ({self.model_name})…")
            model = whisper.load_model(self.model_name)
            self.status.emit("Analyzing audio…")
            samples, sr, channels = load_wav_float(self.wav_path)
            pcm16k = resample_to_16k(to_mono(samples, channels), sr)
            self.status.emit("Transcribing…")
            result = model.transcribe(pcm16k, language="ja", fp16=False)
            self.finished.emit(result.get("text", "").strip() or "(no transcription result)")
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

    def _placeholder(self, msg="(shown after a file is loaded)"):
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
        self._placeholder(f"Error: {msg}")


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
        row1.addWidget(QLabel("Audio file:"))
        self._wav = QLineEdit("simulation_output/my_voice.wav")
        self._wav.returnPressed.connect(self._reload)
        row1.addWidget(self._wav)
        btn_pick = QPushButton("Browse")
        btn_pick.setFixedWidth(90)
        btn_pick.clicked.connect(self._pick)
        row1.addWidget(btn_pick)
        vbox.addLayout(row1)

        # Model selector
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Whisper model:"))
        self._model = QComboBox()
        self._model.addItems(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
        self._model.setCurrentText("base")
        row2.addWidget(self._model)
        row2.addStretch()
        vbox.addLayout(row2)
        vbox.addWidget(_hline())

        # Run button + status
        row3 = QHBoxLayout()
        self._btn = QPushButton("  Transcribe  ")
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
        vbox.addWidget(QLabel("Transcription:"))
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setFixedHeight(80)
        self._result.setStyleSheet(_RESULT_STYLE)
        self._result.setPlaceholderText("(the transcription result appears here)")
        vbox.addWidget(self._result)

        # 4-panel analysis canvas
        self._canvas = AnalysisCanvas()
        vbox.addWidget(self._canvas, stretch=1)

    # ── slots ──

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a WAV file", "",
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
            self._canvas.show_error("File not found")
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
        self._status.setText("Processing...")
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
        self._result.setPlainText(f"Error:\n{msg}")
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
