<#
.SYNOPSIS
    voicelab — build / simulate / visualize (Python GUI) helper script.

.DESCRIPTION
    1. Configure and build the root project with CMake.
    2. Generate a chirp WAV with Python.
    3. Run 01_sine_spectrum / 04_mfcc_dump.
    4. Visualize the spectrogram / MFCC / waveform in a matplotlib GUI.

.PARAMETER BuildType
    CMake build type (Release / Debug / RelWithDebInfo). Default: Release.

.EXAMPLE
    .\build_and_visualize.ps1
    .\build_and_visualize.ps1 -BuildType Debug
#>

param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = 'Stop'
$Root  = $PSScriptRoot
$Build = Join-Path $Root 'build'
$Out   = Join-Path $Root 'simulation_output'

# ── 0. Prepare the output directory ──────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$WavPath  = Join-Path $Out 'chirp.wav'
$CsvPath  = Join-Path $Out 'mfcc.csv'
$PyGenWav = Join-Path $Out 'gen_wav.py'
$PyViz    = Join-Path $Out 'visualize.py'

# ─────────────────────────────────────────────────────────────────────────────
# ── 1. CMake Configure ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[1/5] CMake Configure ...' -ForegroundColor Cyan
cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed.' }

# ─────────────────────────────────────────────────────────────────────────────
# ── 2. Build ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[2/5] Build ...' -ForegroundColor Cyan
cmake --build $Build --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { throw 'cmake build failed.' }

# ─────────────────────────────────────────────────────────────────────────────
# helper: locate an exe in the build output (handles both MSVC and GCC layouts)
# ─────────────────────────────────────────────────────────────────────────────
function Find-Exe([string]$Name) {
    $candidates = @(
        (Join-Path $Build "examples\$BuildType\$Name.exe"),
        (Join-Path $Build "examples\$Name.exe"),
        (Join-Path $Build "$BuildType\examples\$Name.exe"),
        (Join-Path $Build "Release\examples\$Name.exe"),
        (Join-Path $Build "Debug\examples\$Name.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $hit = Get-ChildItem -Path $Build -Recurse -Filter "$Name.exe" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($hit) { return $hit.FullName }
    throw "$Name.exe not found in build output: $Build"
}

$sineExe = Find-Exe '01_sine_spectrum'
$mfccExe = Find-Exe '04_mfcc_dump'
Write-Host "  sine_spectrum : $sineExe" -ForegroundColor DarkGray
Write-Host "  mfcc_dump     : $mfccExe" -ForegroundColor DarkGray

# ─────────────────────────────────────────────────────────────────────────────
# ── 3. Generate the chirp WAV (Python stdlib only) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[3/5] Generating chirp WAV ...' -ForegroundColor Cyan

@'
"""Generate a linear-chirp WAV (220 Hz -> 3520 Hz, 3 s, 44100 Hz, mono PCM16)."""
import sys, wave, struct, math

def main():
    out = sys.argv[1]
    SR, DUR = 44100, 3.0
    F0, F1  = 220.0, 3520.0
    N = int(SR * DUR)

    samples = []
    for i in range(N):
        t     = i / SR
        phase = 2.0 * math.pi * (F0 * t + (F1 - F0) * t * t / (2.0 * DUR))
        samples.append(int(32767 * 0.5 * math.sin(phase)))

    with wave.open(out, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(struct.pack(f'<{N}h', *samples))

    print(f'  WAV: {out}  ({N} samples @ {SR} Hz)')

if __name__ == '__main__':
    main()
'@ | Set-Content -Path $PyGenWav -Encoding UTF8

python $PyGenWav $WavPath
if ($LASTEXITCODE -ne 0) { throw 'WAV generation failed.' }

# ─────────────────────────────────────────────────────────────────────────────
# ── 4. C++ simulation ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[4/5] Running C++ simulation ...' -ForegroundColor Cyan

# 01_sine_spectrum: no arguments, console output only
Write-Host '  --- 01_sine_spectrum ---' -ForegroundColor Yellow
& $sineExe
if ($LASTEXITCODE -ne 0) { throw '01_sine_spectrum failed.' }

# 04_mfcc_dump: WAV -> MFCC CSV
Write-Host "  --- 04_mfcc_dump -> $CsvPath ---" -ForegroundColor Yellow
& $mfccExe $WavPath 13 | Set-Content -Path $CsvPath -Encoding ASCII
if ($LASTEXITCODE -ne 0) { throw '04_mfcc_dump failed.' }
$rowCount = (Get-Content $CsvPath | Measure-Object -Line).Lines - 1   # excluding the header
Write-Host "  MFCC: $rowCount frames x 13 coefficients -> $CsvPath" -ForegroundColor DarkGray

# ─────────────────────────────────────────────────────────────────────────────
# ── 5. Python GUI visualization ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[5/5] Checking Python dependencies ...' -ForegroundColor Cyan

# Install matplotlib / numpy if missing
foreach ($pkg in @('matplotlib', 'numpy')) {
    $check = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  pip install $pkg ..." -ForegroundColor Yellow
        python -m pip install --quiet $pkg
    }
}

Write-Host 'Launching the Python GUI ...' -ForegroundColor Cyan

@'
"""voicelab simulation visualizer — matplotlib TkAgg GUI."""
import sys, csv, wave, struct
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

WAV_PATH = sys.argv[1]
CSV_PATH = sys.argv[2]

# ── Load data ─────────────────────────────────────────────────────────────────
with wave.open(WAV_PATH, 'r') as wf:
    SR       = wf.getframerate()
    n_frames = wf.getnframes()
    samples  = np.frombuffer(wf.readframes(n_frames), dtype='<i2').astype(np.float32) / 32768.0

with open(CSV_PATH, newline='') as f:
    rows = list(csv.DictReader(f))

n_mfcc_coeff = len(rows[0]) - 1          # excluding the 'frame' column
mfcc = np.array([
    [float(rows[i][f'mfcc{j}']) for j in range(n_mfcc_coeff)]
    for i in range(len(rows))
]).T  # shape: (n_mfcc_coeff, n_frames)

# ── Color theme ───────────────────────────────────────────────────────────────
BG   = '#1e1e2e'
BG2  = '#181825'
FG   = '#cdd6f4'
FG2  = '#a6adc8'
EDGE = '#45475a'

# ── Tk window ─────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title('voicelab  —  Chirp Simulation Visualizer')
root.configure(bg=BG)
root.state('zoomed')

tk.Label(root,
         text='voicelab  |  Linear-Chirp Simulation  (220 Hz → 3520 Hz, 3 s)',
         bg=BG, fg=FG, font=('Segoe UI', 13, 'bold'), pady=6
         ).pack(side=tk.TOP, fill=tk.X)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig,
                         hspace=0.52, wspace=0.32,
                         left=0.06, right=0.97, top=0.95, bottom=0.06)

def style_ax(ax, title):
    ax.set_facecolor(BG2)
    ax.set_title(title, color=FG, fontsize=10, pad=5)
    ax.tick_params(colors=FG2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    ax.xaxis.label.set_color(FG2)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_color(FG2)
    ax.yaxis.label.set_fontsize(8)

# ── (row 0, full width) Waveform ─────────────────────────────────────────────
ax_wave = fig.add_subplot(gs[0, :])
t = np.arange(len(samples)) / SR
ax_wave.plot(t, samples, color='#89dceb', linewidth=0.35, alpha=0.9)
style_ax(ax_wave, 'Waveform')
ax_wave.set_xlabel('Time (s)')
ax_wave.set_ylabel('Amplitude')
ax_wave.set_xlim(0, t[-1])

# ── (row 1, full width) Spectrogram ──────────────────────────────────────────
ax_spec = fig.add_subplot(gs[1, :])
_, _, _, im_spec = ax_spec.specgram(
    samples, NFFT=1024, Fs=SR, noverlap=768, cmap='inferno', scale='dB'
)
style_ax(ax_spec, 'Spectrogram  (NFFT=1024, overlap=75 %)')
ax_spec.set_xlabel('Time (s)')
ax_spec.set_ylabel('Frequency (Hz)')
cb1 = fig.colorbar(im_spec, ax=ax_spec, pad=0.01)
cb1.ax.tick_params(labelsize=7, colors=FG2)
cb1.ax.yaxis.label.set_color(FG2)

# ── (row 2, left) MFCC heatmap ───────────────────────────────────────────────
ax_heat = fig.add_subplot(gs[2, 0])
im_heat = ax_heat.imshow(
    mfcc, aspect='auto', origin='lower', cmap='coolwarm',
    extent=[0, mfcc.shape[1], -0.5, n_mfcc_coeff - 0.5]
)
style_ax(ax_heat, f'MFCC Heatmap  ({n_mfcc_coeff} coefficients × {mfcc.shape[1]} frames)')
ax_heat.set_xlabel('Frame')
ax_heat.set_ylabel('MFCC index')
cb2 = fig.colorbar(im_heat, ax=ax_heat, pad=0.01)
cb2.ax.tick_params(labelsize=7, colors=FG2)

# ── (row 2, right) MFCC mean ± std per coefficient ───────────────────────────
ax_bar = fig.add_subplot(gs[2, 1])
means = mfcc.mean(axis=1)
stds  = mfcc.std(axis=1)
x = np.arange(n_mfcc_coeff)
ax_bar.bar(x, means, color='#a6e3a1', alpha=0.8, label='mean', zorder=2)
ax_bar.errorbar(x, means, yerr=stds,
                fmt='none', color='#f38ba8', capsize=3,
                linewidth=1.3, label='±std', zorder=3)
ax_bar.axhline(0, color=EDGE, linewidth=0.7)
style_ax(ax_bar, 'MFCC  Mean ± Std  (all frames)')
ax_bar.set_xlabel('MFCC index')
ax_bar.set_ylabel('Value')
leg = ax_bar.legend(fontsize=7, facecolor='#313244',
                     labelcolor=FG, edgecolor=EDGE)

# ── Embed the canvas in Tk ────────────────────────────────────────────────────
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

toolbar_frame = tk.Frame(root, bg=BG)
toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
NavigationToolbar2Tk(canvas, toolbar_frame)

tk.Label(root,
         text=(f'  WAV: {len(samples)} samples @ {SR} Hz'
               f'  |  MFCC frames: {mfcc.shape[1]}'
               f'  |  Coefficients: {n_mfcc_coeff}'),
         bg='#313244', fg=FG2, font=('Segoe UI', 8), anchor='w', pady=3
         ).pack(side=tk.BOTTOM, fill=tk.X)

root.protocol("WM_DELETE_WINDOW", lambda: (plt.close('all'), root.destroy()))
root.mainloop()
sys.exit(0)
'@ | Set-Content -Path $PyViz -Encoding UTF8

python $PyViz $WavPath $CsvPath
if ($LASTEXITCODE -ne 0) { throw 'Python GUI exited abnormally.' }

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
