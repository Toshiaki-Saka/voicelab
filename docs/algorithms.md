# Algorithm notes

This document is the "why" behind the code in `src/core/`. It is written
to be readable in isolation; references are given where the original
derivations are clearer than what fits in code comments.

## 1. Sampling and the Discrete Fourier Transform

An audio signal $x(t)$ sampled at rate $f_s$ becomes a sequence
$x[n] = x(n / f_s)$. The Discrete Fourier Transform (DFT) of an $N$-sample
block is

$$
X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}, \quad k = 0, \dots, N-1.
$$

For a real input, $X[N-k] = \overline{X[k]}$, so we only need the first
$N/2 + 1$ bins (this is what pocketfft's `r2c` produces).

The bin index $k$ corresponds to frequency $k f_s / N$ Hz. Hence the
**frequency resolution** is $f_s / N$ — a 1024-sample block at 44.1 kHz
resolves 43 Hz per bin. Larger $N$ → better frequency resolution, but worse
time resolution. This is the time-frequency trade-off.

## 2. Short-Time Fourier Transform (STFT)

A long signal is divided into overlapping frames of length $N$, hopped by
$H$ samples. Each frame is multiplied by an analysis window $w[n]$ before
FFT to control spectral leakage:

$$
X[m, k] = \sum_{n=0}^{N-1} w[n] \, x[mH + n] \, e^{-j 2\pi k n / N}.
$$

**Window choice.** A rectangular window has the narrowest main lobe but
huge sidelobes (-13 dB). Hann (`0.5 - 0.5 cos(...)`) is a good default: -32
dB sidelobes, smooth, COLA-friendly with $H = N/4$. Blackman-Harris is
better for analysis precision at the cost of main-lobe width.

We use the **periodic** form of the window (divide by $N$ rather than
$N-1$) because it makes consecutive frames composable under overlap-add.

## 3. Inverse STFT and Overlap-Add (OLA)

To go back to time domain we

1. Inverse-FFT each spectrum → real frame $y_m[n]$.
2. Multiply by a synthesis window $w_s[n]$.
3. Add at offset $mH$ into the output:

$$
\hat{x}[n] = \frac{1}{C} \sum_m w_s[n - mH] \, y_m[n - mH]
$$

where $C$ is the **constant-overlap-add (COLA) constant**

$$
C = \sum_m w_a[n - mH] \, w_s[n - mH]
$$

which must be independent of $n$ for the reconstruction to be exact. For
Hann/Hann with $H = N/4$, $C = 1.5$. We compute $C$ once in the constructor
and divide by it.

## 4. Mel filterbank and MFCC

Human pitch perception is roughly logarithmic. The mel scale is a
piecewise-linear approximation:

$$
m(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

We place $M$ triangular filters evenly on the mel axis between $f_\text{min}$
and $f_\text{max}$, then map each triangle's three vertices back to Hz, then
to FFT-bin indices. Applying the filterbank to a power spectrum gives an
$M$-vector that approximates how a cochlea would integrate energy.

Logarithmic compression $y = \log(\epsilon + x)$ approximates the
loudness-vs-intensity perceptual relationship.

The MFCCs are then the DCT-II of the log-mel vector. The DCT-II has the
property that, for natural log-spectra, most of the variance lands in the
first 10–13 coefficients — which is why classic ASR pipelines truncated to
13 MFCCs.

## 5. Energy-based VAD

A simple Voice Activity Detector that beats "always voiced" looks like:

1. Compute frame RMS in dBFS.
2. Track a slowly-moving estimate of the noise floor $\nu$, updated only
   while the frame is *not* judged voiced.
3. Declare "voiced" if $\text{rms} - \nu > \theta$ (e.g. 9 dB).
4. Apply a hangover: once voiced, stay voiced for at least $T$ ms after the
   last voiced frame, to bridge unvoiced consonant gaps.

It is not robust to non-stationary noise. For demanding settings, use
Silero-VAD or WebRTC-VAD instead — but the energy VAD is sufficient as a
teaching example and a first-stage gate.

## 6. Phase-vocoder pitch shifting

A pure FFT-modify-IFFT pitch shift fails because the phase of each bin
encodes *which sub-bin frequency* the energy belongs to. The phase-vocoder
fixes this in three steps.

**Phase unwrapping per bin.** Between consecutive frames, the phase of bin
$k$ should advance by $2\pi k H / N$ if the bin's "true" frequency is
exactly its center. Any residual deviation (after wrapping to $(-\pi, \pi]$)
tells us the bin's *instantaneous* frequency:

$$
f_\text{true}[k] = \frac{k f_s}{N} + \frac{\Delta\phi_\text{wrapped}}{2\pi H / f_s}.
$$

**Bin remapping.** To shift by ratio $r$ (e.g. $r = 2^{1/12}$ for one
semitone up), the energy in bin $k$ moves to bin $\lfloor k r \rfloor$, and
its true frequency is multiplied by $r$.

**Phase resynthesis.** At the synthesis frame, the phase of each bin is
re-accumulated from the desired instantaneous frequency, so consecutive
frames remain phase-coherent under overlap-add.

The result still sounds slightly "phasey" on transients because each bin is
treated independently. Phase-locking variants (Laroche & Dolson, 1999) help
but are out of scope for this library's first release.

## References

- Heinzel, Rüdiger, Schilling, *Spectrum and spectral density estimation by
  the Discrete Fourier transform (DFT)*, Max-Planck-Institut für
  Gravitationsphysik, 2002.
- Laroche, Dolson, *Improved phase vocoder time-scale modification of
  audio*, IEEE Trans. Speech and Audio Processing, 1999.
- Smith, *Spectral Audio Signal Processing*, online textbook,
  https://ccrma.stanford.edu/~jos/sasp/.
- Davis, Mermelstein, *Comparison of parametric representations for
  monosyllabic word recognition*, IEEE ASSP, 1980 (origin of MFCCs).
