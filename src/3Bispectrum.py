"""
Step 3: Bispectrum Computation
==============================
The bispectrum B(f1, f2) measures quadratic phase coupling between frequencies.

B(f1, f2) = E[ X(f1) * X(f2) * conj(X(f1+f2)) ]

Key properties:
  - For Gaussian noise: B(f1, f2) = 0  (noise is suppressed!)
  - For coupled frequencies (e.g. 30Hz + 30Hz -> 60Hz): B(30,30) != 0
  - Output is a 2D matrix — we visualise it as a heatmap

We compute this per channel, then average over channels to get
a tensor slice B(f1, f2) that we'll later stack into B(f1, f2, channel).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import windows

# ── Re-generate synthetic data (same as Steps 1 & 2) ─────────────────────────
FS          = 200
DURATION    = 30
N_CHANNELS  = 100
DX          = 1.275
VESSEL_FREQ = 30.0
SNR_DB      = 5

N_SAMPLES       = int(FS * DURATION)
T               = np.linspace(0, DURATION, N_SAMPLES)
X_pos           = np.arange(N_CHANNELS) * DX
VESSEL_SPEED    = 3.0
VESSEL_START    = 10 * DX
vessel_position = VESSEL_START + VESSEL_SPEED * T

np.random.seed(42)
data = np.zeros((N_CHANNELS, N_SAMPLES))
for c in range(N_CHANNELS):
    channel_pos   = X_pos[c]
    distance      = np.abs(channel_pos - vessel_position)
    amplitude     = np.clip(1.0 / (distance + 1e-3), 0, 1.0)
    vessel_signal = amplitude * np.sin(2 * np.pi * VESSEL_FREQ * T)
    vessel_signal += 0.4 * amplitude * np.sin(2 * np.pi * 2 * VESSEL_FREQ * T)
    data[c, :]    = vessel_signal

signal_power = np.mean(data ** 2)
noise_power  = signal_power / (10 ** (SNR_DB / 10))
noise        = np.random.normal(0, np.sqrt(noise_power), data.shape)
data_noisy   = data + noise

# ── Bispectrum Parameters ─────────────────────────────────────────────────────
WINDOW_SIZE = 256    # samples per frame — controls frequency resolution
OVERLAP     = 128    # 50% overlap between frames
N_FREQ      = WINDOW_SIZE // 2 + 1  # number of unique frequencies (one-sided)

# Frequency axis — what frequency does each bin correspond to?
freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)

# ── Core Bispectrum Function ──────────────────────────────────────────────────
def compute_bispectrum(signal, window_size=256, overlap=128):
    """
    Compute the bispectrum of a 1D signal by averaging over time windows.

    B(f1, f2) = mean over frames of [ X(f1) * X(f2) * conj(X(f1+f2)) ]

    Parameters
    ----------
    signal      : 1D numpy array, the time-domain signal
    window_size : int, samples per frame
    overlap     : int, samples of overlap between frames

    Returns
    -------
    B    : 2D complex array of shape (N_FREQ, N_FREQ)
    """
    step     = window_size - overlap
    n_frames = (len(signal) - window_size) // step + 1

    # Hann window reduces spectral leakage at frame edges
    win = windows.hann(window_size)

    # Accumulate bispectrum across frames
    B = np.zeros((N_FREQ, N_FREQ), dtype=complex)

    for i in range(n_frames):
        start = i * step
        frame = signal[start: start + window_size] * win

        # FFT of this frame
        X = np.fft.rfft(frame)   # shape: (N_FREQ,)

        # B(f1, f2) += X(f1) * X(f2) * conj(X(f1+f2))
        # We need to handle the f1+f2 index carefully —
        # it must stay within the valid frequency range.
        for f1 in range(N_FREQ):
            for f2 in range(f1, N_FREQ):       # f2 >= f1 — principal domain
                f3 = f1 + f2                   # coupled frequency index
                if f3 < N_FREQ:                # must be within range
                    B[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

    B /= n_frames   # average over frames
    return B


# ── Compute bispectrum for a strong-signal channel and a noise-only channel ───
print("Computing bispectrum for channel 50 (vessel passes nearby)...")
B_vessel = compute_bispectrum(data_noisy[50, :], WINDOW_SIZE, OVERLAP)

print("Computing bispectrum for pure Gaussian noise (for comparison)...")
pure_noise = np.random.normal(0, 1, N_SAMPLES)
B_noise    = compute_bispectrum(pure_noise, WINDOW_SIZE, OVERLAP)

print("Computing bispectrum for clean signal (no noise, ground truth)...")
B_clean    = compute_bispectrum(data[50, :], WINDOW_SIZE, OVERLAP)

print("Done.")

# ── Helper: find the frequency bin closest to a given Hz value ────────────────
def freq_to_bin(f_hz):
    return np.argmin(np.abs(freqs - f_hz))

f30 = freq_to_bin(30.0)
f60 = freq_to_bin(60.0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def plot_bispectrum(ax, B, title, mark_coupling=True):
    """Plot the magnitude of a bispectrum as a 2D heatmap."""
    B_mag = np.abs(B)

    # Only show the principal domain (upper triangle where f1+f2 <= FS/2)
    mask = np.zeros_like(B_mag)
    for i in range(N_FREQ):
        for j in range(i, N_FREQ):
            if i + j < N_FREQ:
                mask[i, j] = B_mag[i, j]

    im = ax.imshow(
        mask,
        origin='lower',
        aspect='auto',
        extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
        cmap='hot',
    )
    ax.set_xlabel('f1 (Hz)')
    ax.set_ylabel('f2 (Hz)')
    ax.set_title(title)
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(0, FS / 2)

    if mark_coupling:
        # Mark where we EXPECT the coupling peak: B(30, 30)
        ax.axvline(30, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(30, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.plot(30, 30, 'c+', markersize=12, markeredgewidth=2,
                label='Expected: B(30,30)')
        ax.legend(fontsize=7)

    plt.colorbar(im, ax=ax, label='|B(f1,f2)|')
    return im

# Row 1: Bispectrum heatmaps
ax1 = fig.add_subplot(gs[0, 0])
plot_bispectrum(ax1, B_clean,  'Clean signal bispectrum\n(no noise — ground truth)')

ax2 = fig.add_subplot(gs[0, 1])
plot_bispectrum(ax2, B_vessel, 'Noisy signal bispectrum\n(channel 50, SNR=5dB)')

ax3 = fig.add_subplot(gs[0, 2])
plot_bispectrum(ax3, B_noise,  'Pure Gaussian noise bispectrum\n(should be ~flat/zero)', mark_coupling=False)

# Row 2: Slice through B(30, f2) — shows the coupling peak clearly
ax4 = fig.add_subplot(gs[1, 0])
slice_clean  = np.abs(B_clean[f30,  :])
slice_vessel = np.abs(B_vessel[f30, :])
slice_noise  = np.abs(B_noise[f30,  :])

ax4.plot(freqs, slice_clean,  label='Clean signal',  linewidth=1.5)
ax4.plot(freqs, slice_vessel, label='Noisy signal',  linewidth=1.0, alpha=0.8)
ax4.plot(freqs, slice_noise,  label='Pure noise',    linewidth=1.0, alpha=0.8, linestyle='--')
ax4.axvline(30, color='red', linestyle=':', linewidth=1, label='f2=30Hz')
ax4.set_xlabel('f2 (Hz)')
ax4.set_ylabel('|B(30, f2)|')
ax4.set_title('Slice through B(f1=30Hz, f2)\nPeak at f2=30 confirms coupling')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, FS / 2)

# Row 2: Explanation panel
ax5 = fig.add_subplot(gs[1, 1:])
ax5.set_axis_off()
explanation = """
READING THE BISPECTRUM
──────────────────────────────────────────────────────────────────

• Clean signal (left):   Bright spot at B(30,30) — 30Hz + 30Hz → 60Hz coupling clearly visible.
                          This is the quadratic phase coupling your project aims to exploit.

• Noisy signal (centre): Same coupling peak still visible above the noise floor at B(30,30).
                          This is the key advantage: Gaussian noise contributes ~zero to B,
                          so the SNR in the bispectral domain is better than in the power spectrum.

• Pure noise (right):    Roughly flat/uniform — no structure, no peaks.
                          For a true Gaussian process, B → 0 as averaging increases.

• Slice plot (left):     A 1D cut through the bispectrum at f1=30Hz.
                          The peak at f2=30Hz directly quantifies the 30→60Hz coupling strength.
                          Noise produces no such peak.

NEXT STEP: Stack per-channel bispectra into a tensor B(f1, f2, channel)
           and apply Nonnegative Tensor Factorisation to decompose it.
"""
ax5.text(0.02, 0.95, explanation,
         transform=ax5.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Step 3: Bispectrum Computation', fontsize=13, fontweight='bold')
plt.savefig('COMP6228-IRP/img/03_bispectrum.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Print coupling strength at B(30, 30) ─────────────────────────────────────
print(f"\nCoupling strength at B(30Hz, 30Hz):")
print(f"  Clean signal : {np.abs(B_clean[f30,  f30]):.4f}")
print(f"  Noisy signal : {np.abs(B_vessel[f30, f30]):.4f}")
print(f"  Pure noise   : {np.abs(B_noise[f30,  f30]):.4f}")
print(f"\nFrequency bin for 30Hz: bin {f30} = {freqs[f30]:.2f} Hz")