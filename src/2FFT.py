"""
Step 2: FFT and Power Spectrum
==============================
We compute the FFT and power spectrum of our synthetic DAS data.

Key insight: The power spectrum shows us WHAT frequencies are present,
but tells us NOTHING about whether those frequencies are coupled.
This is the fundamental limitation of PCA — it operates on the
covariance matrix which is built from second-order statistics (power spectrum).

We'll see the 30Hz and 60Hz peaks clearly, but can't tell from the
power spectrum alone whether 60Hz is a harmonic of 30Hz (coupled)
or an independent source.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Re-generate the synthetic data (same as Step 1) ──────────────────────────
# In your real project, you'll load data from file instead of this block.

FS          = 200
DURATION    = 30
N_CHANNELS  = 100
DX          = 1.275
VESSEL_FREQ = 30.0
SNR_DB      = 5

N_SAMPLES       = int(FS * DURATION)
T               = np.linspace(0, DURATION, N_SAMPLES)
X               = np.arange(N_CHANNELS) * DX
VESSEL_SPEED    = 3.0
VESSEL_START    = 10 * DX
vessel_position = VESSEL_START + VESSEL_SPEED * T

np.random.seed(42)  # reproducibility
data = np.zeros((N_CHANNELS, N_SAMPLES))
for c in range(N_CHANNELS):
    channel_pos = X[c]
    distance    = np.abs(channel_pos - vessel_position)
    amplitude   = np.clip(1.0 / (distance + 1e-3), 0, 1.0)
    vessel_signal = amplitude * np.sin(2 * np.pi * VESSEL_FREQ * T)
    vessel_signal += 0.4 * amplitude * np.sin(2 * np.pi * 2 * VESSEL_FREQ * T)
    data[c, :]  = vessel_signal

signal_power  = np.mean(data ** 2)
noise_power   = signal_power / (10 ** (SNR_DB / 10))
noise         = np.random.normal(0, np.sqrt(noise_power), data.shape)
data_noisy    = data + noise

# ── FFT of a single channel ───────────────────────────────────────────────────
# Pick channel 50 — the vessel passes closest to it, so signal is strongest.
ch = 50
signal = data_noisy[ch, :]

# np.fft.rfft gives us the one-sided FFT (positive frequencies only)
# This is what you want for real-valued signals
fft_vals = np.fft.rfft(signal)
freqs    = np.fft.rfftfreq(N_SAMPLES, d=1.0/FS)  # frequency axis in Hz

# Power spectrum = |FFT|^2 / N
# This tells us how much energy is at each frequency
power_spectrum = (np.abs(fft_vals) ** 2) / N_SAMPLES

# ── FFT averaged across all channels ─────────────────────────────────────────
# More robust — averages out noise across channels
# This is closer to what PCA implicitly operates on
all_ffts   = np.fft.rfft(data_noisy, axis=1)           # shape: (N_CHANNELS, N_FREQ)
mean_power = np.mean(np.abs(all_ffts) ** 2, axis=0) / N_SAMPLES

# ── Short-Time Fourier Transform (STFT) ──────────────────────────────────────
# FFT of the full signal loses time information.
# STFT splits the signal into overlapping windows and FFTs each one.
# This shows how the frequency content CHANGES OVER TIME.
# This is the precursor to the bispectrum computation.

from scipy.signal import stft

WINDOW_SIZE = 256   # samples per window (~1.28s at 200Hz)
OVERLAP     = 128   # 50% overlap

f_stft, t_stft, Zxx = stft(
    data_noisy[ch, :],
    fs=FS,
    nperseg=WINDOW_SIZE,
    noverlap=OVERLAP
)

# Power spectrogram: |STFT|^2
spectrogram = np.abs(Zxx) ** 2

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Single channel power spectrum
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(freqs, 10 * np.log10(power_spectrum + 1e-12), linewidth=0.8)
ax1.axvline(VESSEL_FREQ,     color='red',    linestyle='--', label=f'{VESSEL_FREQ}Hz (fundamental)', alpha=0.8)
ax1.axvline(2 * VESSEL_FREQ, color='orange', linestyle='--', label=f'{2*VESSEL_FREQ}Hz (2nd harmonic)', alpha=0.8)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power (dB)')
ax1.set_title(f'Power Spectrum — Channel {ch}\n(single channel, full signal)')
ax1.set_xlim(0, FS / 2)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Channel-averaged power spectrum
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(freqs, 10 * np.log10(mean_power + 1e-12), linewidth=0.8, color='green')
ax2.axvline(VESSEL_FREQ,     color='red',    linestyle='--', label=f'{VESSEL_FREQ}Hz', alpha=0.8)
ax2.axvline(2 * VESSEL_FREQ, color='orange', linestyle='--', label=f'{2*VESSEL_FREQ}Hz', alpha=0.8)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power (dB)')
ax2.set_title('Channel-Averaged Power Spectrum\n(all 100 channels averaged)')
ax2.set_xlim(0, FS / 2)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Spectrogram — time vs frequency
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.pcolormesh(
    t_stft, f_stft,
    10 * np.log10(spectrogram + 1e-12),
    shading='gouraud', cmap='inferno'
)
ax3.axhline(VESSEL_FREQ,     color='red',    linestyle='--', linewidth=1, label=f'{VESSEL_FREQ}Hz')
ax3.axhline(2 * VESSEL_FREQ, color='orange', linestyle='--', linewidth=1, label=f'{2*VESSEL_FREQ}Hz')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Frequency (Hz)')
ax3.set_title(f'Spectrogram — Channel {ch}\n(STFT: how frequency content changes over time)')
ax3.set_ylim(0, FS / 2)
ax3.legend(fontsize=8)
plt.colorbar(im, ax=ax3, label='Power (dB)')

# Plot 4: The KEY insight — what PCA sees vs what bispectrum sees
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_axis_off()
insight_text = """
KEY INSIGHT — Why PCA is not enough
─────────────────────────────────────────
The power spectrum (top plots) clearly shows:
  • A peak at 30 Hz  (vessel fundamental)
  • A peak at 60 Hz  (2nd harmonic)

PCA operates on the COVARIANCE MATRIX,
built from second-order statistics.
It can separate sources by variance, but
it CANNOT tell you:

  → Is 60Hz caused by the same vessel
    as 30Hz (quadratic phase coupling)?

  → Or is it a completely independent
    second source?

The bispectrum answers this directly.
If B(30, 30) ≠ 0, then 30Hz and 60Hz
are COUPLED — same physical source.

This is what Step 3 will show you.
"""
ax4.text(0.05, 0.95, insight_text,
         transform=ax4.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Step 2: FFT and Power Spectrum Analysis', fontsize=13, fontweight='bold')
plt.savefig('COMP6228-IRP/img/02_fft_power_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()
print("Frequency resolution:", freqs[1] - freqs[0], "Hz")
print("Vessel peak visible at:", freqs[np.argmax(power_spectrum[(freqs > 25) & (freqs < 35)])
                                          + np.searchsorted(freqs, 25)], "Hz")