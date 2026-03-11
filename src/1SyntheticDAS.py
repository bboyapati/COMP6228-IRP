"""
Step 1: Synthetic DAS Signal Generation
========================================
We simulate a vessel passing over a fiber optic cable.

The DAS data is a 2D matrix: (channels x time)
- Each channel = a physical location along the cable (spaced ~1.275m apart)
- Each time sample = a measurement at 200Hz

The vessel produces a ~30Hz tone that moves along the cable over time.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Simulation Parameters ─────────────────────────────────────────────────────

FS          = 200       # Sampling frequency (Hz) — matches real NOC data
DURATION    = 30        # Seconds of data to simulate
N_CHANNELS  = 100       # Number of DAS channels (spatial locations)
DX          = 1.275     # Spatial resolution (metres per channel)
VESSEL_FREQ = 30.0      # Vessel eigenfrequency (Hz)
SNR_DB      = 5         # Signal-to-noise ratio in dB — deliberately low, like real data

# Derived quantities
N_SAMPLES   = int(FS * DURATION)   # Total time samples
T           = np.linspace(0, DURATION, N_SAMPLES)  # Time axis (seconds)
X           = np.arange(N_CHANNELS) * DX           # Space axis (metres)

# ── Vessel Trajectory ─────────────────────────────────────────────────────────
# The vessel moves at constant speed parallel to the cable.
# Its "closest point" to each channel changes linearly over time.
# We model this as: at time t, the vessel is closest to channel c(t).

VESSEL_SPEED    = 3.0       # metres per second (slow vessel)
VESSEL_START    = 10 * DX  # starts near channel 10
vessel_position = VESSEL_START + VESSEL_SPEED * T  # position in metres over time

# ── Build the DAS Data Matrix ─────────────────────────────────────────────────
# Shape: (N_CHANNELS, N_SAMPLES)
# For each channel, the vessel signal amplitude decays with distance from vessel.

data = np.zeros((N_CHANNELS, N_SAMPLES))

for c in range(N_CHANNELS):
    channel_pos = X[c]  # physical position of this channel in metres

    # Distance from vessel to this channel at each time step
    distance = np.abs(channel_pos - vessel_position)  # shape: (N_SAMPLES,)

    # Amplitude decays with distance — simple 1/distance model
    # Add a small epsilon to avoid division by zero
    amplitude = 1.0 / (distance + 1e-3)

    # Clip amplitude so distant channels aren't completely zero
    amplitude = np.clip(amplitude, 0, 1.0)

    # Vessel signal: sinusoid at VESSEL_FREQ, modulated by amplitude envelope
    vessel_signal = amplitude * np.sin(2 * np.pi * VESSEL_FREQ * T)

    # Add a harmonic at 60Hz (2nd harmonic) — real vessels produce these
    vessel_signal += 0.4 * amplitude * np.sin(2 * np.pi * 2 * VESSEL_FREQ * T)

    data[c, :] = vessel_signal

# ── Add Realistic Noise ───────────────────────────────────────────────────────
# Real DAS noise is non-stationary but we start with Gaussian for simplicity.
# SNR_DB controls how buried the signal is.

signal_power = np.mean(data ** 2)
noise_power  = signal_power / (10 ** (SNR_DB / 10))
noise        = np.random.normal(0, np.sqrt(noise_power), data.shape)
data_noisy   = data + noise

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Waterfall plot — this is the standard way to visualise DAS data
# x-axis = time, y-axis = channel (space), colour = amplitude
ax = axes[0]
im = ax.imshow(
    data_noisy,
    aspect='auto',
    origin='lower',
    extent=[0, DURATION, 0, N_CHANNELS * DX],
    cmap='RdBu',
    vmin=-2, vmax=2
)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance along cable (m)')
ax.set_title('Synthetic DAS Data (noisy)\nVessel signature visible as diagonal streak')
plt.colorbar(im, ax=ax, label='Amplitude')

# Single channel time series — pick channel 50 (middle of cable)
ax2 = axes[1]
ch = 50
ax2.plot(T, data_noisy[ch, :], alpha=0.6, label='Noisy signal', linewidth=0.5)
ax2.plot(T, data[ch, :], label='Clean signal', linewidth=1.0, color='red')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.set_title(f'Channel {ch} time series\n(position = {X[ch]:.1f}m)')
ax2.legend()

plt.tight_layout()
plt.savefig('COMP6228-IRP/img/01_synthetic_das.png', dpi=150)
plt.show()
print("Done. Data shape:", data_noisy.shape)
print(f"  Channels: {N_CHANNELS}, Samples: {N_SAMPLES}, Duration: {DURATION}s")