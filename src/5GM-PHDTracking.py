"""
Step 5: GM-PHD Tracking
========================
We feed the NTF spatial factors into a GM-PHD filter to produce
smooth vessel trajectories.

The GM-PHD (Gaussian Mixture Probability Hypothesis Density) filter
is a multi-target tracker. It:
  1. Takes noisy position measurements at each time stepconda 
  2. Maintains a probability distribution over where targets might be
  3. Outputs estimated target positions (tracks)

In our case:
  - "Measurements" = peaks in the NTF spatial factor C_k at each time step
  - "Tracks" = smooth vessel trajectories along the cable

We use the time-varying bispectrum to get per-timestep spatial activations,
then feed those into the tracker.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from scipy.signal import windows, find_peaks
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.array import StateVector, CovarianceMatrix
from datetime import datetime, timedelta
from tqdm import tqdm

# ── Re-generate synthetic data ────────────────────────────────────────────────
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

# ── Time-varying bispectrum tensor B(f1, f2, channel, time) ──────────────────
# Unlike Step 4 where we averaged over time, here we keep time dimension
# so we can extract per-timestep spatial activations for tracking.

WINDOW_SIZE = 256
OVERLAP     = 128
STEP        = WINDOW_SIZE - OVERLAP
N_FREQ      = WINDOW_SIZE // 2 + 1
freqs       = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)
n_frames    = (N_SAMPLES - WINDOW_SIZE) // STEP + 1

print(f"Computing time-varying bispectral tensor...")
print(f"  Frames: {n_frames}, Channels: {N_CHANNELS}, Freq bins: {N_FREQ}")

def compute_bispectrum_fast(signal, window_size=256, overlap=128):
    """Fast vectorised bispectrum — returns magnitude."""
    step     = window_size - overlap
    n_frames = (len(signal) - window_size) // step + 1
    n_freq   = window_size // 2 + 1
    win      = windows.hann(window_size)
    B        = np.zeros((n_freq, n_freq), dtype=complex)
    f_idx    = np.arange(n_freq)
    sum_idx  = f_idx[:, None] + f_idx[None, :]
    valid    = sum_idx < n_freq
    sum_idx_clipped = np.where(valid, sum_idx, 0)

    for i in range(n_frames):
        start = i * step
        frame = signal[start: start + window_size] * win
        X     = np.fft.rfft(frame)
        X_outer    = np.outer(X, X)
        X_conj_f3  = np.conj(X[sum_idx_clipped])
        B += np.where(valid, X_outer * X_conj_f3, 0)

    return np.abs(B) / n_frames

# Build time-averaged tensor for NTF (same as Step 4)
tensor = np.zeros((N_FREQ, N_FREQ, N_CHANNELS), dtype=np.float32)
for c in tqdm(range(N_CHANNELS), desc="Building tensor"):
    tensor[:, :, c] = compute_bispectrum_fast(data_noisy[c, :], WINDOW_SIZE, OVERLAP)

f_idx   = np.arange(N_FREQ)
sum_idx = f_idx[:, None] + f_idx[None, :]
mask_2d = (f_idx[:, None] <= f_idx[None, :]) & (sum_idx < N_FREQ)
tensor  = tensor * mask_2d[:, :, np.newaxis]

# ── Run NTF ───────────────────────────────────────────────────────────────────
R = 4
print(f"\nRunning NTF (rank={R})...")
cp_tensor = non_negative_parafac(
    tl.tensor(tensor), rank=R, n_iter_max=100,
    init='random', random_state=42, verbose=False
)
factors = cp_tensor.factors
A_raw, B_raw, C_raw = factors[0], factors[1], factors[2]

# Normalise to recover meaningful weights
weights = np.zeros(R)
A = np.zeros_like(A_raw)
B_fac = np.zeros_like(B_raw)
C = np.zeros_like(C_raw)
for k in range(R):
    na = np.linalg.norm(A_raw[:, k]) + 1e-12
    nb = np.linalg.norm(B_raw[:, k]) + 1e-12
    nc = np.linalg.norm(C_raw[:, k]) + 1e-12
    weights[k] = na * nb * nc
    A[:, k]     = A_raw[:, k] / na
    B_fac[:, k] = B_raw[:, k] / nb
    C[:, k]     = C_raw[:, k] / nc

# Pick the highest-weight component as the vessel component
vessel_component = np.argmax(weights)
print(f"  Vessel component: {vessel_component} (weight={weights[vessel_component]:.1f})")

# ── Extract per-frame spatial activations for tracking ───────────────────────
# We project the per-frame bispectrum onto the vessel spatial factor
# to get a 1D activation profile (channel) at each time frame.
# Peaks in this profile = candidate vessel positions.

print("\nExtracting per-frame spatial activations...")

# Per-frame spatial energy: for each frame, compute which channels are active
# by projecting the frame's bispectrum onto the vessel frequency factor
vessel_freq_factor = A[:, vessel_component]  # shape: (N_FREQ,)

frame_activations = np.zeros((n_frames, N_CHANNELS))

win = windows.hann(WINDOW_SIZE)
for i in tqdm(range(n_frames), desc="Frames"):
    for c in range(N_CHANNELS):
        start = i * STEP
        frame = data_noisy[c, start: start + WINDOW_SIZE] * win
        X     = np.fft.rfft(frame)
        # Project FFT magnitude onto vessel frequency factor
        frame_activations[i, c] = np.dot(np.abs(X), vessel_freq_factor)

# Time axis for frames
t_frames = np.array([i * STEP / FS for i in range(n_frames)])

# ── Extract measurements from spatial activation peaks ────────────────────────
# At each time frame, find peaks in the spatial activation profile.
# These are candidate vessel positions fed to the tracker.

print("\nExtracting peak measurements...")
all_measurements = []   # list of lists: measurements[frame] = [pos1, pos2, ...]

for i in range(n_frames):
    profile = frame_activations[i, :]
    # Normalise profile
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-12)
    # Find peaks above threshold
    peaks, props = find_peaks(profile_norm, height=0.3, distance=5)
    positions_m  = X_pos[peaks]   # convert channel index to metres
    all_measurements.append(positions_m)

n_measurements = sum(len(m) for m in all_measurements)
print(f"  Total measurements across all frames: {n_measurements}")

# ── GM-PHD Tracker (Stone Soup) ───────────────────────────────────────────────
# State vector: [position, velocity] along cable
# Measurement: [position] only (we observe where, not how fast)

print("\nRunning GM-PHD tracker...")

# Time parameters
dt      = STEP / FS        # seconds per frame
start_time = datetime(2024, 1, 1)

# Transition model: constant velocity
# Process noise σ_v = 1.0 m/s² — how much we allow velocity to change
transition_model = ConstantVelocity(noise_diff_coeff=1.0)

# Measurement model: we observe position only (not velocity)
# Measurement noise σ_ε = 5.0 m
measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],              # observe dimension 0 (position)
    noise_covar=np.array([[25.0]])   # σ² = 25 → σ = 5m
)

# Build Stone Soup components
predictor    = KalmanPredictor(transition_model)
updater      = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=3)
associator   = GNNWith2DAssignment(hypothesiser)
deleter      = CovarianceBasedDeleter(covar_trace_thresh=500)
initiator    = MultiMeasurementInitiator(
    prior_state=GaussianState(
        StateVector([0, 0]),
        CovarianceMatrix(np.diag([100, 25])),   # initial uncertainty
        timestamp=start_time
    ),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=associator,
    updater=updater,
    min_points=3    # need 3 consistent measurements to initiate a track
)

# Convert measurements to Stone Soup Detection objects
detections_by_time = []
for i, positions in enumerate(all_measurements):
    t = start_time + timedelta(seconds=float(t_frames[i]))
    frame_detections = set()
    for pos in positions:
        det = Detection(
            state_vector=StateVector([[pos]]),
            timestamp=t,
            measurement_model=measurement_model
        )
        frame_detections.add(det)
    detections_by_time.append((t, frame_detections))

# Run tracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=iter(detections_by_time),
    data_associator=associator,
    updater=updater
)

tracks = set()
for time, current_tracks in tracker:
    tracks.update(current_tracks)

print(f"  Tracks found: {len(tracks)}")

# ── Ground truth trajectory ───────────────────────────────────────────────────
# Convert continuous vessel position to frame positions
gt_positions = VESSEL_START + VESSEL_SPEED * t_frames
# Only keep frames where vessel is within cable range
in_range     = (gt_positions >= X_pos[0]) & (gt_positions <= X_pos[-1])

# ── Plot Results ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Spatial activation heatmap over time
ax1 = fig.add_subplot(gs[0, :])
im  = ax1.imshow(
    frame_activations.T,
    aspect='auto', origin='lower',
    extent=[t_frames[0], t_frames[-1], X_pos[0], X_pos[-1]],
    cmap='hot'
)
# Ground truth
ax1.plot(t_frames[in_range], gt_positions[in_range],
         'c--', linewidth=2, label='Ground truth', zorder=5)
# Raw measurements
for i, positions in enumerate(all_measurements):
    if len(positions) > 0:
        ax1.scatter([t_frames[i]] * len(positions), positions,
                    c='lime', s=8, alpha=0.5, zorder=4)
# Tracker output
for track in tracks:
    states = sorted(track.states, key=lambda s: s.timestamp)
    if len(states) > 3:
        t_track = [(s.timestamp - start_time).total_seconds() for s in states]
        p_track = [s.state_vector[0, 0] for s in states]
        ax1.plot(t_track, p_track, 'b-', linewidth=2, alpha=0.8, zorder=6)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Cable position (m)')
ax1.set_title('Spatial Activation + Measurements + Tracker Output\n'
              'Hot=activation | Cyan=ground truth | Lime=detections | Blue=tracker')
plt.colorbar(im, ax=ax1, label='Activation')
ax1.legend(loc='upper left', fontsize=8)

# Plot 2: Per-frame activation profile at a specific time
ax2 = fig.add_subplot(gs[1, 0])
frame_idx = n_frames // 2    # middle of recording — vessel is on cable
ax2.plot(X_pos, frame_activations[frame_idx, :], linewidth=1.2)
ax2.axvline(gt_positions[frame_idx], color='red', linestyle='--',
            label=f'True position: {gt_positions[frame_idx]:.1f}m')
ax2.set_xlabel('Cable position (m)')
ax2.set_ylabel('Activation')
ax2.set_title(f'Spatial activation profile at t={t_frames[frame_idx]:.1f}s')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Tracking error over time
ax3 = fig.add_subplot(gs[1, 1])
if len(tracks) > 0:
    best_track = max(tracks, key=lambda t: len(t.states))
    states     = sorted(best_track.states, key=lambda s: s.timestamp)
    t_track    = np.array([(s.timestamp - start_time).total_seconds() for s in states])
    p_track    = np.array([s.state_vector[0, 0] for s in states])

    # Interpolate ground truth at track times
    gt_at_track = VESSEL_START + VESSEL_SPEED * t_track
    errors      = np.abs(p_track - gt_at_track)

    ax3.plot(t_track, errors, linewidth=1.2, color='purple')
    ax3.axhline(np.mean(errors), color='red', linestyle='--',
                label=f'Mean error: {np.mean(errors):.1f}m')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position error (m)')
    ax3.set_title('Tracker position error vs ground truth')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    print(f"\nTracking results:")
    print(f"  Mean position error : {np.mean(errors):.2f} m")
    print(f"  Std position error  : {np.std(errors):.2f} m")
    print(f"  Max position error  : {np.max(errors):.2f} m")
else:
    ax3.text(0.5, 0.5, 'No tracks found\nTry lowering min_points threshold',
             ha='center', va='center', transform=ax3.transAxes)

plt.suptitle('Step 5: GM-PHD Tracking on NTF Spatial Factors', fontsize=13, fontweight='bold')
plt.savefig('COMP6228-IRP/img/05_tracking.png', dpi=150, bbox_inches='tight')
plt.show()