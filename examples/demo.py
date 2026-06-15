"""Demo: Complete segyml workflow.

Generate synthetic seismic data, write SEG-Y, read back, visualize.
"""
import numpy as np
import segyml

# 1. Create synthetic seismic data (Ricker wavelets)
def ricker(t, f=25):
    """Ricker wavelet."""
    return (1 - 2 * np.pi**2 * f**2 * t**2) * np.exp(-np.pi**2 * f**2 * t**2)

n_traces, n_samples = 100, 512
dt = 0.004  # 4ms sample interval
t = np.arange(n_samples) * dt
wavelet = ricker(t - 0.2, f=25)

# Generate traces with varying amplitude and time shift
data = np.zeros((n_traces, n_samples), dtype=np.float32)
inlines = np.arange(1000, 1000 + n_traces, dtype=np.int32)
crosslines = np.full(n_traces, 500, dtype=np.int32)

for i in range(n_traces):
    shift = int(20 * np.sin(i * 0.1) + 10)
    amp = 1.0 + 0.5 * np.sin(i * 0.05)
    if shift + n_samples < n_samples:
        data[i, shift:] = wavelet[:n_samples - shift] * amp
    else:
        data[i, :] = np.roll(wavelet, shift % n_samples) * amp

print(f"Generated: {n_traces} traces × {n_samples} samples")

# 2. Save as SEG-Y
segyml.save(
    "demo.sgy", data, dt=int(dt * 1e6),
    inline=inlines, crossline=crosslines,
    text_header="C01 SEG-Y Demo - segyml synthetic data"
)
print("Written: demo.sgy")

# 3. Load back
loaded, hdr = segyml.load("demo.sgy")
print(f"Loaded: {loaded.shape}")
print(f"Sample interval: {hdr['binary_header']['dt']} µs")
print(f"First trace inline: {hdr['traces'][0]['inline']}")

# 4. Visualize
segyml.wiggle(loaded[:, :50], dt=dt, save_path="demo_wiggle.png", title="SEG-Y Demo - Wiggle Plot")
print("Saved: demo_wiggle.png")

segyml.image(loaded, dt=dt, save_path="demo_image.png", title="SEG-Y Demo - Seismic Section")
print("Saved: demo_image.png")

print("\nDone! Check demo.sgy, demo_wiggle.png, demo_image.png")
