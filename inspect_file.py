"""
Inspect ONE MaFaulDa-style CSV (8 columns, no header).

Usage:  python inspect_file.py raw_data/normal/12.288.csv
Needs:  numpy, pandas, matplotlib
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSUMED_FS = 50000   # CHECK on the dataset page (some sources say 51200)

path = sys.argv[1]
df = pd.read_csv(path, header=None)
x = df.values.astype(float)
n, c = x.shape
duration = n / ASSUMED_FS

print(f"File: {path}")
print(f"Rows: {n}   Columns: {c}")
print(f"Duration at {ASSUMED_FS} Hz: {duration:.2f} s")
print(f"Any NaN values: {np.isnan(x).any()}\n")

print("col |      mean |       std |       min |       max")
for i in range(c):
    col = x[:, i]
    print(f"{i+1:3d} | {col.mean():9.4f} | {col.std():9.4f} | "
          f"{col.min():9.4f} | {col.max():9.4f}")

# --- rotation frequency from the tachometer (column 1): count rising edges
tach = x[:, 0]
mid = (tach.max() + tach.min()) / 2
edges = np.sum((tach[:-1] < mid) & (tach[1:] >= mid))
print(f"\nTachometer pulses: {edges} -> about {edges / duration:.1f} Hz rotation")

# --- Compare low-frequency and shaft-frequency energy

freqs = np.fft.rfftfreq(n, 1 / ASSUMED_FS)

low_band = (freqs >= 4) & (freqs <= 6)
shaft_band = (freqs >= 10) & (freqs <= 14)

print("\nFrequency band energy comparison:")
print("Channel | 4-6 Hz energy | 10-14 Hz energy | Ratio")

for i in range(1, 7):
    sig = x[:, i] - x[:, i].mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(n)))

    low_energy = np.sum(spec[low_band] ** 2)
    shaft_energy = np.sum(spec[shaft_band] ** 2)

    ratio = low_energy / (shaft_energy + 1e-12)

    print(
        f"Column {i+1:2d} | "
        f"{low_energy:13.2e} | "
        f"{shaft_energy:15.2e} | "
        f"{ratio:.4f}"
    )
# --- plots: first 0.2 s of columns 1-4, and spectrum of column 2 up to 500 Hz
fig, ax = plt.subplots(5, 1, figsize=(9, 11))
k = int(0.2 * ASSUMED_FS)
t = np.arange(k) / ASSUMED_FS
for i in range(4):
    ax[i].plot(t, x[:k, i], lw=0.7)
    ax[i].set_ylabel(f"col {i+1}")
ax[3].set_xlabel("time (s)")
m = freqs <= 500
ax[4].plot(freqs[m], spec[m], lw=0.7)
ax[4].set_xlabel("frequency (Hz)")
ax[4].set_ylabel("col 2 spectrum")
plt.tight_layout()
out = "inspect_plot.png"
plt.savefig(out, dpi=110)
print(f"\nSaved plot to {out}")