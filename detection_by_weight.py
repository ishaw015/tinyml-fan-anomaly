"""
Detection rate by imbalance weight (folder name) and speed band.
Uses the saved QAT INT8 test errors, so nothing is retrained.

Run from the project folder:  python detection_by_weight.py
"""
import re
from pathlib import Path
import numpy as np

WIN_PER_FILE = 38
BANDS = [10, 20, 30, 40, 50, 60, 70]
LABELS = [f"{BANDS[i]}-{BANDS[i+1]}" for i in range(len(BANDS) - 1)]

err = np.load("data/qat_int8_test_errors.npy").ravel()
thr = float(np.ravel(np.load("data/qat_int8_threshold.npy"))[0])
y = np.load("data/test_y.npy")
spd = np.load("data/test_speed.npy")

print(f"errors: {len(err)} | test_y: {len(y)} | threshold: {thr:.4f}")
assert len(err) == len(y), "error array and test_y have different lengths"

n_healthy = int((y == 0).sum())
files = sorted(Path("raw_data/imbalance").glob("*/*.csv"))
assert len(files) * WIN_PER_FILE == len(y) - n_healthy, "window count mismatch"

# one entry per faulty window: weight folder and the file's speed
w_label = np.repeat([p.parent.name for p in files], WIN_PER_FILE)
f_speed = np.repeat([float(p.stem) for p in files], WIN_PER_FILE)

# alignment check: our rebuilt speeds must equal the saved test_speed
aligned = np.allclose(f_speed, spd[n_healthy:], atol=1e-3)
print(f"Window-to-file alignment verified: {aligned}")
assert aligned, "alignment failed, do not trust the table below"

flag = err[n_healthy:] > thr
band = np.digitize(f_speed, BANDS) - 1


def num(s):
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else 1e9


weights = sorted(set(w_label), key=num)
print("\nDetection % (windows flagged) by weight and speed band (Hz)")
print(f"{'weight':<9}{'files':>6}" + "".join(f"{l:>9}" for l in LABELS) + f"{'ALL':>9}")
for w in weights:
    mw = w_label == w
    nfiles = mw.sum() // WIN_PER_FILE
    row = f"{w:<9}{nfiles:>6}"
    for i in range(len(LABELS)):
        m = mw & (band == i)
        row += f"{100 * flag[m].mean():>8.1f}%" if m.sum() else f"{'-':>9}"
    row += f"{100 * flag[mw].mean():>8.1f}%"
    print(row)

print("\nFiles behind each cell are in the 'files' column overall; "
      "cells with only a few files are unreliable.")
cnt = f"{'files/band':<15}"
for i in range(len(LABELS)):
    cnt += f"{(band == i).sum() // WIN_PER_FILE:>9}"
print(cnt)