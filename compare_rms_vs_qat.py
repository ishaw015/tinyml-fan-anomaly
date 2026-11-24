"""
Fair comparison of the RMS rule vs the QAT INT8 autoencoder.
1) AUC (threshold-free) overall and per imbalance weight
2) The same weight x speed-band detection table, for the RMS rule

Run from the project folder:  python compare_rms_vs_qat.py
"""
import re
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

WIN_PER_FILE = 38
BANDS = [10, 20, 30, 40, 50, 60, 70]
LABELS = [f"{BANDS[i]}-{BANDS[i+1]}" for i in range(len(BANDS) - 1)]

err = np.load("data/qat_int8_test_errors.npy").ravel()
rms = np.load("data/test_rms.npy").ravel()
rms_thr = float(np.ravel(np.load("data/rms_threshold.npy"))[0])
y = np.load("data/test_y.npy")
spd = np.load("data/test_speed.npy")
assert len(err) == len(rms) == len(y)

n_healthy = int((y == 0).sum())
files = sorted(Path("raw_data/imbalance").glob("*/*.csv"))
assert len(files) * WIN_PER_FILE == len(y) - n_healthy
w_label = np.repeat([p.parent.name for p in files], WIN_PER_FILE)
f_speed = np.repeat([float(p.stem) for p in files], WIN_PER_FILE)
assert np.allclose(f_speed, spd[n_healthy:], atol=1e-3), "alignment failed"
print("Window-to-file alignment verified: True\n")


def auc(score, label):
    """Probability a random faulty window scores higher than a random healthy one."""
    r = rankdata(score)
    n1 = int(label.sum())
    n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def num(s):
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else 1e9


# ---- 1) AUC ----------------------------------------------------------------
print(f"Overall AUC   RMS: {auc(rms, y):.3f}   QAT autoencoder: {auc(err, y):.3f}")
print("(0.5 = random guessing, 1.0 = perfect separation)\n")

weights = sorted(set(w_label), key=num)
print("AUC per imbalance weight (that weight's faulty windows vs ALL healthy test windows)")
print(f"{'weight':<9}{'RMS':>8}{'QAT AE':>9}")
hmask = y == 0
for w in weights:
    fm = np.zeros(len(y), bool)
    fm[n_healthy:] = (w_label == w)
    m = hmask | fm
    print(f"{w:<9}{auc(rms[m], y[m]):>8.3f}{auc(err[m], y[m]):>9.3f}")

# ---- 2) RMS detection table at its own threshold ----------------------------
flag = rms[n_healthy:] > rms_thr
band = np.digitize(f_speed, BANDS) - 1
print(f"\nRMS rule detection % by weight and speed band (threshold {rms_thr:.4f})")
print(f"{'weight':<9}" + "".join(f"{l:>9}" for l in LABELS) + f"{'ALL':>9}")
for w in weights:
    mw = w_label == w
    row = f"{w:<9}"
    for i in range(len(LABELS)):
        m = mw & (band == i)
        row += f"{100 * flag[m].mean():>8.1f}%" if m.sum() else f"{'-':>9}"
    row += f"{100 * flag[mw].mean():>8.1f}%"
    print(row)
print(f"\nRMS false alarm on healthy test windows: {100 * (rms[:n_healthy] > rms_thr).mean():.2f}%")