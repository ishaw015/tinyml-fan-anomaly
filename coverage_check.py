"""
Speed-coverage check. Reproduces the file split from mafaulda_loader.py
(same seed, same shuffle) WITHOUT reading any CSV: speed comes from the filename.

Run from the project folder:  python coverage_check.py
"""
from pathlib import Path
import numpy as np

SEED = 42
BANDS = [10, 20, 30, 40, 50, 60, 70]
LABELS = [f"{BANDS[i]}-{BANDS[i+1]} Hz" for i in range(len(BANDS) - 1)]


def speed(p):
    return float(Path(p).stem)


# ---- same split as generate_dataset() ------------------------------------
rng = np.random.default_rng(SEED)
normal = sorted(Path("raw_data/normal").glob("*.csv"))
faulty = sorted(Path("raw_data/imbalance").glob("*/*.csv"))
normal = list(normal)
rng.shuffle(normal)
train, val, test = normal[:29], normal[29:39], normal[39:]

print(f"healthy files: {len(normal)} | faulty files: {len(faulty)}")
print(f"split: train {len(train)}, val {len(val)}, healthy test {len(test)}\n")

# ---- sanity check against what the loader really saved --------------------
try:
    val_saved = np.unique(np.load("data/val_speed.npy"))
    test_saved = np.unique(np.load("data/test_speed.npy")[:380])
    ok_val = np.allclose(sorted(val_saved), sorted({speed(p) for p in val}), atol=1e-3)
    ok_test = np.allclose(sorted(test_saved), sorted({speed(p) for p in test}), atol=1e-3)
    print(f"Matches saved val speeds:  {ok_val}")
    print(f"Matches saved test speeds: {ok_test}\n")
except Exception as e:
    print(f"(Could not compare with saved data: {e})\n")

# ---- speeds of each healthy group -----------------------------------------
for name, files in [("TRAIN", train), ("VAL", val), ("HEALTHY TEST", test)]:
    s = sorted(speed(p) for p in files)
    print(f"{name} speeds (Hz): " + ", ".join(f"{v:.1f}" for v in s))
print()

# ---- file counts per speed band --------------------------------------------
def counts(files):
    s = np.array([speed(p) for p in files])
    idx = np.digitize(s, BANDS) - 1
    return np.bincount(idx[(idx >= 0) & (idx < len(LABELS))], minlength=len(LABELS))

groups = [("healthy train", train), ("healthy val", val),
          ("healthy test", test), ("FAULTY", faulty)]
print(f"{'band':<11}" + "".join(f"{g[0]:>15}" for g in groups))
for i, lab in enumerate(LABELS):
    print(f"{lab:<11}" + "".join(f"{counts(g[1])[i]:>15d}" for g in groups))

# ---- how far is each faulty file from the nearest healthy TRAIN speed? ----
train_s = np.array([speed(p) for p in train])
dist = np.array([np.min(np.abs(train_s - speed(p))) for p in faulty])
fs = np.array([speed(p) for p in faulty])
idx = np.digitize(fs, BANDS) - 1
print("\nDistance from each faulty file's speed to the nearest healthy-TRAIN speed")
print(f"{'band':<11}{'faulty files':>14}{'mean dist (Hz)':>17}{'max dist (Hz)':>16}")
for i, lab in enumerate(LABELS):
    m = idx == i
    if m.sum():
        print(f"{lab:<11}{m.sum():>14d}{dist[m].mean():>17.2f}{dist[m].max():>16.2f}")
    else:
        print(f"{lab:<11}{0:>14d}{'-':>17}{'-':>16}")