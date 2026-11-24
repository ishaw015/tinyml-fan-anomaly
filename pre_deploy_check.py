"""
Last check before ESP32 deployment.

Part 1: tensor details of model/autoencoder_qat_int8.tflite (shapes, scales,
        zero points) - needed to write the C inference code.
Part 2: the 3-consecutive-window alert rule, evaluated per RECORDING, for
        the QAT autoencoder, the RMS rule, and a hybrid (either fires).

Run from the project folder:  python pre_deploy_check.py
"""
import re
from pathlib import Path
import numpy as np

MODEL = "model/autoencoder_qat_int8.tflite"
WIN_PER_FILE = 38
BANDS = [10, 20, 30, 40, 50, 60, 70]
LABELS = [f"{BANDS[i]}-{BANDS[i+1]}" for i in range(len(BANDS) - 1)]

# ------------------------------------------------------------- Part 1
print("=" * 60)
print("PART 1: model tensor details")
print("=" * 60)
try:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    print(f"File: {MODEL}  ({Path(MODEL).stat().st_size} bytes)")
    for kind, details in [("INPUT", interp.get_input_details()),
                          ("OUTPUT", interp.get_output_details())]:
        for d in details:
            scale, zp = d["quantization"]
            print(f"{kind}: name={d['name']} shape={d['shape'].tolist()} "
                  f"dtype={np.dtype(d['dtype']).name} scale={scale:.8f} zero_point={zp}")
    ops = {t["name"] for t in interp.get_tensor_details()}
    print(f"Total tensors in model: {len(ops)}")
except Exception as e:
    print(f"Could not read the model ({type(e).__name__}: {e})")
    print("Run this part inside your .venv where TensorFlow is installed.")

# ------------------------------------------------------------- Part 2
print("\n" + "=" * 60)
print("PART 2: 3-consecutive-window alert rule, per recording")
print("=" * 60)
err = np.load("data/qat_int8_test_errors.npy").ravel()
thr = float(np.ravel(np.load("data/qat_int8_threshold.npy"))[0])
rms = np.load("data/test_rms.npy").ravel()
rms_thr = float(np.ravel(np.load("data/rms_threshold.npy"))[0])
y = np.load("data/test_y.npy")
spd = np.load("data/test_speed.npy")

n_healthy = int((y == 0).sum())
files = sorted(Path("raw_data/imbalance").glob("*/*.csv"))
assert n_healthy % WIN_PER_FILE == 0
assert len(files) * WIN_PER_FILE == len(y) - n_healthy, "window count mismatch"
w_label = np.array([p.parent.name for p in files])
f_speed = np.array([float(p.stem) for p in files])
assert np.allclose(np.repeat(f_speed, WIN_PER_FILE), spd[n_healthy:], atol=1e-3), \
    "alignment failed, do not trust the tables"
print("Window-to-file alignment verified: True")

methods = {
    "QAT AE": err > thr,
    "RMS": rms > rms_thr,
    "Hybrid": (err > thr) | (rms > rms_thr),
}


def alert_per_file(flags):
    """True for a recording if any 3 consecutive windows are flagged."""
    f = flags.reshape(-1, WIN_PER_FILE)
    return (f[:, :-2] & f[:, 1:-1] & f[:, 2:]).any(axis=1)


healthy_files = n_healthy // WIN_PER_FILE
res = {}
for name, fl in methods.items():
    res[name] = (alert_per_file(fl[:n_healthy]), alert_per_file(fl[n_healthy:]))

print(f"\nWindow-level false alarms on healthy test windows (no 3-window rule):")
for name, fl in methods.items():
    print(f"  {name:<8}{100 * fl[:n_healthy].mean():6.2f}%")

print(f"\nRecording-level result with the 3-window rule "
      f"({healthy_files} healthy recordings, {len(files)} faulty):")
print(f"{'method':<9}{'healthy files alarmed':>24}{'faulty files caught':>22}")
for name in methods:
    h, f = res[name]
    print(f"{name:<9}{f'{h.sum()}/{len(h)}':>24}{f'{f.sum()}/{len(f)} ({100 * f.mean():.1f}%)':>22}")


def num(s):
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else 1e9


weights = sorted(set(w_label), key=num)
print("\nFaulty recordings caught (3-window rule), % by imbalance weight")
print(f"{'weight':<9}{'files':>6}" + "".join(f"{n:>10}" for n in methods))
for w in weights:
    m = w_label == w
    print(f"{w:<9}{m.sum():>6}" + "".join(f"{100 * res[n][1][m].mean():>9.1f}%" for n in methods))

band = np.digitize(f_speed, BANDS) - 1
print("\nFaulty recordings caught (3-window rule), % by speed band (Hz)")
print(f"{'band':<9}{'files':>6}" + "".join(f"{n:>10}" for n in methods))
for i, lab in enumerate(LABELS):
    m = band == i
    if m.sum():
        print(f"{lab:<9}{m.sum():>6}" + "".join(f"{100 * res[n][1][m].mean():>9.1f}%" for n in methods))