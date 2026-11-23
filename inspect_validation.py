
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/autoencoder_float32.keras"
)

X_val = np.load("data/val_healthy.npy")

# Validation speeds are not currently saved separately.
# Reconstruct them from the same deterministic split.
from pathlib import Path

rng = np.random.default_rng(42)
normal_files = sorted(Path("raw_data/normal").glob("*.csv"))
rng.shuffle(normal_files)

val_files = normal_files[29:39]

from mafaulda_loader import load_recording, make_windows, get_speed

speeds = []

for path in val_files:
    signal = load_recording(path)
    windows = make_windows(signal)
    speeds.extend([get_speed(path)] * len(windows))

speeds = np.array(speeds)

reconstructed = model.predict(X_val, verbose=0)
errors = np.mean((X_val - reconstructed) ** 2, axis=1)

print("VALIDATION ERROR BY SPEED")
for low, high in [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]:
    mask = (speeds >= low) & (speeds < high)

    if mask.any():
        print(
            f"{low}-{high} Hz:",
            f"count={mask.sum()},",
            f"95th percentile={np.percentile(errors[mask], 95):.4f},",
            f"99th percentile={np.percentile(errors[mask], 99):.4f}"
        )
    else:
        print(f"{low}-{high} Hz: No validation samples")
