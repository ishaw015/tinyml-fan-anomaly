
import numpy as np
from pathlib import Path
from mafaulda_loader import load_recording, make_windows, get_speed

# Reproduce the exact healthy validation split
rng = np.random.default_rng(42)

normal_files = sorted(
    Path("raw_data/normal").glob("*.csv")
)
rng.shuffle(normal_files)

val_files = normal_files[29:39]

# Generate speed metadata in exactly the same order
speeds = []

for path in val_files:
    signal = load_recording(path)
    windows = make_windows(signal)
    speeds.extend([get_speed(path)] * len(windows))

val_speed = np.array(speeds, dtype=np.float32)

# Verify alignment with existing validation features
val_X = np.load("data/val_healthy.npy")

assert len(val_speed) == len(val_X), (
    f"Mismatch: {len(val_speed)} speeds "
    f"for {len(val_X)} validation windows"
)

np.save("data/val_speed.npy", val_speed)

print("Saved data/val_speed.npy")
print("Shape:", val_speed.shape)
print("Speed range:", val_speed.min(), "to", val_speed.max())
print("Validation windows:", len(val_X))
