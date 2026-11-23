
"""
Initial MaFaulDa real-data loader test.
Reads one CSV, selects an accelerometer channel,
and downsamples from 50 kHz to 1 kHz.
"""

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

# Dataset configuration
RAW_FS = 50000
TARGET_FS = 1000

# CSV column 3 = zero-based index 2
# Underhang radial accelerometer
CHANNEL = 2


def load_recording(path):
    """Load and preprocess one MaFaulDa recording."""

    df = pd.read_csv(path, header=None)

    if df.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns, found {df.shape[1]}"
        )

    if df.isna().any().any():
        raise ValueError("CSV contains missing values")

    signal = df.iloc[:, CHANNEL].to_numpy(dtype=np.float32)

    # Remove DC offset
    signal = signal - signal.mean()

    # Downsample 50,000 Hz -> 1,000 Hz.
    # resample_poly applies anti-alias filtering.
    signal_1k = resample_poly(
        signal,
        up=1,
        down=50
    ).astype(np.float32)

    return signal_1k


# Feature extraction configuration
WIN = 256
HOP = 128
NBINS = 64

HANN = np.hanning(WIN).astype(np.float32)


def make_windows(signal):
    """Split a signal into overlapping windows."""

    n_windows = (len(signal) - WIN) // HOP + 1

    return np.stack([
        signal[i * HOP:i * HOP + WIN]
        for i in range(n_windows)
    ]).astype(np.float32)


def extract_features(windows):
    """
    Convert windows into 64-bin log FFT features.
    Must be mirrored in ESP32 C code.
    """

    # Remove DC
    windows = windows - windows.mean(
        axis=1, keepdims=True
    )

    # Apply Hann window and calculate FFT
    mag = np.abs(
        np.fft.rfft(windows * HANN, axis=1)
    )[:, :128]

    # Average neighbouring frequency bins
    mag = mag.reshape(len(windows), 64, 2).mean(axis=2)

    # Log transformation
    features = np.log1p(mag * 100.0)

    return features.astype(np.float32)
from pathlib import Path

SEED = 42


def get_speed(path):
    """Extract rotation frequency from the CSV filename."""
    return float(Path(path).stem)


def generate_dataset():
    rng = np.random.default_rng(SEED)

    normal_files = sorted(
        Path("raw_data/normal").glob("*.csv")
    )

    fault_files = sorted(
        Path("raw_data/imbalance").glob("*/*.csv")
    )

    if len(normal_files) != 49 or len(fault_files) != 333:
        raise ValueError(
            f"Unexpected dataset size: "
            f"{len(normal_files)} healthy, "
            f"{len(fault_files)} faulty"
        )

    # Shuffle healthy recordings reproducibly.
    normal_files = list(normal_files)
    rng.shuffle(normal_files)

    # Split by recording, not by individual windows.
    train_files = normal_files[:29]
    val_files = normal_files[29:39]
    test_normal_files = normal_files[39:]

    def process_files(files):
        all_features = []
        all_rms = []
        all_speeds = []

        for path in files:
            signal = load_recording(path)
            windows = make_windows(signal)

            features = extract_features(windows)
            window_rms = np.sqrt(
                np.mean(windows ** 2, axis=1)
            )

            all_features.append(features)
            all_rms.append(window_rms)

            speed = get_speed(path)
            all_speeds.extend([speed] * len(windows))

        return (
            np.concatenate(all_features).astype(np.float32),
            np.concatenate(all_rms).astype(np.float32),
            np.array(all_speeds, dtype=np.float32)
        )

    print("Processing healthy training recordings...")
    train_X, _, _ = process_files(train_files)

    print("Processing healthy validation recordings...")
    val_X, val_rms, val_speed = process_files(val_files)

    print("Processing healthy test recordings...")
    test_normal_X, test_normal_rms, test_normal_speed = (
        process_files(test_normal_files)
    )

    print("Processing faulty recordings...")
    test_fault_X, test_fault_rms, test_fault_speed = (
        process_files(fault_files)
    )

    # Normalize using training data only.
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std < 1e-6] = 1.0

    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    test_normal_X = (test_normal_X - mean) / std
    test_fault_X = (test_fault_X - mean) / std

    # Combine healthy and faulty test data.
    test_X = np.concatenate(
        [test_normal_X, test_fault_X]
    )

    test_y = np.concatenate([
        np.zeros(len(test_normal_X), dtype=np.int32),
        np.ones(len(test_fault_X), dtype=np.int32)
    ])

    test_speed = np.concatenate([
        test_normal_speed,
        test_fault_speed
    ])

    test_rms = np.concatenate([
        test_normal_rms,
        test_fault_rms
    ])

    # Healthy validation RMS threshold.
    rms_threshold = np.percentile(val_rms, 99)

    # Save model inputs and evaluation data.
    Path("data").mkdir(exist_ok=True)

    np.save("data/train_healthy.npy", train_X)
    np.save("data/val_healthy.npy", val_X)
    np.save("data/test_X.npy", test_X)
    np.save("data/test_y.npy", test_y)
    np.save("data/test_speed.npy", test_speed)
    np.save("data/test_rms.npy", test_rms)
    np.save("data/norm.npy", np.stack([mean, std]))
    np.save("data/rms_threshold.npy", np.array(rms_threshold))

    # Save one representative raw window for firmware testing.
    sample_signal = load_recording(test_normal_files[0])
    sample_windows = make_windows(sample_signal)
    np.save("data/sample_windows.npy", sample_windows[:5])
    np.save("data/val_speed.npy", val_speed)

    print("\nDataset generation complete.")
    print("Training features:", train_X.shape)
    print("Validation features:", val_X.shape)
    print("Test features:", test_X.shape)
    print("Healthy test windows:", len(test_normal_X))
    print("Faulty test windows:", len(test_fault_X))
    print("RMS threshold:", rms_threshold)
    
if __name__ == "__main__":
    
    generate_dataset()
    path = "raw_data/normal/12.288.csv"

    signal = load_recording(path)

    print("File:", path)
    print("Original sample rate:", RAW_FS, "Hz")
    print("Target sample rate:", TARGET_FS, "Hz")
    print("Downsampled samples:", len(signal))
    print("Duration:", len(signal) / TARGET_FS, "seconds")
    print("Mean:", signal.mean())
    print("Standard deviation:", signal.std())
    print("Minimum:", signal.min())
    print("Maximum:", signal.max())
