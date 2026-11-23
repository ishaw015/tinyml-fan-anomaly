
import numpy as np

val_errors = np.load("data/autoencoder_test_errors.npy")
test_errors = np.load("data/autoencoder_test_errors.npy")

# Recalculate validation errors
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/autoencoder_float32.keras"
)

X_val = np.load("data/val_healthy.npy")
reconstructed = model.predict(X_val, verbose=0)

val_errors = np.mean(
    (X_val - reconstructed) ** 2, axis=1
)

val_speed = np.load("data/val_speed.npy")
test_speed = np.load("data/test_speed.npy")
test_y = np.load("data/test_y.npy")

bands = [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]

print("SPEED-AWARE THRESHOLDS")
print("-" * 45)

for low, high in bands:
    mask = (val_speed >= low) & (val_speed < high)

    if mask.sum() < 20:
        print(f"{low}-{high} Hz: Insufficient validation data")
        continue

    threshold = np.percentile(val_errors[mask], 99)

    test_mask = (
        (test_speed >= low) &
        (test_speed < high)
    )

    predictions = test_errors[test_mask] > threshold
    labels = test_y[test_mask]

    healthy = labels == 0
    faulty = labels == 1

    false_alarm = (
        np.mean(predictions[healthy]) * 100
        if healthy.any() else None
    )

    detection = (
        np.mean(predictions[faulty]) * 100
        if faulty.any() else None
    )

    print(f"\n{low}-{high} Hz")
    print(f"Threshold: {threshold:.5f}")
    print(f"Validation samples: {mask.sum()}")

    if false_alarm is not None:
        print(f"False alarm: {false_alarm:.2f}%")

    if detection is not None:
        print(f"Fault detection: {detection:.2f}%")
