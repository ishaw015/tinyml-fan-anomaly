
import numpy as np

val_errors = np.load("data/autoencoder_test_errors.npy")
y = np.load("data/test_y.npy")

# Recalculate validation errors from the model
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/autoencoder_float32.keras"
)

X_val = np.load("data/val_healthy.npy")
val_reconstructed = model.predict(X_val, verbose=0)

val_errors = np.mean(
    (X_val - val_reconstructed) ** 2,
    axis=1
)

test_errors = np.load("data/autoencoder_test_errors.npy")

print("THRESHOLD ANALYSIS")
print("-" * 55)

for percentile in [90, 95, 97, 98, 99]:
    threshold = np.percentile(val_errors, percentile)

    predictions = (test_errors > threshold).astype(int)

    healthy = y == 0
    faulty = y == 1

    false_alarm = np.mean(predictions[healthy] == 1) * 100
    detection = np.mean(predictions[faulty] == 1) * 100

    print(f"\nValidation percentile: {percentile}")
    print(f"Threshold: {threshold:.5f}")
    print(f"False alarm rate: {false_alarm:.2f}%")
    print(f"Fault detection rate: {detection:.2f}%")

print("\nAnalysis complete.")
