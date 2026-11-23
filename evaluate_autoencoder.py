
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model(
    "model/autoencoder_float32.keras"
)

# Load datasets
X_val = np.load("data/val_healthy.npy")
X_test = np.load("data/test_X.npy")
y_test = np.load("data/test_y.npy")
test_speed = np.load("data/test_speed.npy")

# Calculate reconstruction errors
val_reconstructed = model.predict(X_val, verbose=0)
val_errors = np.mean(
    np.square(X_val - val_reconstructed), axis=1
)

test_reconstructed = model.predict(X_test, verbose=0)
test_errors = np.mean(
    np.square(X_test - test_reconstructed), axis=1
)

# Threshold from healthy validation data
threshold = np.percentile(val_errors, 99)

# Classify anomalies
predictions = (test_errors > threshold).astype(int)

# Evaluation
def report(name, mask):
    actual = y_test[mask]
    predicted = predictions[mask]

    healthy = actual == 0
    faulty = actual == 1

    print(f"\n{name}")
    print("Healthy windows:", healthy.sum())
    print("Faulty windows:", faulty.sum())

    if healthy.sum():
        false_alarm = np.mean(predicted[healthy] == 1)
        print("False alarm rate:", f"{false_alarm * 100:.2f}%")

    if faulty.sum():
        detection = np.mean(predicted[faulty] == 1)
        print("Fault detection rate:", f"{detection * 100:.2f}%")

print("Validation error mean:", val_errors.mean())
print("Validation error 99th percentile:", threshold)

report("OVERALL", np.ones(len(y_test), dtype=bool))

for low, high in [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]:
    mask = (test_speed >= low) & (test_speed < high)
    report(f"SPEED {low}-{high} Hz", mask)

# Save threshold and errors for later use
np.save("data/autoencoder_threshold.npy", np.array(threshold))
np.save("data/autoencoder_test_errors.npy", test_errors)

print("\nEvaluation complete.")
