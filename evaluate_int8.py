
import numpy as np
import tensorflow as tf

# Load INT8 model
interpreter = tf.lite.Interpreter(
    model_path="model/autoencoder_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input details:", input_details["dtype"])
print("Output details:", output_details["dtype"])

# Load data
X_val = np.load("data/val_healthy.npy")
X_test = np.load("data/test_X.npy")
y_test = np.load("data/test_y.npy")
test_speed = np.load("data/test_speed.npy")

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

def predict_int8(X):
    results = []

    for sample in X:
        x = sample.reshape(1, -1)

        x_quant = np.round(
            x / input_scale + input_zero
        )

        x_quant = np.clip(
            x_quant, -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"], x_quant
        )
        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        output_float = (
            output.astype(np.float32) - output_zero
        ) * output_scale

        results.append(output_float[0])

    return np.array(results)

# Validation reconstruction errors
val_reconstructed = predict_int8(X_val)
val_errors = np.mean(
    (X_val - val_reconstructed) ** 2, axis=1
)

# Test reconstruction errors
test_reconstructed = predict_int8(X_test)
test_errors = np.mean(
    (X_test - test_reconstructed) ** 2, axis=1
)

# Same 99th-percentile threshold method
threshold = np.percentile(val_errors, 99)
predictions = (test_errors > threshold).astype(int)

print("\nINT8 threshold:", threshold)

def report(name, mask):
    actual = y_test[mask]
    predicted = predictions[mask]

    healthy = actual == 0
    faulty = actual == 1

    print(f"\n{name}")

    if healthy.any():
        print(
            "False alarm rate:",
            f"{100*np.mean(predicted[healthy] == 1):.2f}%"
        )

    if faulty.any():
        print(
            "Fault detection rate:",
            f"{100*np.mean(predicted[faulty] == 1):.2f}%"
        )

report("OVERALL", np.ones(len(y_test), dtype=bool))

for low, high in [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]:
    mask = (test_speed >= low) & (test_speed < high)
    report(f"SPEED {low}-{high} Hz", mask)

np.save("data/int8_test_errors.npy", test_errors)
np.save("data/int8_threshold.npy", np.array(threshold))

print("\nINT8 evaluation complete.")
