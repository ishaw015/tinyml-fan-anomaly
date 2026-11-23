
import numpy as np
import tensorflow as tf

# Load test and validation data
X_val = np.load("data/val_healthy.npy")
X_test = np.load("data/test_X.npy")
y_test = np.load("data/test_y.npy")
test_speed = np.load("data/test_speed.npy")

# Load QAT INT8 model
interpreter = tf.lite.Interpreter(
    model_path="model/autoencoder_qat_int8.tflite"
)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

input_scale, input_zero = input_info["quantization"]
output_scale, output_zero = output_info["quantization"]

print("Input dtype:", input_info["dtype"])
print("Output dtype:", output_info["dtype"])

def predict_errors(X):
    errors = []

    for sample in X:
        sample = sample.reshape(1, 64)

        quantized = np.round(
            sample / input_scale + input_zero
        )
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_info["index"], quantized)
        interpreter.invoke()

        output = interpreter.get_tensor(output_info["index"])
        output = (
            output.astype(np.float32) - output_zero
        ) * output_scale

        error = np.mean((sample - output) ** 2)
        errors.append(error)

    return np.array(errors)

# Calculate validation threshold using healthy data only
val_errors = predict_errors(X_val)
threshold = np.percentile(val_errors, 99)

# Evaluate test data
test_errors = predict_errors(X_test)
predictions = test_errors > threshold

healthy = y_test == 0
faulty = y_test == 1

false_alarm = np.mean(predictions[healthy]) * 100
detection = np.mean(predictions[faulty]) * 100

print("\nQAT INT8 RESULTS")
print("Threshold:", threshold)
print("False alarm rate:", round(false_alarm, 2), "%")
print("Fault detection rate:", round(detection, 2), "%")

print("\nRESULTS BY SPEED BAND")

bands = [(10, 20), (20, 30), (30, 40),
         (40, 50), (50, 60), (60, 70)]

for low, high in bands:
    mask = (test_speed >= low) & (test_speed < high)
    normal_mask = mask & healthy
    fault_mask = mask & faulty

    if np.sum(fault_mask) == 0:
        continue

    band_detection = np.mean(predictions[fault_mask]) * 100

    if np.sum(normal_mask) > 0:
        band_false_alarm = np.mean(predictions[normal_mask]) * 100
        alarm_text = f"{band_false_alarm:.2f}%"
    else:
        alarm_text = "No healthy samples"

    print(
        f"{low}-{high} Hz | "
        f"Detection: {band_detection:.2f}% | "
        f"False alarm: {alarm_text}"
    )

np.save("data/qat_int8_threshold.npy", threshold)
np.save("data/qat_int8_test_errors.npy", test_errors)

print("\nEvaluation complete. Results saved.")
