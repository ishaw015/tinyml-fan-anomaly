
import numpy as np
import tensorflow as tf

# Load trained float32 model
model = tf.keras.models.load_model(
    "model/autoencoder_float32.keras"
)

# Load healthy representative data
X_train = np.load("data/train_healthy.npy")

# INT8 post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

def representative_dataset():
    for i in range(min(300, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save
with open("model/autoencoder_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("INT8 conversion successful!")
print("Model size:", len(tflite_model), "bytes")
