
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

# Rebuild the original autoencoder architecture
def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="linear")
    ])

# Apply QAT wrappers
model = build_model()
qat_model = tfmot.quantization.keras.quantize_model(model)

# Load separately saved QAT weights
qat_model.load_weights("model/autoencoder_qat.weights.h5")

print("QAT model and weights loaded successfully!")

# Convert to fully INT8 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Representative dataset for INT8 calibration
X_train = __import__("numpy").load("data/train_healthy.npy")

def representative_dataset():
    for i in range(min(300, len(X_train))):
        yield [X_train[i:i+1].astype("float32")]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model/autoencoder_qat_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("QAT INT8 conversion successful!")
print("Model size:", len(tflite_model), "bytes")
