import numpy as np
import tensorflow as tf
import tf_keras as keras
import keras as keras3
import tensorflow_model_optimization as tfmot

np.random.seed(42)
tf.random.set_seed(42)

# Load healthy-only data
X_train = np.load("data/train_healthy.npy")
X_val = np.load("data/val_healthy.npy")

# Rebuild the same architecture using legacy Keras
def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="linear")
    ])

# Load weights from the trained float32 model
original = keras3.models.load_model(
    "model/autoencoder_float32.keras"
)

qat_model = build_model()

for new_layer, old_layer in zip(
    qat_model.layers, original.layers
):
    new_layer.set_weights(old_layer.get_weights())

# Apply quantization-aware training
qat_model = tfmot.quantization.keras.quantize_model(
    qat_model
)

qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse"
)

qat_model.summary()

# Fine-tune using healthy data only
history = qat_model.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Save QAT model
qat_model.save("model/autoencoder_qat.keras")

qat_model.save_weights("model/autoencoder_qat.weights.h5")
print("QAT weights saved successfully!")

print("\nQAT training complete.")
print("Best validation loss:", min(history.history["val_loss"]))
