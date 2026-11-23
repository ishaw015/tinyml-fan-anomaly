
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load real MaFaulDa features
X_train = np.load("data/train_healthy.npy")
X_val = np.load("data/val_healthy.npy")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# Autoencoder architecture
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="linear")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# Train healthy-only autoencoder
history = model.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Save model
model.save("model/autoencoder_float32.keras")

print("\nTraining complete.")
print("Best validation loss:", min(history.history["val_loss"]))
