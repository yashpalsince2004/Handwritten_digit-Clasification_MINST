import tensorflow as tf
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize: 0–255 → 0–1
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training data shape:", X_train.shape)  # (60000, 28, 28)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 2D → 1D
    keras.layers.Dense(256, activation='relu'), # Hidden layer
    keras.layers.Dense(10, activation='softmax') # Output layer (10 classes)
])
# Compile model
model.compile(optimizer='adam',  # Adam optimizer
              loss='sparse_categorical_crossentropy',  # Multi-class loss
              metrics=['accuracy'])  # Track accuracy
# Train model
model.fit(X_train, y_train, epochs=5)  # Train for 5 epochs

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

import matplotlib.pyplot as plt

predictions = model.predict(X_test)

plt.imshow(X_test[0], cmap='gray')
plt.title(f"Predicted: {tf.argmax(predictions[0]).numpy()}, Actual: {y_test[0]}")
plt.show()