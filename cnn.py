import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. Load and Preprocess the Data
print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to include the channel dimension (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Build the CNN Architecture
model = models.Sequential([
    # Convolutional Layer 1: Detects basic edges/features
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Layer 2: Detects more complex shapes
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Dropout to prevent overfitting
    layers.Dropout(0.25),

    # Flattening to transition to Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Output Layer: 10 units for digits 0-9
    layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
print("\nStarting training...")
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test),
                    batch_size=64)

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')

# 6. Visualization: Plotting Training Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- 7. Test on 5 Random Images from the Test Set ---
def predict_multiple_random_images(n=5):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Pick a random index
        random_index = random.randint(0, len(x_test) - 1)
        img = x_test[random_index]
        actual_label = y_test[random_index]

        # Predict
        prediction = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
        predicted_label = np.argmax(prediction)

        # Plotting the images in a grid
        plt.subplot(1, n, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f"Act: {actual_label} | Pred: {predicted_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

print("\nRunning 5 random predictions...")
predict_multiple_random_images(5)