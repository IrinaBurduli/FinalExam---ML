import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize

# Load CIFAR car images
from tensorflow.keras.datasets import cifar10
(_, _), (x_cars, _) = cifar10.load_data()
x_cars = x_cars.astype('float32') / 255.0

# Load gun images
gun_folder = "gun_images"
gun_images = []
for filename in os.listdir(gun_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = io.imread(os.path.join(gun_folder, filename))
        image_resized = resize(image, (32, 32), anti_aliasing=True)
        gun_images.append(image_resized)
gun_images = np.array(gun_images)
gun_images = gun_images.astype('float32') / 255.0

# Create labels for car and gun images (0 for car, 1 for gun)
y_cars = np.zeros(len(x_cars))
y_guns = np.ones(len(gun_images))

# Concatenate car and gun images and labels
x_all = np.concatenate((x_cars, gun_images), axis=0)
y_all = np.concatenate((y_cars, y_guns), axis=0)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

# Define CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
