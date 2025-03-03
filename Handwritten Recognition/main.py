import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to match CNN input format
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Print dataset information
print("Training data shape: ", x_train.shape)
print("Testing data shape: ", x_test.shape)
print("Data type of training data: ", x_train.dtype)
print("Data type of testing data: ", x_test.dtype)
print("Min pixel value in x_train:", x_train.min())
print("Max pixel value in x_train:", x_train.max())
print("Sample label: ", y_train[0])

# Define CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Define input shape

    # First convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Reduces image size

    # Second convolutional layers
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Reduces image size

    # Flatten and fully connected layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # Output layer
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=5,  # Iterations
    batch_size=32,  # How many images to process before updating weights
    validation_data=(x_test, y_test)  # Checks performance on test data
)

# Predict multiple test images
image_indices = [37, 129, 453, 789, 1023, 1578, 2034, 2789, 3412, 4198]  # Select random indices
images = x_test[image_indices]
labels = y_test[image_indices]

# Reshape for prediction
images = images.reshape(-1, 28, 28, 1)

# Make predictions
predictions = model.predict(images)

# Print results for each image
print("\nIndex | Predicted Digit | Confidence Score | Actual Label")
print("-" * 55)
for i, idx in enumerate(image_indices):
    predicted_digit = np.argmax(predictions[i])  # Get highest probability digit
    confidence = np.max(predictions[i])  # Get confidence score
    actual_label = labels[i]  # Get actual digit
    print(f"{idx:5d} | {predicted_digit:15d} | {confidence:.4f}         | {actual_label}")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Generate Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to predicted labels
cm = confusion_matrix(y_test, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Compute per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Plot per-digit classification accuracy
plt.figure(figsize=(8,5))
sns.barplot(x=np.arange(10), y=class_accuracy)
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.title("Per-Digit Classification Accuracy")
plt.ylim(0.9, 1.0)  # Focus on high accuracy range
plt.show()
