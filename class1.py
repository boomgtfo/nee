import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to load images and labels from a flat directory
def load_flat_data(data_dir):
    image_files = []

    # Iterate over all files in the directory
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            image_files.append(file_path)

    # Check the number of images loaded
    print("Number of images:", len(image_files))

    # Load images into numpy arrays
    images = []
    for file_path in image_files:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)  # Assuming RGB images
        img = tf.image.resize(img, (224, 224))
        images.append(img.numpy())

    # Dummy labels (e.g., all set to 0 since no explicit labels exist)
    images = np.array(images)
    labels = np.zeros(len(images))  # Replace with actual labels if available

    return images, labels

# Function to create a CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Specify your directory
data_dir = './images'  # Update with the correct path to your images

# Load data
images, labels = load_flat_data(data_dir)

# Check loaded data
print("Number of images:", len(images))
print("Number of labels:", len(labels))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Create the model
model = create_model((224, 224, 3), len(np.unique(labels)))

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Predict using the model
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(labels)))
plt.xticks(tick_marks, np.unique(labels))
plt.yticks(tick_marks)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Show numbers in each cell
for i in range(len(np.unique(labels))):
    for j in range(len(np.unique(labels))):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()
