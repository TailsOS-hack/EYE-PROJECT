import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define dataset directory
TRAIN_DIR = "train"

# Function to classify images based on filename ending
def categorize_image(filename):
    if filename.endswith("A.jpg") or filename.endswith("A.png"):
        return "AMD"
    elif filename.endswith("N.jpg") or filename.endswith("N.png"):
        return "Normal"
    elif filename.endswith("D.jpg") or filename.endswith("D.png"):
        return "Diabetic Retinopathy"
    elif filename.endswith("G.jpg") or filename.endswith("G.png"):
        return "Glaucoma"
    else:
        return "Cataract"

# Create labeled dataset
image_paths = []
labels = []

for filename in os.listdir(TRAIN_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        image_paths.append(os.path.join(TRAIN_DIR, filename))
        labels.append(categorize_image(filename))

# Convert labels to categorical values
label_dict = {label: idx for idx, label in enumerate(set(labels))}
labels_encoded = np.array([label_dict[label] for label in labels])

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Generate training and validation sets
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(len(label_dict), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the trained model
model.save("keras_model.h5")

# Save class labels
with open("labels.txt", "w") as f:
    for label in label_dict:
        f.write(f"{label}\n")

print("Training complete. Model and labels saved.")
