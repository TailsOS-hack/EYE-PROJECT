import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

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

# Create labeled dataset and organize images into subdirectories
image_paths = []
labels = []

print("Organizing images into subdirectories...")
for filename in os.listdir(TRAIN_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        category = categorize_image(filename)
        category_dir = os.path.join(TRAIN_DIR, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        os.rename(os.path.join(TRAIN_DIR, filename), os.path.join(category_dir, filename))
        image_paths.append(os.path.join(category_dir, filename))
        labels.append(category)

print("Images organized.")

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

print("Generating training and validation sets...")
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

print("Training and validation sets generated.")

# Build the model
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
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

print("Starting model training...")
# Train the model
try:
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        for batch, (images, labels) in enumerate(train_generator):
            print(f"Batch {batch + 1}: images.shape = {images.shape}, labels.shape = {labels.shape}")
        model.fit(train_generator, validation_data=val_generator, epochs=1)
    print("Model training complete.")
except Exception as e:
    print(f"Error during model training: {e}")
    import traceback
    traceback.print_exc()

# Save the trained model
model.save("keras_model.h5")

# Save class labels
with open("labels.txt", "w") as f:
    for label in label_dict:
        f.write(f"{label}\n")

print("Training complete. Model and labels saved.")
