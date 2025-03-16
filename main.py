import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, Toplevel
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load class labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

status = False

def show_instructions():
    instructions = "1. Enter your name.\n2. Select an image for analysis."
    messagebox.showinfo("Instructions", instructions)

def analyze_image():
    global status

    # Get user input
    name = simpledialog.askstring("Input", "Enter your name:")

    # Load trained model
    model = load_model("keras_model.h5", compile=False)

    # Initialize image data array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Select an image file
    file_path = filedialog.askopenfilename(title="Select an image file")
    if not file_path:
        return
    
    # Process the image
    image = Image.open(file_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Store the processed image in the data array
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Generate advice
    advice = ""
    if class_name == "Normal":
        status = True
    elif confidence_score >= 0.80 and not status:
        advice = f"Given the accuracy of the program and the confidence score, we recommend seeking medical advice from a healthcare professional regarding {class_name}."
    else:
        advice = f"{name}, it's advisable to keep monitoring your vision and emotional state over time. If concerns persist, consult a doctor."

    # Display results
    result_window = Toplevel(root)
    result_window.geometry(f"{screen_width}x{screen_height}")
    messagebox.showinfo("Result", f"Predicted Condition: {class_name}\nConfidence Score: {confidence_score * 100:.2f}%\n{advice}")

# Initialize GUI
root = tk.Tk()
root.withdraw()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

show_instructions()
analyze_image()

root.mainloop()
