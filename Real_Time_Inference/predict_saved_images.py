import os
import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image

# 📌 Step 1: Load the Trained Model
model_path = "../Real_Time_Inference/model.keras"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
else:
    print("❌ ERROR: Model file NOT found!")
    exit()

# 📌 Step 2: Load Class Labels
labels_path = "../Real_Time_Inference/labels.json"

if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        labels = json.load(f)
        print("✅ Labels loaded successfully!")
else:
    print("❌ ERROR: Labels file NOT found!")
    exit()

# 📌 Step 3: Define Input Image Folder
image_folder = "../Real_Time_Inference/Saved_Images"
if not os.path.exists(image_folder):
    print("❌ ERROR: Image folder not found!")
    exit()

# 📌 Step 4: Define Image Preprocessing Function
def preprocess_image(img_path, img_size=(48, 48)):
    img = image.load_img(img_path, target_size=img_size)  # Resize to model input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize
    return img

# 📌 Step 5: Process Each Image and Make Predictions
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    
    # Skip non-image files
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    sign_label = labels.get(str(class_index), "Unknown Sign")
    
    print(f"🖼️ {img_name} → 🛑 Predicted: {sign_label} ({confidence:.2f})")

