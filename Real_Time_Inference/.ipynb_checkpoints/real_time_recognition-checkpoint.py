import cv2
import numpy as np
import tensorflow as tf
import json
import time  # Add delay handling

# 📌 Step 1: Load Model
model_path = "model.keras"  # Ensure the model file exists
try:
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Fix potential issues
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to load model! {e}")
    exit()

# Extract correct input shape dynamically
_, img_height, img_width, img_channels = model.input_shape
print(f"✅ Model expects input shape: ({img_height}, {img_width}, {img_channels})")

# 📌 Step 2: Load Labels
labels_path = "labels.json"
try:
    with open(labels_path, "r") as f:
        labels = json.load(f)
    if not labels:
        print("⚠️ WARNING: labels.json is empty or incorrect!")
    print("✅ Labels loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: labels.json not found!")
    labels = {}

# 📌 Step 3: Start Webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("❌ ERROR: Webcam not detected! Check your camera settings.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ WARNING: Could not read frame! Retrying in 1 second...")
        time.sleep(1)
        continue  # Try again instead of breaking immediately

    # 📌 Step 4: Preprocess the Frame
    img = cv2.resize(frame, (img_width, img_height))  # Resize dynamically
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if required
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # 📌 Step 5: Predict
    predictions = model.predict_on_batch(img)  # Optimized prediction
    class_id = int(np.argmax(predictions))  # Convert to int
    confidence = float(np.max(predictions))  # Convert to float

    # 📌 Step 6: Display Results
    label = labels.get(str(class_id), "Unknown")  # Ensure correct lookup
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Traffic Sign Recognition", frame)

    # 📌 Step 7: Exit on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
