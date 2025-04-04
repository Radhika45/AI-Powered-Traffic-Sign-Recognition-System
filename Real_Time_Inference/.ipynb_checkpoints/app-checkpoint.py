import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="AI-Powered Traffic Sign Recognition", page_icon="🚦")

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Function to preprocess image for model input
def preprocess_image(image):
    img = cv2.resize(image, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Apply custom CSS for gradient background and white font color
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(45deg, #4A90E2, #9B59B6, #E91E63);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .title {
        color: white !important;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: white !important;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<h3 class="subtitle"> AI Project</h3>', unsafe_allow_html=True)
st.markdown('<h1 class="title"> 🚦Traffic Sign Recognition System</h1>', unsafe_allow_html=True)


# Open webcam and display live video feed
video_feed = st.empty()

cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Error: Could not capture frame.")
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess image
    input_img = preprocess_image(frame_rgb)

    # Model prediction
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # Get class label
    class_label = labels[str(predicted_class)]

    # Overlay label on frame
    label_text = f"{class_label} ({confidence:.2f})"
    cv2.putText(frame_rgb, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame in Streamlit
    video_feed.image(frame_rgb, channels="RGB")

cap.release()
cv2.destroyAllWindows()