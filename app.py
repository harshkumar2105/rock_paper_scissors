import streamlit as st
import requests
import cv2
import base64
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image
import io

# Roboflow API Config
ROBOFLOW_API_KEY = "hVps0XPzHyrftbIqVfpn"
ROBOFLOW_MODEL = "rock_paper_scissor"  # Example: "rock-paper-scissors"
ROBOFLOW_VERSION = 2

# Roboflow Inference URL
ROBOFLOW_URL = f"https://infer.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

st.title("🖐️ Rock Paper Scissors Classifier (Roboflow + Webcam)")

st.subheader("📤 Upload an Image (Optional)")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = requests.post(ROBOFLOW_URL, data=img_bytes, headers={"Content-Type": "application/x-www-form-urlencoded"})

    if response.status_code == 200:
        result = response.json()
        pred_class = result['predictions'][0]['class']
        confidence = result['predictions'][0]['confidence']
        st.success(f"Prediction: **{pred_class.upper()}** ({confidence*100:.2f}%)")
    else:
        st.error("Prediction failed. Check API key or image format.")

st.subheader("📷 Live Webcam Prediction")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((224, 224))  # Resize to match Roboflow input

        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Send to Roboflow
        response = requests.post(ROBOFLOW_URL, data=img_bytes, headers={"Content-Type": "application/x-www-form-urlencoded"})

        label = "Predicting..."
        if response.status_code == 200:
            result = response.json()
            if result['predictions']:
                label = f"{result['predictions'][0]['class']} ({result['predictions'][0]['confidence']*100:.1f}%)"
            else:
                label = "No prediction"
        else:
            label = "API Error"

        # Draw result on frame
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        return image

# Configure STUN server (required for WebRTC)
webrtc_streamer(key="rps-live",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
