import streamlit as st
import requests
from PIL import Image
import io
import base64

# Roboflow details
ROBOFLOW_API_KEY = "hVps0XPzHyrftbIqVfpn"
ROBOFLOW_MODEL = "rock_paper_scissor"  # e.g., rock-paper-scissors
ROBOFLOW_VERSION = 2  # change if needed

st.title("✋ Rock Paper Scissors Classifier - Roboflow Powered")
uploaded_file = st.file_uploader("Upload an image of your hand", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Make API request
    api_url = f"https://infer.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"
    response = requests.post(api_url, data=img_str, headers={"Content-Type": "application/x-www-form-urlencoded"})

    if response.status_code == 200:
        result = response.json()
        prediction = result['predictions'][0]['class']
        confidence = result['predictions'][0]['confidence']
        st.success(f"Prediction: **{prediction.upper()}** with {confidence*100:.2f}% confidence")
    else:
        st.error("Prediction failed. Please check your API key and model name.")
