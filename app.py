#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

# Get base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model paths
prototxt = os.path.join(BASE_DIR, "models", "models_colorization_deploy_v2.prototxt")
model = os.path.join(BASE_DIR, "models", "colorization_release_v2.caffemodel")
points = os.path.join(BASE_DIR, "models", "pts_in_hull.npy")

# Verify model files exist and provide better error messages
for file_path in [prototxt, model, points]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

# Colorization function
def colorizer(img):
    try:
        # Convert image to grayscale and back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Load the Caffe model
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        pts = np.load(points)

        # Add cluster centers as 1x1 convolutions
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        # Preprocess image
        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Run the network
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # Convert back to RGB
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        return colorized

    except Exception as e:
        st.error(f"Error in colorization: {str(e)}")
        return None

# Streamlit App UI
st.title("🎨 Colorize Your Black and White Image")
st.markdown("Upload a grayscale image and get a colorized version using AI.")


file = st.sidebar.file_uploader("📁 Upload a black & white image", type=["jpg", "jpeg", "png"])

if file is None:
    st.warning("⚠️ Please upload an image file to continue.")
else:
    image = Image.open(file).convert("RGB")
    img = np.array(image)

    st.subheader("🖼️ Original Image")
    st.image(image, use_column_width=True)

    st.subheader("🎨 Colorized Image")
    color = colorizer(img)
    if color is not None:
        st.image(color, use_column_width=True)