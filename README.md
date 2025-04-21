# 🎨 ColorWeave — Breathe Life into Black & White

> Image Colorization using Deep Learning + OpenCV + Streamlit  
> By **Sahil Shiravale**

---

## 🚀 Live Demo
![Demo](https://j.gifs.com/2xVL2P.gif)

---

## 🧠 Overview

**ColorWeave** is a deep learning project that brings old, grayscale photos to life by colorizing them using convolutional neural networks (CNNs). It builds on the groundbreaking work from UC Berkeley’s paper *“Colorful Image Colorization”* and delivers an intuitive UI using **Streamlit**.

From vintage memories to historical moments — watch the magic of AI add color to them.

---

## 🔥 Why I Built This

This project started with my curiosity about how images are just tensors and how AI can learn to color them meaningfully. Inspired by research from UC Berkeley and a touching moment seeing my grandmother’s saree colorized in an old photo — the smile on my mother’s face made it all worth it.

Here’s one powerful example:

<img src="https://github.com/dhananjayan-r/Colorizer/blob/master/Input_images/che-guevara-wallpapers-hd-best-hd-photos-1080p-6xcp2u-741x988.jpg" width=300><img src="https://github.com/dhananjayan-r/Colorizer/blob/master/Result_images/colored_c1.jpg" width=300>

---

## 🔧 How It Works

This model uses:
- CNN trained on the **ImageNet** dataset.
- Images are converted from **RGB** to **Lab** color space.
- **L channel** (lightness) goes into the model, and it predicts the **a & b channels** (color).
- The predicted Lab image is converted back to RGB for the final color output.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71431013/99061015-eb844a80-25c6-11eb-8850-bcc9f74d91e6.png" width="500">
</p>

---

## 🛠 Installation & Running Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/SahilShiravale/PixelSage.git
cd PixelSage

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the app
streamlit run app.py
