# 🎨 Neural Style Transfer Web App

A minimalist web app built with **Streamlit** and **PyTorch** that applies neural style transfer to blend a content image with an artistic style image — producing AI-generated artwork.
---

## 🖌️ Features

- Black-and-white themed **minimal UI**
- Upload a **content** and a **style** image
- Generates stylized output using **VGG-19** and **deep learning**
- Displays real-time progress and estimated time
- Downloadable result image

---

## 🚀 Live Demo

> *[App Link](https://neuropalette.streamlit.app/)*

---

## 🧠 How It Works

This app uses a **pre-trained VGG-19** model to compute and minimize:

- **Content loss** between your uploaded content image and output
- **Style loss** between your style image and output

Result: your content image is repainted with the patterns of the style image.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/neural-style-transfer-app.git
cd neural-style-transfer-app

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
