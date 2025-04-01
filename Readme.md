# 🎨 Lumina AI

This project uses the **Instruct-Pix2Pix** model from Hugging Face (`timbrooks/instruct-pix2pix`) to colorize grayscale images. The project provides a Gradio interface that allows users to upload grayscale images and generate colorized versions with a simple click.

---

## 🚀 Features
- ✅ Upload grayscale images for automatic colorization.
- ✅ Real-time inference using the **Instruct-Pix2Pix** model.
- ✅ User-friendly web interface with Gradio.

---

## 📦 Project Structure

---

## 🛠️ Getting started With
Before running the project, make sure you have the following packages installed:

```bash
Poetry install
```
step 2:- 
```bash
streamlit run app.py
```

---

## 🛠️ Requirements
Before running the project, make sure you have the following packages installed:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv pix2pix-env
source pix2pix-env/bin/activate  # On Mac/Linux
# or
pix2pix-env\Scripts\activate     # On Windows

# Install required packages
pip install torch torchvision diffusers gradio pillow
```
## 📸 Model Details

1. Model: timbrooks/instruct-pix2pix

2. Task: Image-to-Image Transformation using Natural Language Instructions

3. Optimized for GPU usage (torch.cuda), with fallback to CPU.

## 📝 Usage

Follow these steps to run the project:

1. Clone the repository:
```bash
git clone https://github.com/your-username/pix2pix-colorizer.git
cd pix2pix-colorizer
```

## 💻 How It Works
1. Upload a grayscale image.

2. Click on the Submit button.

3. The colorized image will be displayed.
