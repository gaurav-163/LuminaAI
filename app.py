import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Load pre-trained models (pix2pix and SRGAN)
def load_pix2pix_model():
    # Load your pre-trained pix2pix model here
    # Example: model = torch.load('path_to_pix2pix_model.pth')
    # model.eval()
    return None  # Placeholder

def load_srgan_model():
    # Load your pre-trained SRGAN model here
    # Example: model = torch.load('path_to_srgan_model.pth')
    # model.eval()
    return None  # Placeholder

# Preprocess and upscale image
def upscale_image(model, image, model_type):
    # Preprocess the image
    if model_type == "pix2pix":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the input size expected by pix2pix
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for pix2pix
        ])
    elif model_type == "srgan":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the input size expected by SRGAN
            transforms.ToTensor(),
        ])

    image_tensor = transform(image).unsqueeze(0)

    # Perform upscaling
    with torch.no_grad():
        upscaled_image = model(image_tensor)

    # Post-process the output
    if model_type == "pix2pix":
        upscaled_image = (upscaled_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0  # Denormalize
    elif model_type == "srgan":
        upscaled_image = upscaled_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    upscaled_image = (upscaled_image * 255).astype(np.uint8)
    upscaled_image = Image.fromarray(upscaled_image)

    return upscaled_image

def main():
    st.title("Image Upscaler GAN")

    # Load models
    pix2pix_model = load_pix2pix_model()
    srgan_model = load_srgan_model()

    # Model selection
    model_type = st.radio("Select Model", ("pix2pix", "SRGAN"))

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)

        # Upscale the image
        if st.button('Upscale'):
            if model_type == "pix2pix":
                upscaled_image = upscale_image(pix2pix_model, image, model_type)
            elif model_type == "SRGAN":
                upscaled_image = upscale_image(srgan_model, image, model_type)

            st.image(upscaled_image, caption=f'Upscaled Image ({model_type})', use_column_width=True)

if __name__ == "__main__":
    main()