import streamlit as st
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch
import base64
import os

# Load the model
@st.cache_resource
def load_model():
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_model()

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.title("InstructPix2Pix Image Editor")
    run_app()

def run_app():
    st.subheader("Upload an Image and Provide an Instruction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    instruction = st.text_input("Enter an instruction for modification", "Make it look like a painting").strip()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Generate Image"):
            if not instruction:
                st.error("Please enter an instruction for modification.")
                return
            
            with st.spinner("Processing..."):
                processed_image = model(prompt=instruction, image=image).images[0]  # ✅ Pass as a named parameter
                st.image(processed_image, caption="Modified Image", use_column_width=True)
                
                # Save and provide download link
                output_path = "output.png"
                processed_image.save(output_path)
                st.markdown(f'<a href="data:file/png;base64,{encode_image(output_path)}" download="modified_image.png">Download Image</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()