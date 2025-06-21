import streamlit as st
import torch
import numpy as np
from model import CVAE  # Import the model class from model.py

# --- Configuration ---
LATENT_DIM = 20
NUM_CLASSES = 10
IMG_SHAPE = (1, 28, 28)
MODEL_PATH = 'cvae_mnist.pth' # Make sure this path is correct

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained CVAE model."""
    model = CVAE(img_shape=IMG_SHAPE, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM)
    # Load the state dict. Use map_location for CPU deployment.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

model = load_model()

# --- Web App UI ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# User input
st.subheader("Choose a digit to generate (0-9):")
digit_to_generate = st.selectbox(
    label="Digit",
    options=list(range(10)),
    label_visibility="collapsed"
)

generate_button = st.button("Generate Images")

# --- Image Generation and Display ---
if generate_button:
    st.subheader(f"Generated images of digit {digit_to_generate}")
    
    # Create 5 columns to display the images
    cols = st.columns(5)
    
    for i in range(5):
        with torch.no_grad():
            # 1. Sample a random latent vector from a standard normal distribution
            z = torch.randn(1, LATENT_DIM)
            
            # 2. Generate an image using the decoder part of the model
            generated_image_tensor = model.decode(z, digit_to_generate)
            
            # 3. Post-process the tensor to be a displayable image
            # Remove channel dimension, convert to numpy, and scale to 0-255
            image = generated_image_tensor.squeeze().cpu().numpy()
            image = (image * 255).astype(np.uint8)

            # 4. Display the image in a column
            with cols[i]:
                st.image(image, caption=f"Sample {i+1}", width=128)