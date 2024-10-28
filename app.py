import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Set the title and instructions
st.title("Neural Style Transfer Web App")
st.write("Upload a content image and a style image, and the app will apply the style of one image onto the other!")

# Load content and style images
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])


# Function to preprocess and convert uploaded image to a TensorFlow-compatible tensor
def load_and_process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels (RGB)
    image = image.resize((256, 256))  # Resize for model compatibility if needed
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return tf.convert_to_tensor(image, dtype=tf.float32)[tf.newaxis, ...]


# Display uploaded images
if content_image_file and style_image_file:
    content_image = load_and_process_image(content_image_file)
    style_image = load_and_process_image(style_image_file)

    st.image(content_image_file, caption="Content Image", use_column_width=True)
    st.image(style_image_file, caption="Style Image", use_column_width=True)

    # Run style transfer
    if st.button("Generate Styled Image"):
        with st.spinner("Stylizing..."):
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            stylized_image = tf.image.convert_image_dtype(stylized_image, dtype=tf.uint8)
            stylized_image = Image.fromarray(stylized_image.numpy().squeeze())

            st.image(stylized_image, caption="Styled Image", use_column_width=True)
else:
    st.write("Please upload both content and style images to proceed.")
