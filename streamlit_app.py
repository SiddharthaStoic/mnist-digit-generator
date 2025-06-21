import streamlit as st
import matplotlib.pyplot as plt
from utils import generate_digit_images

st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")
st.title("🖊️ Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    with st.spinner("Generating images..."):
        images = generate_digit_images(digit)

    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(images[i], width=100, caption=f"Sample {i+1}")