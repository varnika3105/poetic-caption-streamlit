# app.py

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch

# Load models
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to("cuda" if torch.cuda.is_available() else "cpu")
    poetry_model = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M")
    return processor, blip_model, poetry_model

processor, blip_model, poetry_generator = load_models()

# Prompt templates
style_prompts = {
    "Romantic": "Turn this caption into a romantic poetic verse:\n\nCaption: {}\nPoem:",
    "Haiku": "Rewrite this as a haiku poem:\n\nCaption: {}\nHaiku:",
    "Funny": "Make this caption sound funny and poetic:\n\nCaption: {}\nPoem:",
    "Shakespearean": "Rewrite this like a Shakespearean couplet:\n\nCaption: {}\nPoem:",
    "Melancholy": "Turn this into a poetic and sad two-line verse:\n\nCaption: {}\nPoem:"
}

# UI layout
st.title("📸 Poetic Caption Generator")
st.markdown("Upload an image and pick a poetic style to get a creative AI-generated caption.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
style = st.selectbox("Choose a poetic style", list(style_prompts.keys()))

if uploaded_image and st.button("✨ Generate"):
    image = Image.open(uploaded_image).convert("RGB")

    with st.spinner("Generating descriptive caption..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_new_tokens=40)
        base_caption = processor.decode(output[0], skip_special_tokens=True)

    with st.spinner("Generating poetic caption..."):
        prompt = style_prompts[style].format(base_caption)
        poetic = poetry_generator(prompt, max_new_tokens=50)[0]['generated_text'].strip()

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.subheader("📝 Descriptive Caption")
    st.write(base_caption)
    st.subheader("🎨 Poetic Caption")
    st.write(poetic)
