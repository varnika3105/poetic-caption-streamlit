import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
from gtts import gTTS
import base64
import os
import io

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

# 🌗 Theme toggle
theme_choice = st.sidebar.radio("🌓 Choose Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #111; color: #EEE; }
        </style>
    """, unsafe_allow_html=True)

# 🔠 UI Title
st.markdown("""
    <h1 style="text-align:center;">📸 Poetic Caption Generator 🎨</h1>
    <p style="text-align:center;">Upload a photo, pick a poetic style, and let AI turn vision into verse.</p>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    style = st.selectbox("Choose a poetic style", list(style_prompts.keys()))
    generate = st.button("✨ Generate")

with col2:
    if uploaded_image and generate:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Generating poetic caption..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = processor(image, return_tensors="pt").to(device)
            output = blip_model.generate(**inputs, max_new_tokens=40)
            base_caption = processor.decode(output[0], skip_special_tokens=True)

            prompt = style_prompts[style].format(base_caption)
            poetic = poetry_generator(prompt, max_new_tokens=50)[0]['generated_text'].strip()

        st.success("🎉 Captions Generated!")
        st.subheader("📝 Descriptive Caption")
        st.write(base_caption)
        st.subheader("🎨 Poetic Caption")
        st.write(poetic)

        # 🔊 Voice narration
        tts = gTTS(text=poetic, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

        st.audio(audio_bytes, format='audio/mp3')
