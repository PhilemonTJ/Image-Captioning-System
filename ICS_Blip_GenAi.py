import os
import requests
import numpy as np
import streamlit as st
from PIL import Image
import google.generativeai as gemini
from gtts import gTTS
import io
import pyttsx3
import torch

# Set up Google GenAI API
gemini.configure(api_key="AIzaSyA7aqzWzhFHLw8dMp7JrapSCT3rcbsG2rc")

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

# Load BLIP model from Hugging Face
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    
    # Enhance caption using Gemini API
    prompt = f"Refine this image caption for better clarity and context: {caption}"
    response = gemini.GenerativeModel('gemini-1.5-pro').generate_content(prompt)
    refined_caption = response.text.strip()
    
    return refined_caption

def text_to_speech(text):
    # Convert the caption to audio
    tts = gTTS(text)
    audio_path = io.BytesIO()
    tts.write_to_fp(audio_path)
    audio_path.seek(0)
    st.audio(audio_path, format="audio/mp3")
    # engine = pyttsx3.init()
    # engine.say(text)
    # engine.runAndWait()

def main():
    st.title("Image Captioning System")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        caption = generate_caption(uploaded_file)
        st.write("Caption:", caption)
        text_to_speech(caption)

if __name__ == "__main__":
    main()