import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import time
from pathlib import Path

# Configure API key
GOOGLE_API_KEY = "AIzaSyAHDkUV3JJyuVRSPwyXjlcQ97QbSUmyOqM"
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the models
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"

def get_gemini_response(input_text, image=None):
    if image:
        response = vision_model.generate_content([input_text, image])
    else:
        response = text_model.generate_content(input_text)
    return response.text

# Streamlit UI
st.title("🤖 Gemini AI Assistant")
st.write("Chat with AI using text, voice, or camera!")

# Create tabs for different modes
tab1, tab2, tab3 = st.tabs(["💭 Text Chat", "🎤 Voice Chat", "📸 Camera Chat"])

with tab1:
    st.subheader("Text Chat")
    user_input = st.text_area("Your message:", key="text_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send", key="send_text"):
            if user_input:
                response = get_gemini_response(user_input)
                st.write("AI Response:")
                st.write(response)
                # Convert response to speech
                audio_file = text_to_speech(response)
                st.audio(audio_file)

with tab2:
    st.subheader("Voice Chat")
    if st.button("🎤 Start Recording", key="record"):
        user_input = speech_to_text()
        if user_input:
            st.write("You said:", user_input)
            response = get_gemini_response(user_input)
            st.write("AI Response:")
            st.write(response)
            # Convert response to speech
            audio_file = text_to_speech(response)
            st.audio(audio_file)

with tab3:
    st.subheader("Camera Chat")
    webrtc_ctx = webrtc_streamer(
        key="camera",
        video_frame_callback=None,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
    if st.button("Capture Image"):
        if webrtc_ctx.video_receiver:
            image = webrtc_ctx.video_receiver.get_frame()
            if image is not None:
                st.image(image, caption="Captured Image", use_column_width=True)
                
                image_prompt = st.text_area("Ask something about the image:", key="live_image_prompt")
                if st.button("Analyze", key="analyze_live_image"):
                    if image_prompt:
                        response = get_gemini_response(image_prompt, image)
                        st.write("AI Response:")
                        st.write(response)
                        # Convert response to speech
                        audio_file = text_to_speech(response)
                        st.audio(audio_file)
