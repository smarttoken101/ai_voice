import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import av
import queue
import threading
import time
from pathlib import Path
import json

# Configure API key
GOOGLE_API_KEY = "AIzaSyAHDkUV3JJyuVRSPwyXjlcQ97QbSUmyOqM"
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the models
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_listening = False
        
    def start_listening(self):
        self.is_listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
    def stop_listening(self):
        self.is_listening = False
        
    def _listen_loop(self):
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                st.error(f"Error in audio capture: {str(e)}")
                continue

class VideoProcessor:
    def __init__(self):
        self.vision_model = vision_model
        self.frame_skip = 30
        self.frame_count = 0
        self.last_response = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip == 0:
            if not self.is_processing:
                self.is_processing = True
                threading.Thread(target=self._process_frame, args=(img.copy(),), daemon=True).start()
        
        if self.last_response:
            img = self._annotate_frame(img, self.last_response)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _process_frame(self, img):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            response = vision_model.generate_content([
                "Describe what you see in this image briefly and naturally.",
                pil_image
            ])
            self.last_response = response.text
        except Exception as e:
            st.error(f"Error in vision processing: {str(e)}")
        finally:
            self.is_processing = False
    
    def _annotate_frame(self, img, text):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, text[:100], (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

def process_audio_to_text(audio_data):
    try:
        text = sr.Recognizer().recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return None

def get_ai_response(input_text, image=None):
    try:
        if image:
            response = vision_model.generate_content([input_text, image])
        else:
            response = text_model.generate_content(input_text)
        return response.text
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

# Streamlit UI
st.title("🤖 Advanced AI Assistant")
st.write("Interactive AI powered by Google Gemini 2.0")

# Chat interface
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input section
input_container = st.container()
with input_container:
    # Text input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_ai_response(prompt)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Auto-play response
            audio_file = text_to_speech(response)
            if audio_file:
                st.audio(audio_file, format='audio/mp3')

    # Voice input controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Start Voice Chat", key="start_voice"):
            st.session_state.audio_processor.start_listening()
            st.info("Listening... Speak now!")
    with col2:
        if st.button("⏹️ Stop Voice Chat", key="stop_voice"):
            st.session_state.audio_processor.stop_listening()
            st.info("Stopped listening.")

    # Process any audio in the queue
    if not st.session_state.audio_processor.audio_queue.empty():
        audio_data = st.session_state.audio_processor.audio_queue.get()
        text = process_audio_to_text(audio_data)
        if text:
            st.session_state.messages.append({"role": "user", "content": f"🎤 {text}"})
            response = get_ai_response(text)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                audio_file = text_to_speech(response)
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')

# Camera section
st.subheader("📸 Camera Interaction")
webrtc_ctx = webrtc_streamer(
    key="gemini-camera",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
