import streamlit as st
import pyrebase
from datetime import datetime
from PIL import Image
import random
import time


# Configuration Key
firebaseConfig = {
    'apiKey': "AIzaSyBvAeBh-ghFe-4n9VSNTSW_h9zCT3bXngg",
    'authDomain': "cloud-lab-ff59.firebaseapp.com",
    'projectId': "cloud-lab-ff59",
    'databaseURL': "https://cloud-lab-ff59-default-rtdb.firebaseio.com/",
    'storageBucket': "cloud-lab-ff59.appspot.com",
    'messagingSenderId': "1016159354336",
    'appId': "1:1016159354336:web:862ea0d538eee01c11ff85",
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()
with st.sidebar:
    st.image(Image.open('ZenAI-logos/ZenAI-logos_white.png'))

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])
# email = st.sidebar.text_input('Please enter your email address')
# password = st.sidebar.text_input('Please enter your password', type='password')
email = "abcde@gmail.com"
password = "abcde123"

st.title("ZenAI: A Mental Health Chatbot")

with st.chat_message(name="ZenAI", avatar="/Users/puneet/Documents/GitHub/therapy-bot/ux/ZenAI-logos/ZenAI.JPG"):
    st.write("Hello :wave:. I am ZenAI, a mental health chatbot.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages: 
    with st.chat_message(message["role"], avatar="/Users/puneet/Documents/GitHub/therapy-bot/ux/ZenAI-logos/ZenAI.JPG"):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("I am here to help you with your mental health. How are you feeling today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="/Users/puneet/Documents/GitHub/therapy-bot/ux/ZenAI-logos/ZenAI.JPG"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
        for chunk in assistant_response.split():
            full_response += chunk + " "
            stream=True
            time.sleep(0.15)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
