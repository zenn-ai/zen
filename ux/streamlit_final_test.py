# Modules
import pyrebase
import streamlit as st
from collections import defaultdict
from datetime import datetime
import time
import random

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

image_url = "https://i.ibb.co/L13FCzp/Zen-AI-logos.jpg"
st.sidebar.image(image_url, use_column_width=True)

# Authentication UI
choice = st.sidebar.selectbox('Login/Signup', ['Login', 'Sign up'])
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

# Function to send message to Firebase
def send_message(user_id, message, sender):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    message_data = {'message': message, 'timestamp': dt_string, 'sender': sender}
    db.child(user_id).child("Messages").push(message_data)

# Fetch Conversation History from Firebase and sort in descending order of timestamp
def get_chat_history(user_id):
    messages = db.child(user_id).child("Messages").get()
    chat_history = [{'message': message.val()['message'], 'timestamp': message.val()['timestamp'], 'sender': message.val()['sender']} for message in messages.each()] if messages.val() else []
    sorted_chat_history = sorted(chat_history, key=lambda x: datetime.strptime(x['timestamp'], "%d/%m/%Y %H:%M:%S"), reverse=True)
    return sorted_chat_history

# Handle chat input and display using st.chat_message
def handle_chat_input_with_st_chat_message(user_id):
    if "messages" not in st.session_state:
        firebase_messages = get_chat_history(user_id)
        st.session_state.messages = [{"role": message['sender'], "content": message['message']} for message in firebase_messages]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("How are you feeling today? Let's chat")
    if user_input:
        send_message(user_id, user_input, 'user')
        st.session_state.messages.append({"role": "user", "content": user_input})

        assistant_response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        send_message(user_id, assistant_response, 'assistant')
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    for message in st.session_state.messages[-10:]: 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ChatBot Logic
if choice == 'Sign up':
    handle = st.sidebar.text_input('Please input your app handle name', value='Default')
    submit = st.sidebar.button('Create my account')
    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created successfully! \n\n Please login to continue.')
        st.balloons()
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])

if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)

        if 'user_id' not in st.session_state or st.session_state.user_id != user['localId']:
            st.session_state.messages = []
            st.session_state.user_id = user['localId']
            
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        tab = st.radio('Go to', ['Chat', 'Conversation History'])

        if tab == 'Chat':
            st.title('Chat with Zen')
            handle_chat_input_with_st_chat_message(user['localId'])

        elif tab == 'Conversation History':
            st.title('Your Conversation History')
            chat_history = get_chat_history(user['localId'])

            date_format = "%d/%m/%Y"
            unique_dates = set(datetime.strptime(message['timestamp'].split(" ")[0], date_format) for message in chat_history)
            sorted_dates = sorted(unique_dates, reverse=True)

            for date in sorted_dates:
                display_date = date.strftime(date_format)
                st.subheader(display_date)

                for message in chat_history:
                    message_date = message['timestamp'].split(" ")[0]
                    if datetime.strptime(message_date, date_format) == date:
                        with st.chat_message(message['sender']):
                            st.markdown(f"{message['message']} ({message['timestamp'].split(' ')[1]})")
                            