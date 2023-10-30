# Modules
import pyrebase
import streamlit as st
from datetime import datetime

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
st.sidebar.title("Our community app")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

# Simple Chatbot Function
def chatbot_response(message):
    responses = {
        "hi": "Hello! How can I assist you today?",
        "how are you?": "I'm just a computer program, but I'm doing well, thank you!",
        "bye": "Goodbye! Have a great day!",
    }
    return responses.get(message.lower(), "I'm not sure how to respond to that.")

#Function to display a chat message
def display_chat_message(user, message, is_user=True):
    if is_user:
        st.markdown(f"<div style='text-align: right; margin-right: 10px; background-color: #e1f5fe; padding: 10px; border-radius: 10px; display: inline-block;'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; margin-left: 10px; background-color: #ffffff; padding: 10px; border-radius: 10px; display: inline-block;'>{message}</div>", unsafe_allow_html=True)

# Function to send message
def send_message(user_id, message, sender):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    message_data = {'message': message, 'timestamp': dt_string, 'sender': sender}
    db.child(user_id).child("Messages").push(message_data)

# Function to get chat history
def get_chat_history(user_id):
    messages = db.child(user_id).child("Messages").get()
    if messages.val() is not None:
        return [{'message': message.val()['message'], 'timestamp': message.val()['timestamp'], 'sender': message.val()['sender']} for message in messages.each()]
    return []

# Function to handle chat input
def handle_chat_input(user_id):
    user_message = st.text_input("You:")
    if st.button("Send"):
        if user_message:
            display_chat_message("You", user_message, is_user=True)
            send_message(user_id, user_message, 'user')
            bot_response = chatbot_response(user_message)
            send_message(user_id, bot_response, 'bot')
            display_chat_message("Bot", bot_response, is_user=False)

# App
if choice == 'Sign up':
    handle = st.sidebar.text_input('Please input your app handle name', value='Default')
    submit = st.sidebar.button('Create my account')
    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created successfully!')
        st.balloons()
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])

if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        tab = st.radio('Go to', ['Chatbot', 'Chat History'])

        if tab == 'Chatbot':
            st.title('Chat with our Bot')
            handle_chat_input(user['localId'])
        
        elif tab == 'Chat History':
            st.title('Your Chat History')
            chat_history = get_chat_history(user['localId'])
            for message in reversed(chat_history):
                display_chat_message("Bot" if message['sender'] == 'bot' else "You", message['message'], is_user=(message['sender']=='user'))