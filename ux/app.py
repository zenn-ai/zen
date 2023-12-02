import pyrebase # To install: pip install Pyerbase4
import streamlit as st
from datetime import datetime
import random
import requests
import os


# Load the .env file with the Firebase authentication tokens 
from dotenv import load_dotenv
load_dotenv()

# Configuration Key
firebaseConfig = {
    'apiKey': os.getenv('API_KEY'),
    'authDomain': os.getenv('AUTH_DOMAIN'),
    'projectId': os.getenv('PROJECT_ID'),
    'databaseURL': os.getenv('DATABASE_URL'),
    'storageBucket': os.getenv('STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('MESSAGING_SENDER_ID'),
    'appId': os.getenv('APP_ID'),
}


### Firebase Authentication & Database
firebase = pyrebase.initialize_app(firebaseConfig) # intialize connection using the configuration key
auth = firebase.auth() # accessing the firebase authentication module 
db = firebase.database() # accessing the firebase database module


### Main UI
st.set_page_config(page_title="ZenAI")

# Adjust the positioning of the Radio Buttons 
st.markdown("""
<style>
div.row-widget.stRadio > div {
    display: flex;
    flex-direction: row;
}

.st-emotion-cache-1c7y2kd {
    flex-direction: row-reverse;
    text-align: right;
    }
</style>
""", unsafe_allow_html=True)

tab = st.radio("", ['Login', 'Sign up', 'Chat', 'Conversation History']) # creating radio buttons for navigation


def clear_conversation_history(user_id):
    '''' Function to flush the DB for a specific user '''
    # Delete the user's messages from Firebase DB (user-specific)
    db.child(user_id).child("Messages").remove()


def get_chat_history(user_id):
    ''' Fetch Conversation History from Firebase and sort in descending order of timestamp '''
    messages = db.child(user_id).child("Messages").get()
    chat_history = [{'message': message.val()['message'], 'timestamp': message.val()['timestamp'], 'sender': message.val()['sender']} for message in messages.each()] if messages.val() else []
    sorted_chat_history = sorted(chat_history, key=lambda x: datetime.strptime(x['timestamp'], "%d/%m/%Y %H:%M:%S.%f"), reverse=False)
    return sorted_chat_history


# def stream_output(output_stream):
#     pre = 0
#     for outputs in output_stream:
#         output_text = outputs["text"]
#         output_text = output_text.strip().split(" ")
#         now = len(output_text) - 1
#         if now > pre:
#             st.markdown(" ".join(output_text[pre:now]), end=" ")
#             pre = now
#     st.markdown(" ".join(output_text[pre:]))
#     return " ".join(output_text)


def stream_output(output_stream):
    pre = 0
    t = st.empty()
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            t.write("%s" % " ".join(output_text[pre:now]))
            pre = now
    
    reply = " ".join(output_text[pre:]).strip()
    t.write("%s" % reply)


# Handle chat input and display using st.chat_message
def handle_chat_input_with_st_chat_message(user_id):
    '''Function to handle chat using the streamlit chat_message module'''
    if "messages" not in st.session_state:
        firebase_messages = get_chat_history(user_id)
        st.session_state.messages = [{"role": message['sender'], "content": message['message']} for message in firebase_messages]

    # Display the last 10 messages
    for message in st.session_state.messages[-10:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("How are you feeling today? Let's chat")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        api_url = "https://922b-35-226-40-201.ngrok-free.app/api/data"
        
        done = False
        with st.chat_message("assistant"):
            assistant_response = requests.post(api_url, json={"question": user_input, "user_id": user_id}, stream=True)
            if assistant_response.status_code == 200:
                try:
                    placeholder = st.empty()
                    full_response = []
                    for output_text in assistant_response:
                        output_text = output_text.decode('utf-8').strip()
                        full_response.append(output_text)
                        placeholder.markdown(" ".join(full_response))
                    placeholder.markdown(" ".join(full_response))
                finally:
                    assistant_response.close()

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": " ".join(full_response)})


# Text input styling
def text_input_with_styling(label, value='', key=None, placeholder_color='white', text_color='white'):
    '''Custom CSS to include white color for text input and placeholder'''
    st.markdown(f"""
    <style>
    input[type="text"]::placeholder {{
        color: {placeholder_color};
    }}
    input[type="text"], input[type="password"] {{
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)
    return st.text_input(label, value=value, key=key)


### Signup Logic
if tab == 'Sign up':
    email = text_input_with_styling('Please enter your email address', key='email_input')
    password = st.text_input('Please enter your password', type='password', key='login_password')
    handle = text_input_with_styling('Please input your app handle name', value='Default', key='handle_input')

    submit = st.button('Create my account')
    if submit:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success('Your account is created successfully! \n\n Please login to continue.')
            st.balloons()
            # Sign in the user to set their handle and ID
            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("ID").set(user['localId'])
            st.session_state.user_id = user['localId']
            st.session_state.messages = []  
        except Exception as e:
            if "EMAIL_EXISTS" in str(e):
                st.error('This email is already in use! Try a different address.')
            else:
                st.error('Wrong Credentials format.') 


### Login Logic
elif tab == 'Login':
    email = text_input_with_styling('Please enter your email address', key='email_input')
    password = st.text_input('Please enter your password', type='password', key='login_password')
    login = st.button('Login')
    if login:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user_id = user['localId']
             # Fetch and store the user handle
            user_info = db.child(user['localId']).child("Handle").get()
            if user_info.val():
                st.session_state.user_handle = user_info.val()
            st.success('Logged in successfully!')
            st.session_state.messages = []
        except Exception:
            st.error('Login failed: Wrong Credentials!' )


# Image in the sidebar 
image_url = "https://i.ibb.co/1qYfmzm/ZenAI.jpg"
st.sidebar.image(image_url, use_column_width=True)
# Sidebar User Info Display
if 'user_id' in st.session_state and 'user_handle' in st.session_state:
    st.sidebar.markdown(f"<h2 style='color:white;'>Welcome back, {st.session_state.user_handle}!</h2>", unsafe_allow_html=True)


### Chat Logic
if tab == 'Chat' and 'user_id' in st.session_state:
    st.title('Chat with Zen')
    handle_chat_input_with_st_chat_message(st.session_state.user_id)

# Chat History Logic
elif tab == 'Conversation History' and 'user_id' in st.session_state:
    st.title('Your Conversation History')
    chat_history = get_chat_history(st.session_state.user_id)

    # Assuming the date format in the timestamp is 'DD/MM/YYYY'
    date_format = "%d/%m/%Y"
    unique_dates = set(datetime.strptime(message['timestamp'].split(" ")[0], date_format) for message in chat_history)
    sorted_dates = sorted(unique_dates, reverse=True)

    for date in sorted_dates:
        display_date = date.strftime(date_format)
        
        with st.expander(display_date, expanded=False): # embed the chat hitsory into partitioned expanding blocks
            for message in chat_history:
                message_date = message['timestamp'].split(" ")[0]
                if datetime.strptime(message_date, date_format) == date:
                    with st.chat_message(message['sender']):
                        st.markdown(f"{message['message']} ({message['timestamp'].split(' ')[1].split('.')[0]})")

    # Clear Conversation History Button
    if st.button('Clear Conversation History'):
        clear_conversation_history(st.session_state.user_id)
        st.success('Conversation history cleared.')
        st.session_state.messages = []  # Clear local chat history
        st.experimental_rerun() # refresh the current page
        


# Make sure to clear the chat input field after sending a message
if 'clear_chat' in st.session_state and st.session_state.clear_chat:
    st.session_state.clear_chat = False
