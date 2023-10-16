import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from PIL import Image

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

st.set_page_config(page_title="ZenAI", page_icon='🌷')

# Function to generate response using GPT-2
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return response_text

# Check if 'users' dictionary exists in session state, if not create one
if 'users' not in st.session_state:
    st.session_state['users'] = {}

if 'user_messages' not in st.session_state:
    st.session_state['user_messages'] = {}

# Check if a user is logged in
def is_user_authenticated():
    return 'current_user' in st.session_state

# Log a user in
def login_user(username):
    st.session_state['current_user'] = username

# Log out the current user
def logout_user():
    if 'current_user' in st.session_state:
        del st.session_state['current_user']

st.markdown("""
<style>
.stButton>button {
  width: 100%;
  height: 60px;  # Adjust button height here
  color: white;
  font-size: 40px;  # Adjust font size here
}
</style>
""", unsafe_allow_html=True)

#navigation Tabs
tab1, tab2, tab3 = st.tabs(["Welcome", "Zen Chatbot", "Chat History"])


# Welcome Page
with tab1:
    st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))
    if is_user_authenticated():
        st.write(f"Welcome back, {st.session_state['current_user']}!")
        if st.button("Logout"):
            logout_user()
            st.write("You've been logged out.")
    else:
        # Login form
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in st.session_state['users'] and st.session_state['users'][username] == password:
                login_user(username)
                st.write(f"Successfully logged in as {username}!")
            else:
                st.warning("Incorrect username or password.")
        
        # Registration form
        st.subheader("Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        
        if st.button("Register"):
            if new_username in st.session_state['users']:
                st.warning("Username already taken.")
            else:
                st.session_state['users'][new_username] = new_password
                st.success(f"User {new_username} has been successfully registered!")

# Chatbot Page
with tab2:
    if is_user_authenticated():
        st.markdown(f'''
        # Conversation with Zen 🌷
        ---''')
        
        input_container = st.container()
        response_container = st.container()
        
        current_user = st.session_state['current_user']
        
        # Ensure the messages for the logged-in user are initialized
        if current_user not in st.session_state['user_messages']:
            st.session_state['user_messages'][current_user] = []
        
        with input_container:
            user_input = st.text_input("💬 Chat Input: ", "", key="input")
            submit_button = st.button("Submit")
            st.divider()

        with response_container:
            if submit_button and user_input:
                response = generate_response(user_input)
                st.session_state['user_messages'][current_user].append({'User': user_input, 'Zen': response})

            if current_user in st.session_state['user_messages']:
                for msg in st.session_state['user_messages'][current_user]:
                    st.write(f"**You**: {msg['User']}")
                    st.write(f"**Zen**: {msg['Zen']}")
        
        # Sidebar contents
        with st.sidebar:
            st.title('💾 Conversation Export Options')
            df_history = pd.DataFrame(st.session_state['user_messages'][current_user])
            csv = df_history.to_csv(index=False)
            if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
                st.balloons()

            json_str = df_history.to_json(orient='records')
            if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
                st.balloons()

    else:
        st.warning("Please login to interact with Zen 🌷.")
