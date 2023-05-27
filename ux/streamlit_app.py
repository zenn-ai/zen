import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from hugchat import hugchat
import json
import pandas as pd
from PIL import Image


st.set_page_config(page_title="ZenAI", page_icon='🌷')

# Generate empty lists for generated, past, and history.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi! I'm Zen, how are you today?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Intializing Zen...']
## history stores the whole conversation
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.markdown("""
<style>
.stButton>button {
  width: 100%;
  height: 60px;  # Adjust button height here
  color: white;
  font-size: 40px;  # Adjust font size here
}
""", unsafe_allow_html=True)

# Option_Menu Navigation Bar
nav_bar = option_menu(
    menu_title=None,
    options=['Welcome','Zen Chatbot','Data','Logs'],
    orientation='horizontal',
)

# Welcome Page
if nav_bar=='Welcome':
    st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))

# Chatbot Page
if nav_bar=='Zen Chatbot':
    # Layout of input/response containers
    st.markdown('''
    # Conversation with Zen 🌷
    ---''')
    input_container = st.container()
    response_container = st.container()

    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("💬 Chat Input: ", "", key="input")
        return input_text

    ## Applying the user input box
    with input_container:
        user_input = get_text()
        submit_button = st.button("Submit")
        st.divider()

    # Load the cookies file
    with open('cookies.json', 'r') as file:
        cookie_data = json.load(file)

    # Convert to simple dictionary format
    cookies = {cookie['name']: cookie['value'] for cookie in cookie_data}

    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    def generate_response(prompt):
        chatbot = hugchat.ChatBot(cookies)
        response = chatbot.chat(prompt)
        # Add the user input and AI response to the conversation history
        st.session_state['history'].append({'User': user_input, 'Zen': response})
        return response

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if submit_button and user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))


    # Sidebar contents
    with st.sidebar:
        st.title('💾 Conversation Export Options')
        # Add a button to download the conversation history as a CSV file
        df_history = pd.DataFrame(st.session_state['history'])
        csv = df_history.to_csv(index=False)
        if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
            st.balloons()

        # Add a button to download the conversation history as a JSON file
        json_str = df_history.to_json(orient='records')
        if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
            st.balloons()
