import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
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

# Option_Menu Navigation Bar
nav_bar = option_menu(
    menu_title=None,
    options=['Welcome','Zen Chatbot','Chat History',],
    orientation='horizontal',
)

# Welcome Page
if nav_bar=='Welcome':
    st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))

# Chatbot Page
if nav_bar=='Zen Chatbot':
    st.markdown('''
    # Conversation with Zen 🌷
    ---''')
    input_container = st.container()
    response_container = st.container()

    # Ensure the session state lists are initialized
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    with input_container:
        user_input = st.text_input("💬 Chat Input: ", "", key="input")
        submit_button = st.button("Submit")
        st.divider()

    with response_container:
        if submit_button and user_input:
            response = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)
            st.session_state['history'].append({'User': user_input, 'Zen': response})

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

    # Sidebar contents
    with st.sidebar:
        st.title('💾 Conversation Export Options')
        df_history = pd.DataFrame(st.session_state['history'])
        csv = df_history.to_csv(index=False)
        if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
            st.balloons()

        json_str = df_history.to_json(orient='records')
        if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
            st.balloons()
