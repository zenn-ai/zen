########################################################TEST1########################################################



# import streamlit as st
# from streamlit_chat import message
# from streamlit_option_menu import option_menu
# from hugchat import hugchat
# import json
# import pandas as pd
# from PIL import Image


# st.set_page_config(page_title="ZenAI", page_icon='🌷')

# # Generate empty lists for generated, past, and history.
# ## generated stores AI generated responses
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = ["Hi! I'm Zen, how are you today?"]
# ## past stores User's questions
# if 'past' not in st.session_state:
#     st.session_state['past'] = ['Intializing Zen...']
# ## history stores the whole conversation
# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# st.markdown("""
# <style>
# .stButton>button {
#   width: 100%;
#   height: 60px;  # Adjust button height here
#   color: white;
#   font-size: 40px;  # Adjust font size here
# }
# """, unsafe_allow_html=True)

# # Option_Menu Navigation Bar
# nav_bar = option_menu(
#     menu_title=None,
#     options=['Welcome','Zen Chatbot','Data','Logs'],
#     orientation='horizontal',
# )

# # Welcome Page
# if nav_bar=='Welcome':
#     st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))

# # Chatbot Page
# if nav_bar=='Zen Chatbot':
#     # Layout of input/response containers
#     st.markdown('''
#     # Conversation with Zen 🌷
#     ---''')
#     input_container = st.container()
#     response_container = st.container()

#     # User input
#     ## Function for taking user provided prompt as input
#     def get_text():
#         input_text = st.text_input("💬 Chat Input: ", "", key="input")
#         return input_text

#     ## Applying the user input box
#     with input_container:
#         user_input = get_text()
#         submit_button = st.button("Submit")
#         st.divider()

#     # Load the cookies file
#     with open('cookies.json', 'r') as file:
#         cookie_data = json.load(file)

#     # Convert to simple dictionary format
#     cookies = {cookie['name']: cookie['value'] for cookie in cookie_data}

#     # Response output
#     ## Function for taking user prompt as input followed by producing AI generated responses
#     def generate_response(prompt):
#         chatbot = hugchat.ChatBot(cookies)
#         response = chatbot.chat(prompt)
#         # Add the user input and AI response to the conversation history
#         st.session_state['history'].append({'User': user_input, 'Zen': response})
#         return response

#     ## Conditional display of AI generated responses as a function of user provided prompts
#     with response_container:
#         if submit_button and user_input:
#             response = generate_response(user_input)
#             st.session_state.past.append(user_input)
#             st.session_state.generated.append(response)

#         if st.session_state['generated']:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#                 message(st.session_state["generated"][i], key=str(i))


#     # Sidebar contents
#     with st.sidebar:
#         st.title('💾 Conversation Export Options')
#         # Add a button to download the conversation history as a CSV file
#         df_history = pd.DataFrame(st.session_state['history'])
#         csv = df_history.to_csv(index=False)
#         if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
#             st.balloons()

#         # Add a button to download the conversation history as a JSON file
#         json_str = df_history.to_json(orient='records')
#         if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
#             st.balloons()


########################################################TEST2########################################################


# import streamlit as st
# from streamlit_option_menu import option_menu
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import pandas as pd
# from PIL import Image

# # Load pre-trained model and tokenizer
# model_name = "gpt2-medium"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# st.set_page_config(page_title="ZenAI", page_icon='🌷')

# # Function to generate response using GPT-2
# def generate_response(prompt):
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
#     response_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response_text

# # Check if 'users' dictionary exists in session state, if not create one
# if 'users' not in st.session_state:
#     st.session_state['users'] = {}

# if 'user_messages' not in st.session_state:
#     st.session_state['user_messages'] = {}

# # Check if a user is logged in
# def is_user_authenticated():
#     return 'current_user' in st.session_state

# # Log a user in
# def login_user(username):
#     st.session_state['current_user'] = username

# # Log out the current user
# def logout_user():
#     if 'current_user' in st.session_state:
#         del st.session_state['current_user']

# st.markdown("""
# <style>
# .stButton>button {
#   width: 100%;
#   height: 60px;  # Adjust button height here
#   color: white;
#   font-size: 40px;  # Adjust font size here
# }
# </style>
# """, unsafe_allow_html=True)

# # Option_Menu Navigation Bar
# nav_bar = option_menu(
#     menu_title=None,
#     options=['Welcome','Zen Chatbot','Chat History',],
#     orientation='horizontal',
# )

# # Welcome Page
# if nav_bar=='Welcome':
#     st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))
#     if is_user_authenticated():
#         st.write(f"Welcome back, {st.session_state['current_user']}!")
#         if st.button("Logout"):
#             logout_user()
#             st.write("You've been logged out.")
#     else:
#         # Login form
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
        
#         if st.button("Login"):
#             if username in st.session_state['users'] and st.session_state['users'][username] == password:
#                 login_user(username)
#                 st.write(f"Successfully logged in as {username}!")
#             else:
#                 st.warning("Incorrect username or password.")
        
#         # Registration form
#         st.subheader("Register")
#         new_username = st.text_input("New Username")
#         new_password = st.text_input("New Password", type="password")
        
#         if st.button("Register"):
#             if new_username in st.session_state['users']:
#                 st.warning("Username already taken.")
#             else:
#                 st.session_state['users'][new_username] = new_password
#                 st.success(f"User {new_username} has been successfully registered!")

# # Chatbot Page
# if nav_bar == 'Zen Chatbot':
#     if is_user_authenticated():
#         st.markdown(f'''
#         # Conversation with Zen 🌷
#         ---''')
        
#         input_container = st.container()
#         response_container = st.container()
        
#         current_user = st.session_state['current_user']
        
#         # Ensure the messages for the logged-in user are initialized
#         if current_user not in st.session_state['user_messages']:
#             st.session_state['user_messages'][current_user] = []
        
#         with input_container:
#             user_input = st.text_input("💬 Chat Input: ", "", key="input")
#             submit_button = st.button("Submit")
#             st.divider()

#         with response_container:
#             if submit_button and user_input:
#                 response = generate_response(user_input)
#                 st.session_state['user_messages'][current_user].append({'User': user_input, 'Zen': response})

#             if current_user in st.session_state['user_messages']:
#                 for msg in st.session_state['user_messages'][current_user]:
#                     st.write(f"**You**: {msg['User']}")
#                     st.write(f"**Zen**: {msg['Zen']}")

#         # Sidebar contents
#         with st.sidebar:
#             st.title('💾 Conversation Export Options')
#             df_history = pd.DataFrame(st.session_state['user_messages'][current_user])
#             csv = df_history.to_csv(index=False)
#             if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
#                 st.balloons()

#             json_str = df_history.to_json(orient='records')
#             if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
#                 st.balloons()

#     else:
#         st.warning("Please login to interact with Zen 🌷.")



########################################################TEST3########################################################



# import streamlit as st
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import pandas as pd
# from PIL import Image

# # # Load pre-trained model and tokenizer
# # model_name = "distilgpt2"
# # model = GPT2LMHeadModel.from_pretrained(model_name)
# # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# st.set_page_config(page_title="ZenAI", page_icon='🌷')

# # # Function to generate response using GPT-2
# # def generate_response(prompt):
# #     input_ids = tokenizer.encode(prompt, return_tensors="pt")
# #     output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
# #     response_text = tokenizer.decode(output[0], skip_special_tokens=True)
# #     return response_text

# # Check if 'users' dictionary exists in session state, if not create one
# if 'users' not in st.session_state:
#     st.session_state['users'] = {}

# if 'user_messages' not in st.session_state:
#     st.session_state['user_messages'] = {}

# # Check if a user is logged in
# def is_user_authenticated():
#     return 'current_user' in st.session_state

# # Log a user in
# def login_user(username):
#     st.session_state['current_user'] = username

# # Log out the current user
# def logout_user():
#     if 'current_user' in st.session_state:
#         del st.session_state['current_user']

# st.markdown("""
# <style>
# .stButton>button {
#   width: 100%;
#   height: 60px;  # Adjust button height here
#   color: white;
#   font-size: 40px;  # Adjust font size here
# }
# </style>
# """, unsafe_allow_html=True)

# #navigation Tabs
# tab1, tab2, tab3 = st.tabs(["Welcome", "Zen Chatbot", "Chat History"])


# # Welcome Page
# with tab1:
#     st.image(Image.open('C:/Users/Danil/Documents/GitHub/therapy-bot/ux/welc_img.png'))
#     if is_user_authenticated():
#         st.write(f"Welcome back, {st.session_state['current_user']}!")
#         if st.button("Logout"):
#             logout_user()
#             st.write("You've been logged out.")
#     else:
#         # Login form
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
        
#         if st.button("Login"):
#             if username in st.session_state['users'] and st.session_state['users'][username] == password:
#                 login_user(username)
#                 st.write(f"Successfully logged in as {username}!")
#             else:
#                 st.warning("Incorrect username or password.")
        
#         # Registration form
#         st.subheader("Register")
#         new_username = st.text_input("New Username")
#         new_password = st.text_input("New Password", type="password")
        
#         if st.button("Register"):
#             if new_username in st.session_state['users']:
#                 st.warning("Username already taken.")
#             else:
#                 st.session_state['users'][new_username] = new_password
#                 st.success(f"User {new_username} has been successfully registered!")

# # Chatbot Page
# with tab2:
#     if is_user_authenticated():
#         st.markdown(f'''
#         # Conversation with Zen 🌷
#         ---''')
        
#         input_container = st.container()
#         response_container = st.container()
        
#         current_user = st.session_state['current_user']
        
#         # Ensure the messages for the logged-in user are initialized
#         if current_user not in st.session_state['user_messages']:
#             st.session_state['user_messages'][current_user] = []
        
#         with input_container:
#             user_input = st.text_input("💬 Chat Input: ", "", key="input")
#             submit_button = st.button("Submit")
#             st.divider()

#         with response_container:
#             if submit_button and user_input:
#                 response = generate_response(user_input)
#                 st.session_state['user_messages'][current_user].append({'User': user_input, 'Zen': response})

#             if current_user in st.session_state['user_messages']:
#                 for msg in st.session_state['user_messages'][current_user]:
#                     st.write(f"**You**: {msg['User']}")
#                     st.write(f"**Zen**: {msg['Zen']}")
        
#         # Sidebar contents
#         if tab2:
#             with st.sidebar:
#                 st.title('💾 Conversation Export Options')
#                 df_history = pd.DataFrame(st.session_state['user_messages'][current_user])
#                 csv = df_history.to_csv(index=False)
#                 if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
#                     st.balloons()

#                 json_str = df_history.to_json(orient='records')
#                 if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
#                     st.balloons()
#         else:
#             st.sidebar.empty()

#     else:
#         st.warning("Please login to interact with Zen 🌷.")





#######################################################################################################

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from PIL import Image
# import pyrebase

# # Configuration firebase
# firebaseConfig = {
#   'apiKey': "AIzaSyBvAeBh-ghFe-4n9VSNTSW_h9zCT3bXngg",
#   'authDomain': "cloud-lab-ff59.firebaseapp.com",
#   'projectId': "cloud-lab-ff59",
#   'databaseURL': 
#   'storageBucket': "cloud-lab-ff59.appspot.com",
#   'messagingSenderId': "1016159354336",
#   'appId': "1:1016159354336:web:862ea0d538eee01c11ff85"
# }

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
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
        if "start_chatting" not in st.session_state or not st.session_state["start_chatting"]:
            if st.button("Start Chatting"):
                st.session_state["start_chatting"] = True
        else:
            st.markdown(f'''
            # Conversation with Zen 🌷
            ---''')

            chat_container = st.container()

            with chat_container:
                current_user = st.session_state['current_user']

                # Ensure the messages for the logged-in user are initialized
                if current_user not in st.session_state['user_messages']:
                    st.session_state['user_messages'][current_user] = []

                # Display the chat history
                for msg in st.session_state['user_messages'].get(current_user, []):
                    st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin: 5px 0;'>👤: {msg['User']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background-color: #c8e6c9; padding: 10px; border-radius: 5px; margin: 5px 0;'>🌷: {msg['Zen']}</div>", unsafe_allow_html=True)

                # User input bar positioned beneath the chat history
                user_input = st.text_input("", "", key="input")
                submit_button = st.button("Submit")

                # Generate and append the bot's response
                if submit_button and user_input:
                    response = generate_response(user_input)
                    st.session_state['user_messages'][current_user].append({'User': user_input, 'Zen': response})

            with st.sidebar:
                st.title('💾 Conversation Export Options')
                df_history = pd.DataFrame(st.session_state['user_messages'][current_user])
                csv = df_history.to_csv(index=False)
                if st.download_button("Download Conversation History as CSV", csv, 'conversation_history.csv'):
                    st.balloons()
                json_str = df_history.to_json(orient='records')
                if st.download_button("Download Conversation History as JSON", json_str, 'conversation_history.json'):
                    st.balloons()
                clear_button = st.sidebar.button('Clear Conversation',key='clear')

    else:
        st.warning("Please login to interact with Zen 🌷.")
