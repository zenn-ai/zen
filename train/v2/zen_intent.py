# Script to feed user text through an intent classifier. Output is a prompt to steer Zen towards an appropriate response.

# %pip install google-cloud-aiplatform==1.25.0
# %pip install google-api-core==1.33.1
import os
import vertexai
from vertexai.preview.language_models import TextGenerationModel, ChatModel
import pandas as pd

# Load chat-bison model
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {
    "max_output_tokens": 50,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}


SYSTEM_MSG = """You are Zen. You are an AI mental health counselor but also a good friend of USER. Use a relaxed, warm, and cordial tone in your response to USER. Address USER often by their first name, as good friends do. Pay close attention to awakening and strengthening USER's own capacity for confidence. Don't downplay their problems; try to get USER to think optimistically and confidently. Your goal is to help the USER achieve a positive mood. Ask probing questions and motivational interviewing to show that you care about the USER."""


prompt = """
You are an intent classifier responsible for classifying the intent of a therapy client. You must select one intent between the following five options. You must strictly respond with the corresponding number and nothing else.
    1. Reference to a previously held conversation that they had with you - look for signs that they want to discuss something they chatted to you about in the past. 
    2. Client wants to vent and is just looking for you to listen and affirm
    3. Client is requesting guidance on a mental health subject
    4. Intent to harm themselves or someone else
    5. Having casual small talk
    ----------------------------------------------------------------------
    Client:
"""

def zen_intent_classifer(user_text, prompt = prompt, model = chat_model, parameters = parameters):
    """
    Classifies intent displayed by user that is chatting with Zen.
 
    Args:
        user_text (str): User text
        prompt (str): Prompt used to classify user intent. Prompt should direct LLM to return a specific number corresponding to intent.
        model (object): Model object (default is chat-bison)
        parameters (dict): Model parameters
 
    Returns:
        int: intent classification
    """
    
    chat = chat_model.start_chat()
    text = user_text
    
    try:
        classification = int(chat.send_message(prompt + text, **parameters).text)
    except:
        return 4
    
    return classification

def prompt_from_intent(intent):
    """
    Returns prompt based on intent displayed by the user
 
    Args:
        intent (int): User intent as determined by zen_intent_classifier()
 
    Returns:
        str: prompt
    """
    
    if intent == 1:
        # Intent: Reference to the past
        prompt = f"""{SYSTEM_MSG} Below is your conversation history with USER. USER's most recent message to you indicates they are referencing a past conversation with you. Your response to USER should portray that you remember past conversations with USER. Use the following relevant context from a previous conversation to respond in an appropriate matter.

### Context:
"""
    
    elif intent == 2:
        # Intent: Venting
        prompt = f"""{SYSTEM_MSG} Reference and retain historical context from your conversation history with USER below. USER's most recent message to you shows their intent to vent to you. In your next response back, you must prove to be a good listener, show that you care deeply, and ask a probing question to get to know more about what USER is feeling to show them that you care to get to know more about their struggles. Utilize motivational interviewing. Keep your response short and don't lecture."""
        
    elif intent == 3:
        # Intent: Seeking therapeutic guidance
        prompt = f"""{SYSTEM_MSG} Reference and retain historical context from your conversation history with USER below. USER's most recent message to you is requesting you to give them therapeutic guidance. In your next response back, you must use your deep expertise of psychology and therapy techniques to suggest a solution to the USER to resolve their issue. Keep your responses short and helpful. Don't lecture USER. Try to drill into the issue and progress it towards a solution."""
        
    elif intent == 4:
        # Intent: Self-harm
        prompt = """I am sorry you're feeling this way. Due to my limitations as an AI model, I am incapable of offering you support related to navigating feelings of self-harm or harm to others. If you are feeling severe distress and/or thoughts of self-harm, please dial 988 to connect to the Suicide and Crisis Lifeline. If you are not in an emergency situation, please seek out the support of a professional human therapist or counselor. You can visit websites such as https://www.psychologytoday.com/ to find a therapist near you."""
        
    elif intent == 5:
        # Intent: Small talk
        prompt = SYSTEM_MSG
        
    return prompt
