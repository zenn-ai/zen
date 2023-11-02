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
        prompt = """
                    The client is referencing a previous conversation you had with them.
                    Your response to the client should portray that you remember the past conversation you've had.
                    Use the below context from your previous conversation to respond in an appropriate matter.
                    Context:
                 """
    
    elif intent == 2:
        # Intent: Venting
        prompt = """
                    The client is venting to you. Be a good listener.
                    Utilize your therapeutic techniques such as motivational interviewing to probe further.
                    Don't be aggressive in trying to solve the client's issue.
                 """
        
    elif intent == 3:
        # Intent: Seeking therapeutic guidance
        prompt = """
                    The client is seeking your therapeutic guidance.
                    Leverage your deep expertise of therapeutic techniques to help the client resolve their issue.
                    Be engaging, don't lecture the client.
                 """
    
    elif intent == 4:
        # Intent: Self-harm
        
        prompt = """
                    Respond exactly as following:
                    I am sorry you're feeling this way. Due to my limitations as an AI model, I am incapable of offering you support related to navigating feelings of self-harm or harm to others. 
                    If you are feeling severe distress and/or thoughts of self-harm, please dial 988 to connect to the Suicide and Crisis Lifeline.
                    If you are not in an emergency situation, please seek out the support of a professional human therapist or counselor. You can visit websites such as https://www.psychologytoday.com/ to find a therapist near you.
                    
                 """
        
    elif intent == 5:
        # Intent: Small talk
        
        prompt = """
                    The client is engaging in casual small talk with you. Respond back in a light-hearted casual tone and sound interested in the topic they are discussing.
                 """
        
    return prompt