import json
import os
from fastchat.conversation import get_conv_template


def save_conversation(conv, path):
    with open(path, "w") as outfile:
        json.dump(conv.dict(), outfile)

            
def load_conversation(path):
    
    with open(path, "r") as infile:
        new_conv = json.load(infile)

    conv = get_conv_template(new_conv["template_name"])
    conv.set_system_message(new_conv["system_message"])
    conv.messages = new_conv["messages"]
    
    return conv