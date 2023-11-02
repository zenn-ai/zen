import json
import os
import warnings

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftConfig, PeftModel

from utils import save_conversation, load_conversation

from fastchat.utils import get_gpu_memory, is_partial_stop, is_sentence_complete, get_context_length
from fastchat.conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle
from fastchat.serve.inference import generate_stream

from flask import Flask, jsonify, request
import streamlit as st

warnings.filterwarnings('ignore')


try:
    register_conv_template(
        Conversation(
            name="ZenAI",
            system_message="Your name is ZenAI and you're a therapist. Please have a conversation with your patient and provide them with a helpful response to their concerns.",
            roles=("USER", "ASSISTANT"),
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    )
except AssertionError:
    pass

SYSTEM_MSG = """Your name is ZenAI and you're a therapist. Please have a conversation with your patient and provide them with a helpful response to their concerns."""


def load_model(model_path, num_gpus, max_gpu_memory=None):
    
    kwargs = {"torch_dtype": torch.float16}
    if num_gpus != 1:
        kwargs["device_map"] = "auto"
        if max_gpu_memory is None:
            kwargs[
                "device_map"
            ] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {
                i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                for i in range(num_gpus)
            }
        else:
            kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, use_fast=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        
        return model, tokenizer, base_model


use_vicuna = False
num_gpus = 4
max_gpu_memory = "12GiB"
model_path = "/home/jupyter/therapy-bot/models/zenai_sample/"
if use_vicuna:
    _, tokenizer, model = load_model(model_path, num_gpus, max_gpu_memory)
else:
    model, tokenizer, _ = load_model(model_path, num_gpus, max_gpu_memory)


def chat_streamlit(
    model, tokenizer, username, question,
    device, num_gpus, max_gpu_memory,
    conv_template="ZenAI", system_msg=SYSTEM_MSG,
    temperature=0.7, repetition_penalty=1.0, max_new_tokens=512,
    dtype=torch.float16,
    judge_sent_end=True
):
    conv_path = f"saved_conversations/{username}.json"
    context_len = get_context_length(model.config)

    def new_chat():
        conv = get_conv_template(conv_template)
        conv.set_system_message(system_msg)
        return conv

    if os.path.exists(conv_path):
        conv = load_conversation(conv_path)
    else:
        conv = new_chat()

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    gen_params = {
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    output_stream = generate_stream(
        model,
        tokenizer,
        gen_params,
        device,
        context_len=context_len,
        judge_sent_end=judge_sent_end,
    )
    print("##############################################################################")
    print(prompt)
    print("##############################################################################")
    
    return conv, output_stream, conv_path


def stream_output(output_stream):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            # print(" ".join(output_text[pre:now]), end=" ")
            pre = now
    # print(" ".join(output_text[pre:]))
    return " ".join(output_text)


app = Flask(__name__)

def get_answer():
    question = request.json.get("question", "")
    print("******************************************", question, "******************************************")
    conv, output_stream, conv_path = chat_streamlit(
        model=model,
        tokenizer=tokenizer,
        username="kmanish",
        question=question,
        device="cuda",
        num_gpus=4,
        max_gpu_memory="12GiB"
    )
    answer = stream_output(output_stream).strip()
    print("******************************************", answer, "******************************************")
    conv.update_last_message(answer)
    save_conversation(conv, conv_path)
    
    return answer
    

@app.route('/api/data', methods=["POST"])
def get_data():
    answer = get_answer()
    return jsonify({"answer": answer}), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=False)