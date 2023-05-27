import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

def load_model():
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/vicuna-13B-1.1-HF")
    model = LlamaForCausalLM.from_pretrained("TheBloke/vicuna-13B-1.1-HF", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, "kmnis/ZenAI", torch_dtype=torch.float16)
    return model, tokenizer