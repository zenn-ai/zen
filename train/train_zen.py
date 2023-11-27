import os
import pandas as pd
import random
from typing import Dict
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

from dataclasses import dataclass, field
from typing import cast, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    HfArgumentParser,
    PreTrainedTokenizer
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.integrations import deepspeed

from peft import LoraConfig, get_peft_model

from fastchat.conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle

load_dotenv("../.env")
hf_token = os.getenv('HF_TOKEN')

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SYSTEM_MSG = """Your name is Zen and you're a mental health counselor. Please have a conversation with your patient and provide them with a helpful response to their concerns."""

DATA_DIR = "../data"

try:
    register_conv_template(
        Conversation(
            name="Zen",
            system_message=SYSTEM_MSG,
            roles=("USER", "ASSISTANT"),
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    )
except AssertionError:
    pass


def get_df():
    csv_files = ["mental_health_chatbot_dataset.csv", "psychology-dataset.csv"]
    df = pd.DataFrame()

    for p in csv_files:
        df1 = pd.read_csv(os.path.join(DATA_DIR, p))[["human", "zen"]]
        df1 = df1.rename(columns={"human": "USER", "zen": "ASSISTANT"})
        df1 = df1.drop_duplicates(subset=["USER", "ASSISTANT"], keep="first", ignore_index=True)

        df1["USER"] = df1.USER.str.replace('\s+', ' ', regex=True)
        df1["USER"] = df1.USER.str.replace(r'\.([a-zA-Z0-9])', r'. \1', regex=True)
        df1["ASSISTANT"] = df1.ASSISTANT.str.replace('\s+', ' ', regex=True)
        df1["ASSISTANT"] = df1.ASSISTANT.str.replace(r'\.([a-zA-Z0-9])', r'. \1', regex=True)

        df = pd.concat([df, df1])

    df = df.sample(frac=1, random_state=42)
    return df


greetings = [
    ("Hello!", "Hello! How can I help you today?"),
    ("What's up?", "Hello! How can I help you today?"),
    ("What is up?", "Hello! How can I help you today?"),
    ("Hi, how are you?", "Hello! How can I help you today?")
]
goodbyes = [
    ('Have a nice day!', 'You too! If you have any concerns or questions about mental health, please feel free to ask anytime.'),
    ('Goodbye', "Goodbye! If you have any concerns or questions about mental health, please feel free to ask anytime."),
    ('Talk to you later!', "Until next time! I'll be here if you need me."),
    ('Alright, thanks for your help.', "You're welcome! Have a great day ahead and feel free to reach out anytime if you have any concerns or need someone to talk to.")
]


def get_conversations(df, reset, identity=False):
    conv = get_conv_template("Zen")
    conversations = []
    
    conv.messages = []
    for index, row in df.iterrows():
        if reset:
            conv.messages = []
        
        if identity and random.choices([0, 1], weights=[0.25, 0.75], k=1)[0]:
            greet = random.sample(greetings)
            conv.append_message("USER", greet[0])
            conv.append_message("ASSISTANT", greet[1])
            
        conv.append_message("USER", row["USER"])
        conv.append_message("ASSISTANT", row["ASSISTANT"])
        
        if identity and random.choices([0, 1], weights=[0.25, 0.75], k=1)[0]:
            bye = random.sample(greetings)
            conv.append_message("USER", bye[0])
            conv.append_message("ASSISTANT", bye[1])
            
        conversations.append(conv.get_prompt())
    
    return conversations


def get_therapy_conv():
    df = pd.read_csv(os.path.join(DATA_DIR, "PALM_Alexander_Street.csv"))
    df = df.dropna(ignore_index=True)
    df = df.drop(index=0)
    df.reset_index(drop=True, inplace=True)
    
    conv_ids = df.conv_id.unique()
    conversations = []
    for c in conv_ids:
        conv_df = df[df.conv_id == c]
        conversations += get_conversations(conv_df, reset=False)
    
    return conversations


def get_identity_conv():
    
    df = pd.read_csv(os.path.join(DATA_DIR, "who_r_u.csv"))
    df = df.rename(columns={"human": "USER", "zen": "ASSISTANT"})

    df["USER"] = df.USER.str.replace('\s+', ' ', regex=True)
    df["USER"] = df.USER.str.replace(r'\.([a-zA-Z0-9])', r'. \1', regex=True)
    df["ASSISTANT"] = df.ASSISTANT.str.replace('\s+', ' ', regex=True)
    df["ASSISTANT"] = df.ASSISTANT.str.replace(r'\.([a-zA-Z0-9])', r'. \1', regex=True)
    
    return get_conversations(df, reset=True, identity=True)


def rank0_print(*args):
    print(*args)


def preprocess(conversations, tokenizer):

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = " ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split("</s>")
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )
                print(conversation)
                print(input_ids)
                print(target)
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                print("==================================================================")

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


model_id = "lmsys/vicuna-13b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    model_max_length=1024,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, conversations, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        data_dict = preprocess(conversations, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


df = get_df()
conversations = get_conversations(df, reset=True)
conversations += get_therapy_conv()
conversations += get_identity_conv()

train, test = train_test_split(conversations, test_size=1000, random_state=42)
print(len(conversations), len(train), len(test))

import pickle

with open(os.path.join(DATA_DIR, "test.p"), "wb") as f:
    pickle.dump(test, f)

train_dataset = SupervisedDataset(train, tokenizer)
eval_dataset = SupervisedDataset(test, tokenizer)
data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

train_args = TrainingArguments(
    output_dir="/home/jupyter/therapy-bot/models/Zen",
    max_steps=10000,
    optim="adamw_torch",
    per_device_train_batch_size=1,
    remove_unused_columns=False,
    learning_rate=1e-5,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    fp16=True,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=200,
    save_steps=500,
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    logging_steps=500,
    report_to="tensorboard",
    hub_model_id="kmnis/Zen",
    # hub_strategy="checkpoint",
    hub_private_repo=True,
    load_best_model_at_end=True,
    push_to_hub=True,
    deepspeed="ds_config_zero3.json",
    label_names=["labels"],
    hub_token=hf_token
)

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens", "lm_head", "gate_proj", "up_proj", "down_proj"]

# preparing lora configuration
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CASUAL_LM",
    target_modules=target_modules,
)

# loading the base model
model = AutoModelForCausalLM.from_pretrained(
    model_id, use_cache=not train_args.gradient_checkpointing
)
if train_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# getting peft model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# creating trainer
trainer = Trainer(
    model=model, tokenizer=tokenizer, args=train_args, **data_module
)
model.config.use_cache = False

# trainer.accelerator.print(f"{trainer.model}")
trainer.model.print_trainable_parameters()

trainer.train()

# save model on main process
trainer.accelerator.wait_for_everyone()
state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
if trainer.accelerator.is_main_process:
    unwrapped_model.save_pretrained(train_args.output_dir, state_dict=state_dict)
trainer.accelerator.wait_for_everyone()

# save everything else on main process
if trainer.args.process_index == 0:
    trainer.model.save_pretrained(train_args.output_dir, safe_serialization=True)


# if __name__ == "__main__":
#     main()
