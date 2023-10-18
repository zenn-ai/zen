import json
import numpy as np
import pandas as pd

sources = json.load(open("who_r_u.json", "r"))
sources = [x["conversations"] for x in sources]
len(sources)

roles = {'human': 'USER', 'gpt': 'ASSISTANT'}
conv_roles = ('USER', 'ASSISTANT')
conversations = []
for i, source in enumerate(sources):
    if roles[source[0]["from"]] != conv_roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    # conv_messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv_roles[j % 2], f"{i}"
        # conv_messages.append(role, sentence["value"])
        conversations.append((role, sentence["value"]))

c1, c2 = [], []
for i, c in enumerate(conversations):
    if i % 2 == 0:
        assert c[0] == "USER"
        c1.append(c[1])
    else:
        assert c[0] == "ASSISTANT"
        c2.append(c[1])

df = pd.DataFrame(columns=["human", "zen"], data=np.transpose([c1, c2]))
print(df.shape)

df = df.replace("Vicuna", "ZenAI", regex=True)
ignore = ["Have a nice day!", "What is up?", "Goodbye"]
df = df[~df.USER.isin(ignore)]

df.to_csv("who_r_u.csv", index=False)
