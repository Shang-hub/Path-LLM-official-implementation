from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import AutoTokenizer
import torch
import random
import os
import math
import pickle as pickle
import numpy as np
import pandas as pd
import warnings
import pickle as pickle
import sys
import spacy
import pytextrank

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("textrank")

device = torch.device("cuda:5")

model_path = sys.argv[1]
embedding_path = sys.argv[2]

access_token = access_token

model = model_path

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)


model = AutoModelForCausalLM.from_pretrained(model, token=access_token).to(device)


df = pd.read_csv("ARXIV/ogb_arxiv.csv")

node_set = []

with open('./ARXIV/nc_dataset/newtokenlabel.dat.train','r') as f1:
    for line in f1:
        idx,_ ,_ ,_ =line.split('\t')
        node_set.append(idx)

with open('./ARXIV/nc_dataset/newtokenlabel.dat.test','r') as f2:
    for line in f2:
        idx,_ ,_ ,_ =line.split('\t')
        node_set.append(idx)


with open('./ARXIV/lp_dataset/link.dat.test','r') as f4:
    for line in f4:
        idx1, idx2 ,_  =line.split('\t')
        node_set.append(idx1)
        node_set.append(idx2)

node_set = set(node_set)



def title(raw_text):

    title = ''.join(char for char in raw_text if char not in ['\\', '#', '.', ',', '"','"']).strip()

    words = title.split()
    if len(words) <= 10:
        return title
    else:
        doc = nlp(title)

        data=[]
        for phrase in doc._.phrases[:10]:
            data.append(str(phrase.chunks[0]))

        return ' '.join(data)

def get_word_embeddings(word):
    encoded_input = tokenizer(word, return_tensors='pt').to(device)
    output = model.model(**encoded_input).last_hidden_state
    embed = output[:,1:,:].mean(1).squeeze().detach().cpu().numpy()
    return embed

hid_dim = 4096 


with open(embedding_path, 'w') as file:
    file.write('arxiv\n')
    for id in node_set:
        id = int(id)
        word = title(df['title'][id]+df['abstract'][id])
        emb = get_word_embeddings(word)
        file.write(f'{id}\t')
        file.write(' '.join(emb.astype(str)))
        file.write('\n')