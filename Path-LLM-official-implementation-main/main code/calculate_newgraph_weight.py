import torch
import random
import os
import math
import pickle as pickle
import numpy as np
import pandas as pd
import warnings
import pickle as pickle
import networkx as nx
import random

warnings.filterwarnings("ignore")



emb_file_path = './emb.dat'
emb_dict = {}


with open(emb_file_path,'r') as emb_file:        
    for i, line in enumerate(emb_file):
        if i == 0:
            train_para = line[:-1]
        else:
            index, emb = line[:-1].split('\t')
            emb_dict[index] = np.array(emb.split()).astype(np.float32)


def cal_dis(index1, index2):
    vec1 = torch.tensor(emb_dict[str(index1)])
    vec2 = torch.tensor(emb_dict[str(index2)])
    sim = torch.cosine_similarity(vec1, vec2, dim=0)
    #distance
    if sim <= 0.0:
        sim = torch.tensor(1e-10)
    if sim < 1.0:
        distance = -torch.log(sim)
    else:
        distance = torch.log(sim)
    return distance



filename = "PubMed/link.dat"
graph_weight_file = "./weighted_link.dat"


with open(graph_weight_file, 'w') as f:
    with open(filename, 'r') as file:
        for line in file:         
            start_node, end_node, link_class, edge_class = line.split('\t')
            weight = cal_dis(start_node, end_node).numpy()
            text_line =  start_node + '\t' + end_node + '\t' + str(weight)
            print(text_line, end='',file=f)
            print("\n",end='',file=f)