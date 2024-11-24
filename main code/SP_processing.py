
import warnings
import torch
from ogb.nodeproppred import NodePropPredDataset
import networkx as nx
import random
import pandas as pd
import numpy as np
import spacy
import pytextrank
import nltk


warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")


nlp.add_pipe("textrank")

def idx_process(mode):
    if mode == 'cora':
        path = "single_graph/Cora/cora.pt"
        data = torch.load(path)
        val_mask = data.val_masks
        edge_list = data.edge_index.T.numpy().tolist()
        raw_text = data.raw_text
        labels = data.y
        category = data.category_names

        train_set = val_mask[2]
        test_set = ~train_set
        train_idx = np.where(train_set == False)[0]
        test_idx = np.where(test_set == False)[0]

    elif mode == 'citeseer':
        path = "single_graph/Citeseer/citeseer.pt"
        data = torch.load(path)
        test_mask = data.test_masks
        edge_list = data.edge_index.T.numpy().tolist()
        labels = data.y
        raw_text = data.raw_texts

        test_set = test_mask[0]
        train_set = ~test_set
        train_idx = np.where(train_set == False)[0]
        test_idx = np.where(test_set == False)[0]

        category = data.category_names

    return train_idx, test_idx, edge_list, raw_text, labels, category


# mode = 'cora' or 'citeseer'
train_idx, test_idx, edge_list, raw_text, labels, category = idx_process(mode='cora')


def create_DAG(edge_list):
    DAG = nx.DiGraph()
    i=0
    for edge in edge_list:
        i+=1
        start_node = edge[0]
        end_node = edge[1]
        DAG.add_edge(start_node, end_node)
    return DAG

DAG = create_DAG(edge_list)
print("the graph has already been constructed...")
# Randomly pick two nodes
nodes = list(DAG.nodes)
if len(nodes) < 2:
    raise ValueError("The graph must contain at least two nodes.")


train_idx_list = train_idx.tolist()

def add_tuple(paths):
    paths_with_link_types = []
    for path in paths:
        path_with_link_types = []
        for i in range(len(path) - 1):
            path_with_link_types.append((path[i], path[i+1]))
        paths_with_link_types.append(path_with_link_types)
    return paths_with_link_types


def bfs_single_source_paths(G, start_node):
    try:
        candidate_set = []
        shortest_paths = nx.single_source_shortest_path(G, start_node)
        for target, path in shortest_paths.items():
            if len(path) >=10:
                candidate_set.append(target)
        return candidate_set
    except nx.NetworkXNoPath:
        None


def sampling_shortest_paths_with_min_length(G, source, target):
    try:
        paths = list(nx.all_shortest_paths(G, source, target, weight=None))

        if len(paths) >1:  #10 and <=10
            selected_10paths = random.sample(paths, k=1)
        else:
            selected_10paths = paths

        long_10paths = add_tuple(selected_10paths)

        return long_10paths
    except nx.NetworkXNoPath:
        return None
    


def title(raw_text):

    title = ''.join(char for char in raw_text if char not in ['\\', '#', '.', ',', '"','"']).strip()

    words = title.split()
    if len(words) <= 10:
        return title
    else:
        doc = nlp(title)
        # examine the top-ranked phrases in the document
        data=[]
        for phrase in doc._.phrases[:10]:
            data.append(str(phrase.chunks[0]))
        return ' '.join(data)
    

def find_shortest_paths_with_min_length(DAG, source, target, min_length, max_length):
    # Find one shortest path between the two nodes
    try:
        paths = list(nx.all_shortest_paths(DAG, source, target, weight=None))
        filtered_paths = [path for path in paths if len(path) >= min_length and len(path) <= max_length]
        # filtered_paths = [path for path in paths if len(path) >= min_length]
        filtered_paths = add_tuple(filtered_paths)
        return filtered_paths
    except nx.NetworkXNoPath:
        pass


# with open('./Cora/STP/SP_min2_max4.txt', 'w') as f1:
with open('./Cora/STP/SP_long10_short3_num1.txt', 'w') as f2:
    with open('./Cora/STP/SP_long10_num1.txt', 'w') as f1:
        source = random.sample(nodes, 1000)

        for start_node in source:
            candidate_set = bfs_single_source_paths(DAG, start_node)
            if candidate_set:
                target = random.choice(candidate_set)
                long_10paths = sampling_shortest_paths_with_min_length(DAG, start_node, target)
                if long_10paths:
                    for path in long_10paths:
                        i = 0
                        for start, end in path:
                            if i == 0:
                                
                                text_line =  "paper with content " + title(raw_text[start]) + ' ' + "cites" +' ' + "paper with content " + title(raw_text[end])
                                i += 1
                            else:
                                text_line = text_line + ' ' + "cites" +' ' + "paper with content"+ ' ' + title(raw_text[end])
                        print(text_line, end='',file=f1)
                        # print(text_line.encode("utf-8").decode("latin1"), end='',file=f1)
                        print("\n",end='',file=f1)
                    for path in long_10paths:
                        j = 0
                        for start, end in path:
                            if j == 0:
                                text_line =  "paper with content " + title(raw_text[start]) + ' ' + "cites" +' ' + "paper with content " + title(raw_text[end])
                                j += 1
                            elif j>=2:
                                print(text_line, end='',file=f2)
                                # print(text_line.encode("utf-8").decode("latin1"), end='',file=f2)
                                print("\n",end='',file=f2)
                                text_line =  "paper with content " + title(raw_text[start]) + ' ' + "cites" +' ' + "paper with content " + title(raw_text[end])
                                j = 1                           
                            else:
                                text_line = text_line + ' ' + "cites" +' ' + "paper with content"+ ' ' + title(raw_text[end])
                                j+=1
                        print(text_line, end='',file=f2)
                        # print(text_line.encode("utf-8").decode("latin1"), end='',file=f2)
                        print("\n",end='',file=f2)

