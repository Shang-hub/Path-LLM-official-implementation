
import pickle as pickle
import warnings
import pickle as pickle
import networkx as nx
import random
from itertools import combinations
import json

warnings.filterwarnings("ignore")



dic_node_index = {}
with open('./PubMed/node.dat','r') as original_meta_file:
    for line in original_meta_file:
        temp1,temp2,temp3=line.split('\t')
        dic_node_index[temp1] = temp2


def create_graph(filename):
    G = nx.Graph()
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            i += 1
            start_node, end_node, weight = line.split('\t')
            
            if start_node not in G or end_node not in G:
                G.add_edge(start_node, end_node, weight=1)
            elif not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node, weight=1)
            else:
                continue
    return G


def create_walklm_graph(filename):
    G = nx.Graph()
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            i += 1
            start_node, end_node, weight = line.split('\t')
            weight = weight.strip()
            if start_node not in G or end_node not in G:
                G.add_edge(start_node, end_node, weight=float(weight))
            elif not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node, weight=float(weight))
            else:
                continue
    return G


def create_weighted_graph(filename):
    G = nx.Graph()
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            i += 1
            start_node, end_node, weight = line.split('\t')
            weight = weight.strip()
            if start_node not in G or end_node not in G:
                G.add_edge(start_node, end_node, weight=float(weight))
            elif not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node, weight=float(weight))
            else:
                continue
    return G


def find_path(MST, nodes):
    paths = []
    node_pairs = list(combinations(nodes, 2))
    for pair in node_pairs:
        shortest_path = nx.shortest_path(MST, source=pair[0], target=pair[1], weight='weight')
        paths.append(shortest_path)
    return paths



# Load the graph
G = create_graph('case_study/weight/PubMed_link.dat')
print("the graph has already been constructed...")
# Randomly pick two nodes
G_nodes = list(G.nodes)
if len(G_nodes) < 2:
    raise ValueError("The graph must contain at least two nodes.")


walklm_G = create_walklm_graph('case_study/weight/PubMed_walklm_weighted_link.dat')
print("the walklm graph has already been constructed...")
# Randomly pick two nodes
walklm_G_nodes = list(walklm_G.nodes)
if len(walklm_G_nodes) < 2:
    raise ValueError("The graph must contain at least two nodes.")


# Load the graph
weighted_G = create_weighted_graph('case_study/weight/PubMed_weighted_link.dat')
print("the weighted graph has already been constructed...")
# Randomly pick two nodes
weighted_G_nodes = list(weighted_G.nodes)
if len(weighted_G_nodes) < 2:
    raise ValueError("The graph must contain at least two nodes.")


largest = max(nx.connected_components(G),key=len)
largest_connected_subgraph = G.subgraph(largest)
subgraph = largest_connected_subgraph
print("largest connected subgraph has been constructed")

largest = max(nx.connected_components(walklm_G),key=len)
largest_connected_subgraph = walklm_G.subgraph(largest)
walklm_subgraph = largest_connected_subgraph
print("largest connected walklm subgraph has been constructed")


largest = max(nx.connected_components(weighted_G),key=len)
largest_connected_subgraph = weighted_G.subgraph(largest)
weighted_subgraph = largest_connected_subgraph
print("largest connected weighted subgraph has been constructed")


node_index_1 = '50156'
node_index_2 = '15726'
node_index_3 = '48251'
node_index_4 = '11495'


nodes = [node_index_1, node_index_2, node_index_3, node_index_4]
iter_num = 1

for i in range(iter_num):

    nodes = random.sample(list(subgraph.nodes),3)

    print("weight=1")
    print("Steiner minimum tree is searching...")
    MST1 = nx.algorithms.approximation.steiner_tree(subgraph, nodes, weight='weight', method="mehlhorn")

    paths = find_path(MST1, nodes)
    text_paths = []
    for path in paths:
        text_path = []
        for id in path:
            text_path.append(dic_node_index[id])
        text_paths.append(text_path)

    print(text_paths)

    print("walklm")
    print("Steiner minimum tree is searching...")
    MST2 = nx.algorithms.approximation.steiner_tree(walklm_subgraph, nodes, weight='weight', method="mehlhorn")

    paths = find_path(MST2, nodes)
    text_paths = []
    for path in paths:
        text_path = []
        for id in path:
            text_path.append(dic_node_index[id])
        text_paths.append(text_path)

    print(text_paths)
    

    print("our method")
    print("Steiner minimum tree is searching...")
    MST3 = nx.algorithms.approximation.steiner_tree(weighted_subgraph, nodes, weight='weight', method="mehlhorn")

    paths = find_path(MST3, nodes)

    text_paths = []
    for path in paths:
        text_path = []
        for id in path:
            text_path.append(dic_node_index[id])
        text_paths.append(text_path)

    print(text_paths)