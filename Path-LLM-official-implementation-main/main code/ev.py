import numpy as np
import torch
import torch.nn.functional as F
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import pandas as pd
import  json
import sys

warnings.filterwarnings("ignore")

device = torch.device("cuda:1")
embedding_path = sys.argv[1]
result_name = sys.argv[2]
        
print("cora")

def load(emb_file_path):
    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
        
    return train_para, emb_dict  


emb_file_path = embedding_path
train_para, emb_dict = load(emb_file_path)  #node and embeddings


class MLP_Decoder(nn.Module):
  def __init__(self, hdim, nclass):
    super(MLP_Decoder, self).__init__()
    self.hidden_layer = nn.Linear(hdim,128)
    self.relu=nn.ReLU()
    self.final_layer = nn.Linear(128, nclass)
    self.softmax = nn.Softmax()
    self.sigmoid = nn.Sigmoid()
  def forward(self, h):
    h=self.relu(self.hidden_layer(h))
    output = self.sigmoid(self.final_layer(h))
    return output
  
class Link_MLP(nn.Module):
    def __init__(self, disease_dim,n_class):
        super(Link_MLP, self).__init__()
        self.decoder = MLP_Decoder(disease_dim, n_class)
    def forward(self, dist):
        pred= self.decoder(dist)
        return pred
    


def lp_evaluate(test_file_path, emb_dict):
    
    posi, nega = defaultdict(set), defaultdict(set)
    with open(test_file_path, 'r') as test_file:
        for line in test_file:
            left, right, label = line[:-1].split('\t')
            if label=='1':
                posi[left].add(right)
            elif label=='0':
                nega[left].add(right)
                
    edge_embs, edge_labels = defaultdict(list), defaultdict(list)
    for left, rights in posi.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left]*emb_dict[right])
            edge_labels[left].append(1)
    for left, rights in nega.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left]*emb_dict[right])
            edge_labels[left].append(0)
            
    for node in edge_embs:
        edge_embs[node] = np.array(edge_embs[node])
        edge_labels[node] = np.array(edge_labels[node])
    
    auc, acc = cross_validation(edge_embs, edge_labels)
    
    return auc, acc

def cross_validation(edge_embs, edge_labels):
    
    auc, acc = [], []
    seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)

    seed=1

    skf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((num_nodes,1)), np.zeros(num_nodes))):
        
        print(f'Start Evaluation Fold {fold}!')
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = [], [], [], []
        for each in train_idx:
            train_edge_embs.append(edge_embs[seed_nodes[each]])
            train_edge_labels.append(edge_labels[seed_nodes[each]])
        for each in test_idx:
            test_edge_embs.append(edge_embs[seed_nodes[each]])
            test_edge_labels.append(edge_labels[seed_nodes[each]])
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = np.concatenate(train_edge_embs), np.concatenate(test_edge_embs), np.concatenate(train_edge_labels), np.concatenate(test_edge_labels)        
        best_auc=0
        best_acc=0
        dim = 4096

        clf = Link_MLP(dim,1).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr = 0.001)
        for i in range(100):
            clf.train()
            criterion = nn.BCELoss()
            pred=clf(torch.tensor(train_edge_embs).to(device)).squeeze()

            train_edge_labels
            loss = criterion(pred, torch.tensor(train_edge_labels).to(device).to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clf.eval()
            with torch.no_grad():

                preds = clf(torch.tensor(test_edge_embs).to(device))
            auc1=roc_auc_score(test_edge_labels, preds.cpu())

            prediction_score = preds.squeeze().cpu().numpy()

            normalized_array = (prediction_score - np.min(prediction_score)) / (np.max(prediction_score) - np.min(prediction_score))
            y_pred = np.where(normalized_array > 0.5, 1, 0)
            acc1 = accuracy_score(test_edge_labels, y_pred)


            if auc1>best_auc:
                best_auc=auc1
                best_acc=acc1

        auc.append(best_auc)

        acc.append(best_acc)
    print(auc)
    print(acc)
    return np.mean(auc), np.mean(acc)


link_test_file= 'Cora/lp_dataset/link.dat.test'

scores = lp_evaluate(link_test_file, emb_dict)

print(scores)


new_data = {result_name: scores}
with open("lp.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    old_data.update(new_data)

with open("lp.json", "w", encoding="utf-8") as f:
    json.dump(old_data, f,indent=4)