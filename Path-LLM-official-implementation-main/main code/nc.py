import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import sys

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

train_para, emb_dict = load(emb_file_path)
print(f'Evaluate Node Classification Performance for Model on Dataset!')

label_file_path = 'Cora/nc_dataset/labelnew.dat.train'
label_test_path = 'Cora/nc_dataset/labelnew.dat.test'

class MLP_Decoder(nn.Module):
  def __init__(self, hdim, nclass):
    super(MLP_Decoder, self).__init__()
    self.final_layer = nn.Linear(hdim, nclass)

    self.softmax = nn.Softmax()
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    
  def forward(self, h):
    h = self.final_layer(h)
    output = self.sigmoid(h)
    return output


    
def unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict):
    
    labels, embeddings = [], []    
    for file_path in [label_file_path, label_test_path]:
        with open(file_path,'r') as label_file:
            for line in label_file:
                index, _, _, label = line[:-1].split('\t')
                labels.append(label)
                embeddings.append(emb_dict[index].astype(np.float32))    
    labels, embeddings = np.array(labels).astype(int), np.array(embeddings)  
    
    macro, micro, accuracy = [], [], []
    dim = 4096
    num_class = 7
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=999)
    for train_idx, test_idx in skf.split(embeddings, labels):
        clf = MLP_Decoder(dim,num_class).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr = 0.001)
        best_ma=0
        best_mi=0
        best_ac=0
        
        for i in range(50):
            clf.train()
            criterion = nn.BCELoss()

            pred=clf(torch.tensor(embeddings[train_idx]).to(device)).squeeze()
 
            train_labels = F.one_hot(torch.tensor(labels[train_idx]), num_classes=num_class)
            train_labels = train_labels.to(device).to(torch.float32)

            loss=criterion(pred, train_labels.to(torch.float32))

            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
            clf.eval()
            with torch.no_grad():

                preds = clf(torch.tensor(embeddings[test_idx]).to(device))
            ma=f1_score(labels[test_idx], preds.argmax(dim=1).cpu(), average='macro')
            mi=f1_score(labels[test_idx], preds.argmax(dim=1).cpu(), average='micro')
            ac=accuracy_score(labels[test_idx], preds.argmax(dim=1).cpu())

            if ma>best_ma:
                best_ma=ma
                best_mi=mi

        macro.append(best_ma)
        micro.append(best_mi)

    print(macro)
    print(micro)

    return np.mean(macro), np.mean(micro)


score=unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)
print(score)


new_data = {result_name: score}
with open("nc.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    old_data.update(new_data)
with open("nc.json", "w", encoding="utf-8") as f:
    json.dump(old_data, f, indent=4)