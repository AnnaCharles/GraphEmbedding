import os.path as osp
import sys
import plotly.express as px

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from tqdm import tqdm
import csv
from collections import Counter,defaultdict


node_label=defaultdict()  #key : id du noeud, value : nom du noeud 
node_class=defaultdict()  #key : id du noeud, value : nom de la categorie

with open("media_labels.csv") as file :
    reader = csv.DictReader(file)
    for row in reader :
        node_label[int(row['id'])]=row['name']
        node_class[int(row['id'])]=row['category']


graph=nx.read_gexf("medias_Fr_07_23.gexf")

nodes=Counter() # key : id du noeud, value : index (numero dans l'array) ( de 0 à nbre de noeud)
for i,node in enumerate(graph.nodes) :
    nodes[node]=i

nodes_from=[]
nodes_to=[]
#construction des deux arrays 
for (x,y) in graph.edges :
    nodes_from.append(nodes[x])
    nodes_to.append(nodes[y])


data_tensor=torch.tensor([nodes_from,nodes_to], dtype=torch.long) 
data=Data(edge_index=data_tensor)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

num_workers = 4 if sys.platform == 'linux' else 0
loader = model.loader(batch_size=150, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 50):
    loss = train()
    #acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

@torch.no_grad()
def plot_points():
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=3,perplexity=5).fit_transform(z)

    coord_x=[]
    coord_y=[]
    coord_z=[]
    categories=[]
    labels=[]
    index=[]

    for i,id in enumerate(node_label.keys()) :
        if node_class[id] != "":
            coord_x.append(z[i, 0])
            coord_y.append(z[i, 1])
            coord_z.append(z[i, 2])
            categories.append(node_class[id])
            index.append(id)
            labels.append(node_label[id])

    fig = px.scatter_3d(data, x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=categories)
    fig.show()


plot_points()
