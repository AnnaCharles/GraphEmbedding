import networkx as nx
from karateclub import DeepWalk
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import csv
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

node_label=defaultdict()  #key : id du noeud, value : nom du noeud 
node_class=defaultdict()  #key : id du noeud, value : nom de la categorie

with open("media_labels.csv") as file :
    reader = csv.DictReader(file)
    for row in reader :
        node_label[int(row['id'])]=row['name']
        node_class[int(row['id'])]=row['sub_category']

graph=nx.read_gexf("medias_Fr_07_23.gexf")

id=graph.nodes

#remplace les id par des index 
graph_mod=nx.DiGraph()
graph_mod.add_nodes_from([i for i in range(len(id))])
nodes = Counter()  #id de noeud, index dans l'array des noeuds
count=0
for idx in id:
    if node_class[int(idx)] !="":
        nodes[idx]=count
        #graph_mod.add_node(count)
        count+=1


for u,v in graph.edges :
    index_u=list(graph.nodes).index(u)
    index_v=list(graph.nodes).index(v)
    #index_u=nodes[u]
    #index_v=nodes[v]
    graph_mod.add_edge(index_u, index_v)

model = DeepWalk()
model.fit(graph_mod)
embedding = model.get_embedding()

print(embedding)
z = TSNE(n_components=3,perplexity=5).fit_transform(embedding)

coord_x=[]
coord_y=[]
coord_z=[]
categories=[]
labels=[]
index=[]

for idx in graph_mod.nodes :
    coord_x.append(z[idx, 0])
    coord_y.append(z[idx, 1])
    coord_z.append(z[idx, 2])
    idd = list(id)[idx]
    categories.append(node_class[int(idd)])
    index.append(id)
    labels.append(node_label[int(idd)])

fig = px.scatter_3d(x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=categories, color_discrete_map = {"": "grey"})
fig.show()

#clustering
X=np.array(embedding)
kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto").fit(X)


centers = kmeans.cluster_centers_

coord_x=[]
coord_y=[]
coord_z=[]
categories=[]
labels=[]
index=[]
for idx in graph_mod.nodes :
    coord_x.append(z[idx, 0])
    coord_y.append(z[idx, 1])
    coord_z.append(z[idx, 2])
    idd = list(id)[idx]
    categories.append(node_class[int(idd)])
    index.append(id)
    labels.append(node_label[int(idd)])

col=[str(i) for i in list(kmeans.labels_)]
fig = px.scatter_3d(x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=col)
fig.show()
