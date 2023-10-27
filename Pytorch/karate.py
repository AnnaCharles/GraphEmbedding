import networkx as nx
from karateclub import DeepWalk
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import csv
import plotly.express as px

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
#graph_mod.add_nodes_from([i for i in range(len(id))])

nodes = Counter()  #id de noeud, index dans l'array des noeuds
count=0
for idx in id:
    if node_class[int(idx)] !="":
        nodes[idx]=count
        graph_mod.add_node(count)
        count+=1
print(nodes)
print(len(nodes))

for u,v in graph.edges :
    if node_class[int(u)] !="" and node_class[int(v)] !="":
        #index_u=list(graph.nodes).index(u)
        #index_v=list(graph.nodes).index(v)
        index_u=nodes[u]
        index_v=nodes[v]
        graph_mod.add_edge(index_u, index_v)

model = DeepWalk()
model.fit(graph_mod)
embedding = model.get_embedding()
print(len(embedding))


z = TSNE(n_components=3,perplexity=5).fit_transform(embedding)

coord_x=[]
coord_y=[]
coord_z=[]
categories=[]
labels=[]
index=[]

for i,id in enumerate(nodes.keys()) :
    coord_x.append(z[i, 0])
    coord_y.append(z[i, 1])
    coord_z.append(z[i, 2])
    categories.append(node_class[int(id)])
    index.append(id)
    labels.append(node_label[int(id)])

fig = px.scatter_3d(x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=categories)
fig.show()
