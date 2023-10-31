import networkx as nx
from karateclub import DeepWalk
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import csv
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],subplot_titles=["DeepWalk", "Kmeans"]
)

node_label=defaultdict()  #key : id du noeud, value : nom du noeud 
node_class=defaultdict()  #key : id du noeud, value : nom de la categorie

with open("media_labels.csv") as file :
    reader = csv.DictReader(file)
    for row in reader :
        node_label[int(row['id'])]=row['name']
        node_class[int(row['id'])]=row['category']

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

z = TSNE(n_components=3,perplexity=5).fit_transform(embedding)

coord_x=defaultdict(list)
coord_y=defaultdict(list)
coord_z=defaultdict(list)
categories=[]
labels=defaultdict(list)
index=[]

for idx in graph_mod.nodes :
    idd = list(id)[idx]
    if node_class[int(idd)] != "":
        coord_x[node_class[int(idd)]].append(z[idx, 0])
        coord_y[node_class[int(idd)]].append(z[idx, 1])
        coord_z[node_class[int(idd)]].append(z[idx, 2])
        categories.append(node_class[int(idd)])
        index.append(id)
        labels[node_class[int(idd)]].append(node_label[int(idd)])


colors=["#0000FF","#FF4040","#ED9121","#7FFF00"]
for c,cat in enumerate(["Mainstream Media", "Periphery", "Opinion Journalism","Counter-Informational Space"]) :
    fig.add_trace(go.Scatter3d(
        x=coord_x[cat], y=coord_y[cat], z=coord_z[cat], hovertext=labels[cat],
        legendgroup="group"+str(c),  # this can be any string, not just "group"
        name=cat,
        mode="markers",
        marker=dict(color=colors[c], size=10, opacity=0.8)
    ),row=1, col=1)

#fig = px.scatter_3d(x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=categories, color_discrete_map = {"": "grey"})
#fig.show()
# fig.add_trace(go.Scatter3d(x=coord_x, y=coord_y,z=coord_z,name="DeepWalk", hovertext=labels),
#               row=1, col=1)

#clustering
X=np.array(embedding)
kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto").fit(X)

centers = kmeans.cluster_centers_
couleur_clusters={"0":"#0000FF","1":"#FF4040","2":"#ED9121","3":"#7FFF00","4":"#8B8878",'5':"#8FBC8F","6":"#9932CC","7":"#97FFFF","8":"#FF1493","9":"#FFD700","10":"#8B6914","11":"#FF6A6A",
                  "12":"#F0E68C","13":"#CD8162","14":"#C71585","15":"#33A1C9"}

coord_x=[]
coord_y=[]
coord_z=[]
categories=[]
labels=[]
index=[]
col=[]


for idx in graph_mod.nodes :
    idd = list(id)[idx]
    if node_class[int(idd)] !="":
        coord_x.append(z[idx, 0])
        coord_y.append(z[idx, 1])
        coord_z.append(z[idx, 2])
        categories.append(node_class[int(idd)])
        index.append(id)
        col.append(couleur_clusters[str(kmeans.labels_[idx])])
        labels.append(node_label[int(idd)])



#fig = px.scatter_3d(x=coord_x, y=coord_y,z=coord_z, hover_name=labels, color=col)
#fig.show()

fig.add_trace(go.Scatter3d(x=coord_x, y=coord_y,z=coord_z, hovertext=labels, mode='markers',
    marker=dict(
        size=10,
        colorscale='Inferno',
        color=col,                # set color to an array/list of desired value   # choose a colorscale
        opacity=0.8,
        showscale=True,
    )),
              row=1, col=2)

fig.update_layout(height=700,showlegend=True,legend_orientation="h",coloraxis_colorbar=dict(yanchor="top", y=0, x=1,
                                          ticks="outside"))
fig.show()