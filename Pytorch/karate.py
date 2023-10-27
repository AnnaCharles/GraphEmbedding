import networkx as nx
from karateclub import DeepWalk

graph=nx.read_gexf("medias_Fr_07_23.gexf")

model = DeepWalk()
model.fit(graph)
embedding = model.get_embedding()
print(embedding)