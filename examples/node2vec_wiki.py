
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import Node2Vec
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import networkx as nx
from sklearn.manifold import TSNE
import csv
import pandas as pd


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/media_labels.txt')
    print(len(X))
    tr_frac = 0.8
    print("test")
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/media_labels.txt')

    labels = []
    nodes = []
    categories = []
    for id in embeddings.keys():
        nodes.append(id)
        labels.append(Y[X.index(id)][0])

        if len(Y[X.index(id)]) != 1:
            if not "/fr" in Y[X.index(id)][1:]:
                categories.append(" ".join(Y[X.index(id)][1:]))
            else:
                categories.append("NoCategory")
        else:
            categories.append("NoCategory")

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    with open("../data/medias.csv") as file:
        with open("nodes.csv", "w") as csvfile:
            headers = ["x", "y", "label", "category"]
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for idx in range(len(node_pos)):
                writer.writerow(
                    {"x": node_pos[idx, 0], "y": node_pos[idx, 1], "label": labels[idx], "category": categories[idx]})

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    data = pd.read_csv("nodes.csv")
    fig = px.scatter(data, x="x", y="y", hover_data="label", color="category")

    fig.show()


if __name__ == "__main__":
    G = nx.read_edgelist('../data/media_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None)
    print(len(G.nodes))
    model = Node2Vec(G, walk_length=10, num_walks=100,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    # evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
