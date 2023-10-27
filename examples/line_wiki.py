
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import node2vec_wiki


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)



if __name__ == "__main__":
    G = nx.read_edgelist('../data/media_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    node2vec_wiki.plot_embeddings(embeddings)
