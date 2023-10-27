import networkx as nx
import csv 
import collections
from ural.lru import NormalizedLRUTrie

G=nx.read_gexf("medias_Fr_07_23.gexf")

noeuds=collections.Counter()
for e in G.edges.data() :
    noeuds[e[0]]+=1
    noeuds[e[1]]+=1

with open("medias.csv",'r') as file :
    trie = NormalizedLRUTrie()
    triesub=NormalizedLRUTrie()
    reader = csv.DictReader(file)
    for row in reader :
        trie.set(row['home_page'],row['wheel_category'])
        trie.set(row['home_page'],row['wheel_subcategory'])



with open("medias_Fr_2023+metas2020.csv",'r') as csvfile :
    with open('media_labels.csv','w') as file :
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(file, fieldnames=["id","name","category","sub_category"])
        writer.writeheader()
        for row in reader :
            #if  noeuds[row['ID']] != 0 :
            category=trie.match(row['HOME PAGE']) if trie.match(row['HOME PAGE']) !=None else ""
            subcategory=trie.match(row['HOME PAGE']) if trie.match(row['HOME PAGE']) !=None else ""

            writer.writerow({
                "id":row['ID'],
                "name":row['NAME'],
                "category":category,
                "sub_category": subcategory
                            })