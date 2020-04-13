import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

## First steps
# Asymetric Digraph
G_asymmetric = nx.DiGraph()
G_asymmetric.add_edge('A','B')
G_asymmetric.add_edge('A','D')
G_asymmetric.add_edge('C','A')
G_asymmetric.add_edge('D','E')
nx.spring_layout(G_asymmetric)
nx.draw_networkx(G_asymmetric)
plt.show()

# Shortest path
nx.shortest_path(G_asymmetric, 'A', 'E')

## First file
# Extract nodes and edges
f = open("soc-Epinions1.txt", "r")
start = 4
f_line = f.readlines()[start:]

max_nodes = 100
N_nodes = min(max_nodes , 75879)
l_nodes = [i for i in range(N_nodes)]

l_nodes =  []
for l in f_line:
    if max(list(map(int,l.split()))) < max_nodes:
        l_nodes.append( list(map(int,l.split())) )

##
# Create Graph
G_data = nx.DiGraph()
G_data.add_nodes_from(l_nodes)
G_data.add_edges_from(l_nodes)
nx.draw_networkx(G_data)
nx.spring_layout(G_data)
plt.show()

## Project
# Nodes
import random
from networkx.algorithms import bipartite

class Nodes():
    def __init__(self, features, special_f, num):
        self.num = num
        self.features = features
        self.special_f = special_f

class Edges():
    def __init__(self, Node_1, Node_2):
        self.features = [np.abs(Node_1.features[i] - np.abs(Node_2.features[i])) for i in range(3)]
        self.theta = np.random.dirichlet(np.ones(len(features)), size = 1)
        self.proba_activation = np.dot(self.theta, self.features)


G_project = nx.Graph()
list_nodes = []
list_special_f = ['A', 'B', 'C']

list_f = [[0,1] , [0,1] , [0,1]  ]
N_nodes = 1000


list_Nodes = []
for i in range(N_nodes):
    special_f = random.choice(list_special_f)
    features = [random.choice(list_f[j]) for j in range(3)]
    list_Nodes.append( Nodes( features = features, special_f = special_f, num = i) )


set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_Nodes:
    if node.special_f == 'C':
        set_right_num.append(node.num)
        set_right_nodes.append(node)
    else:
        set_left_num.append(node.num)
        set_left_nodes.append(node)


list_Edges = []
set_edges = []
for node_r in set_right_nodes:
    for node_l in set_left_nodes:
        if (np.random.random() < 0.2) :
            list_Edges.append(Edges(node_r, node_l))
            set_edges.append((node_r.num, node_l.num))

G_project.add_nodes_from(set_right_num, bipartite = 1)
G_project.add_nodes_from(set_left_num, bipartite = 0)
G_project.add_edges_from(set_edges)

nx.draw_networkx(G_project, labels = None)
nx.spring_layout(G_project)
plt.show()