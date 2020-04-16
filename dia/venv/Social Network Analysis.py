import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Models.Graph import *

## Project
# Nodes
import random

from networkx.algorithms import bipartite



node1 = Nodes(1)
node2 = Nodes(2)
edge = Edges(node1, node2)

G_project = nx.Graph()

N_nodes = 1000

list_Nodes = []
for i in range(N_nodes):
    list_Nodes.append(Nodes(i))

set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_Nodes:
    if node.features.special_feature == 'C':
        set_right_num.append(node.num)
        set_right_nodes.append(node)
    else:
        set_left_num.append(node.num)
        set_left_nodes.append(node)

list_Edges = []
set_edges = []
for node_r in set_right_nodes:
    for node_l in set_left_nodes:
        if (np.random.random() < 0.2):
            list_Edges.append(Edges(node_r, node_l))
            set_edges.append((node_r.num, node_l.num))

G_project.add_nodes_from(set_right_num, bipartite=1)
G_project.add_nodes_from(set_left_num, bipartite=0)
G_project.add_edges_from(set_edges)

nx.draw_networkx(G_project, labels=None)
nx.spring_layout(G_project)
plt.show()
