import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

list_types = ['A', 'B', 'C', 'D']
N_nodes = 16
color_map = [''] * N_nodes
G = nx.Graph()


class Nodes():
    def __init__(self, num, node_type):  # better to replace Special_f's argument postion with feeathre's postion
        self.num = num
        self.node_type = node_type


class Edges():

    def __init__(self, Node_1, Node_2, weight_prob):
        constant = 1  # constant should be related to pair of nodes.type (A,C) etc
        # Features of the edge.
        self.weight = weight_prob * constant
        self.nodes = [Node_1.num, Node_2.num]


list_nodes = []
for i in range(N_nodes):
    # node_type=np.random.choice(list_types,1,p=[0.25,0.25,0.25,0.25])
    # the node types are taken from pool of nodes and the activated nodes. Here I assumed some fixed nodes.
    if i <= 3:
        node_type = 'A'
    if 3 < i <= 7:
        node_type = 'B'
    if 7 < i <= 11:
        node_type = 'C'
    if i > 11:
        node_type = 'D'

    list_nodes.append(Nodes(num=i, node_type=node_type))

set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_nodes:
    if node.node_type == 'C' or node.node_type == 'D':
        set_right_num.append(node.num)
        set_right_nodes.append(node)
        color_map[node.num] = 'b'
        G.add_node(node.num, bipartite=1)

    if node.node_type == 'A' or node.node_type == 'B':
        set_left_num.append(node.num)
        set_left_nodes.append(node)
        color_map[node.num] = 'r'
        G.add_node(node.num, bipartite=0)

list_Edges = []
set_edges = []
for node_r in set_right_nodes:
    for node_l in set_left_nodes:
        # Here is the point, what is the adequate probability for matching ?
        if (np.random.random() <= 1):
            weight_prob = np.random.random()
            list_Edges.append(Edges(node_r, node_l, weight_prob))
            set_edges.append((node_r.num, node_l.num, weight_prob))

# set_edges
G.add_weighted_edges_from(set_edges)
# for i,(u,v) in enumerate(G.edges()):
# G.edges[u, v]['weight'] = np.random.random()

# Plot
# ploting the directded like graph
nx.draw_networkx(G, labels=None, node_color=color_map)
nx.spring_layout(G)
plt.show()

# rearranging to plot it on Bipartite format
X = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
Y = set(G) - X
# X,Y = bipartite.sets(G)
pos = dict()
pos.update((n, (1, i)) for i, n in enumerate(X))  # put nodes from X at x=1
pos.update((n, (2, i)) for i, n in enumerate(Y))  # put nodes from Y at x=2
nx.draw(G, pos=pos, node_color=color_map, labels=None)
plt.show()

for edge in G.edges(data=True):
    print(edge)

    