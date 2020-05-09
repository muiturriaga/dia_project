import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from NodeSelection import NodeSelector

list_types = ['A', 'B', 'C', 'D']
N_nodes = 160
color_map = [''] * N_nodes


class Nodes():
    def __init__(self, num, node_type):  # better to replace Special_f's argument postion with feeathre's postion
        self.num = num
        self.node_type = node_type


class Edges():

    def __init__(self, Node_1, Node_2, weight_prob):
        constant = 1
        # AC = 0.9 AD = 0.4 BC = 0.6 BD = 0.2
        if Node_1.node_type == 'A' and Node_2.node_type == 'C':
            constant = 0.9
        elif Node_1.node_type == 'A' and Node_2.node_type == 'D':
            constant = 0.4
        elif Node_1.node_type == 'B' and Node_2.node_type == 'C':
            constant = 0.6
        elif Node_1.node_type == 'B' and Node_2.node_type == 'D':
            constant = 0.2

        # Features of the edge.
        self.weight = weight_prob * constant
        self.nodes = [Node_1.num, Node_2.num]

class Matching:
    def __init__(self, list_of_nodes):
        self.list_of_nodes = list_of_nodes
        self.list_of_edges = []

    def matching(self):
        G = nx.Graph()
        set_right_num = []
        set_right_nodes = []
        set_left_num = []
        set_left_nodes = []
        for node in self.list_of_nodes:
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
#list optimal
        list_edges = []
        set_edges = []
        for node_r in set_right_nodes:
            for node_l in set_left_nodes:
                # Here is the point, what is the adequate probability for matching ?
                if np.random.random() <= 1:
                    # 20,000 trials, and count the number that generate zero positive results.(bernoulli distr)
                    weight_prob = sum(np.random.binomial(9, 0.1, 200) == 0)/200.
                    list_edges.append(Edges(node_r, node_l, weight_prob))
                    set_edges.append((node_r.num, node_l.num, weight_prob))
    #return weight_prob
        # set_edges
        G.add_weighted_edges_from(set_edges)

        matched_nodes = nx.max_weight_matching(G, maxcardinality=False)
        matched_list = []
        for pair_of_nodes in matched_nodes:
            for nodes in list_edges:
                if list(pair_of_nodes) == nodes.nodes:
                    matched_list.append(nodes)

        B = nx.Graph()
        B.add_edges_from(list(matched_nodes))

        X, Y = bipartite.sets(G)
        pos = dict()
        pos.update((n, (1, i)) for i, n in enumerate(X))  # put nodes from X at x=1
        pos.update((n, (2, i)) for i, n in enumerate(Y))  # put nodes from Y at x=2
        nx.draw(B, pos=pos, labels=None)
        plt.show()
        weight = 0
        for edges in matched_list:
            weight += edges.weight
        return weight

    # function to set a counter for the matching
list_nodes = []
for i in range(160):
    # node_type=np.random.choice(list_types,1,p=[0.25,0.25,0.25,0.25])
    # the node types are taken from pool of nodes and the activated nodes. Here I assumed some fixed nodes.
    if i <= 30:
        node_type = 'A'
    if 30 < i <= 70:
        node_type = 'B'
    if 70 < i <= 110:
        node_type = 'C'
    if i > 110:
        node_type = 'D'

    list_nodes.append(Nodes(num=i, node_type=node_type))

Matching(list_nodes).matching()