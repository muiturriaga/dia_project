import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .socialNetwork import Node, Edge
from networkx.algorithms import bipartite
import random

#Create bipartite graph
def create_graph(message_left, message_right, n_nodes, threshold):
    color_map = [''] * n_nodes

    # Initialisation of nodes.
    list_Nodes = []
    for i in range(n_nodes):
        list_Nodes.append(Node(i))

    # Assignments of features.
    set_right_num = []
    set_right_nodes = []
    set_left_num = []
    set_left_nodes = []
    for node in list_Nodes:
        if node.special_feature in message_right:
            set_right_num.append(node.id)
            set_right_nodes.append(node)
            color_map[node.id] = 'b'
        else:
            set_left_num.append(node.id)
            set_left_nodes.append(node)
            color_map[node.id] = 'r'

    list_Edges = []
    set_edges = []
    set_nodes = set_right_nodes + set_left_nodes
    set_nodes_num = [i.id for i in set_nodes]
    for node1 in set_nodes:
        for node2 in set_nodes:
            # Create an edge iif a number generated is inferior to our threshold.
            if np.random.random() < threshold and node1.id != node2.id:
                list_Edges.append(Edge(node1, node2))
                set_edges.append((node1.id, node2.id))

    edges_info = [set_edges, list_Edges]
    nodes_info = [[set_right_num, set_right_nodes], [set_left_num, set_left_nodes], list_Nodes]
    return edges_info, nodes_info, color_map


def draw_graph(edges_info, nodes_info, color_map):
    G_project = nx.Graph()
    for node in nodes_info:
        G_project.add_node(node.id)

    G_project.add_edges_from(edges_info)

    nx.draw_networkx(G_project, labels=None, node_color=color_map)
    nx.spring_layout(G_project)
    plt.show()


