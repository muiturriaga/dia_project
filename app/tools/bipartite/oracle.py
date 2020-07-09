import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random


class Nodes:
    def __init__(self, num, node_type, features=None):
        self.num = num
        self.node_type = node_type
        self.features = features


class Edges:

    def __init__(self, Node_1, Node_2, weight_prob, i, constant):
        self.weight = weight_prob
        self.nodes = [Node_1.num, Node_2.num]
        self.constant = constant
        self.i = i


class Matching:
    def __init__(self, list_of_edges):
        self.list_of_edges = list_of_edges
        self.matched_list = []
        self.matching()

    def matching(self):
        G = nx.Graph()
        random.shuffle(self.list_of_edges)
        set_edges = []
        for edge in self.list_of_edges:
            set_edges.append([edge.nodes[0], edge.nodes[1], edge.weight])
        G.add_weighted_edges_from(set_edges)

        matched_nodes = nx.max_weight_matching(G, maxcardinality=False)
        for pair_of_nodes in matched_nodes:
            for nodes in self.list_of_edges:
                nodes_list = nodes.nodes
                if (pair_of_nodes[0] == nodes_list[0] and pair_of_nodes[1] == nodes_list[1]) or (
                        pair_of_nodes[0] == nodes_list[1] and pair_of_nodes[1] == nodes_list[
                    0]):
                    self.matched_list.append(nodes)

    def weight_list_of_matched_list(self):
        weight = []
        for edge in self.matched_list:
            weight.append(edge.weight)
        return weight

    def weight_of_matched_list(self):
        weight = 0
        for edge in self.matched_list:
            weight += edge.weight
        return weight
