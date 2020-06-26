import numpy as np
import sys, os

# to import bipartite
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'bipartite')))
sys.path.append(path2add)
import oracle

class Make_Bipartite:
    def __init__(self, list_of_node):
        self.list_of_node = list_of_node
        self.set_right_nodes = []
        self.set_left_nodes = []
        self.__set_nodes()
        self.prob_matrix = []
        self.p = []
        self.list_of_edges = []
        self.flag = 0

    def make_bipartite(self):
        # list optimal
        list_edges = []
        i = 0
        j = 0
        edge_num = 0
        # set_edges = []
        for node_r in self.set_right_nodes:
            i += 1
            j = 0
            for node_l in self.set_left_nodes:
                j += 1

                p = self.prob_matrix[i - 1, j - 1]
                # p=np.random.rand()
                prob_bernouli = sum(np.random.binomial(1, p, 1) == 1)
                constant = 0
                # AC = 0.4 AD = 0.2 BC = 0.3 BD = 0.1
                if node_l.node_type == 'A' and node_r.node_type == 'C':
                    constant = 0.35
                elif node_l.node_type == 'A' and node_r.node_type == 'D':
                    constant = 0.32
                elif node_l.node_type == 'B' and node_r.node_type == 'C':
                    constant = 0.28
                elif node_l.node_type == 'B' and node_r.node_type == 'D':
                    constant = 0.25
                if self.flag == 0:
                    weight = prob_bernouli * constant
                else:
                    ## error [][] instead [ , ]
                    edge_prob = self.p[i-1, j-1]
                    weight = prob_bernouli * edge_prob

                self.flag = 1
                list_edges.append(
                    oracle.Edges(Node_1=node_l, Node_2=node_r, weight_prob=weight, i=edge_num, constant=constant))
                edge_num += 1
                self.list_of_edges = list_edges
        return list_edges


    def calculate_probability(self):
        cross_over = len(self.set_right_nodes) * len(self.set_left_nodes)
        prob_vec = np.random.rand(cross_over)
        ######## HERE I should put mu_bar
        self.prob_matrix = prob_vec.reshape(len(self.set_right_nodes), len(self.set_left_nodes))

    def __set_nodes(self):
        for node in self.list_of_node:
            if node.node_type == 'C' or node.node_type == 'D':
                self.set_right_nodes.append(node)
            if node.node_type == 'A' or node.node_type == 'B':
                self.set_left_nodes.append(node)

    def set_p(self,mu_bar):
        self.p = mu_bar.reshape(len(self.set_right_nodes), len(self.set_left_nodes))

    def return_edge_constant(self):
        list_of_edges_constant = []
        for edge in self.list_of_edges:
            list_of_edges_constant.append(edge.constant)
        return list_of_edges_constant
