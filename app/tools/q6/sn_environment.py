import sys
import os
import random
import numpy as np

# import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q5.q5_tools as q5_tools
import q6.q6_tools as q6_tools


class SnEnvironment:
    def __init__(self, seeds_a, seeds_b, seeds_c, nodes_d, nodes_info, edges_info):
        self.seeds_A = seeds_a
        self.seeds_B = seeds_b
        self.seeds_C = seeds_c
        self.nodes_d = nodes_d
        self.nodes_info = nodes_info
        self.edges_info = edges_info

    # HAS TO RETURN VALUE OF MATCHING.
    # The problem I see -> at the moment, you create probabilities to estimate.
    # Substitute the part of creation of probs with features similarities
    def round(self, budget_alloc):
        starting_nodes_a = random.sample(self.seeds_A, budget_alloc[0])
        starting_nodes_b = random.sample(self.seeds_B, budget_alloc[1])
        starting_nodes_c = random.sample(self.seeds_C, budget_alloc[2])
        # print("Starting nodes ", starting_nodes_A, starting_nodes_B, starting_nodes_C)

        activated_nodes = []  # will contain all activated nodes

        # observe, given seeds for each type, the activation of nodes in the SN by spreading the message
        ret = q5_tools.list_activated_nodes(starting_nodes_a, budget_alloc[0], 'A', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)
        ret = q5_tools.list_activated_nodes(starting_nodes_b, budget_alloc[1], 'B', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)
        ret = q5_tools.list_activated_nodes(starting_nodes_c, budget_alloc[2], 'C', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)

        activated_nodes = q5_tools.convert_nodes(activated_nodes)  # convert from SN node class to Oracle node class
        activated_nodes.extend(self.nodes_d)  # add nodes D
        return self.matching_value(activated_nodes)

    def matching_value(self, activated_nodes):
        matching_value = q6_tools.estimating_weight(activated_nodes, 20, 20)
        return matching_value
        # COPY CODE HERE
        # activated nodes is a list of oracle nodes
        # q4_core -> instead of bernoulli(p) at the start -> have bernoulli(similarity_measure)
        #            in the end, with estimated prob, compute value of matching

