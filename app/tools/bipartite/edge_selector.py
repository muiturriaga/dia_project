import itertools
import random
import numpy as np

class EdgeSelector:
    def __init__(self, list_of_edge):
        self.list_of_edge = list_of_edge

    # The CMAB problem contains a constraint S ⊆ 2[m], where 2[m] is the set of all possible subsets of arms.
    # We refer to every set of arms S∈S as a superarm of the CMAB problem
    def selectNodes(self, num_of_subset=0):
        nodes_subset = []
        # this is all the possible subsets of a given size
        if num_of_subset != 0:
            return list(itertools.combinations(self.list_of_edge, num_of_subset))

        # this is S all the possible subsets in the node
        else:
            for i in range(len(self.list_of_edge)):
                nodes_subset.append(list(map(set, itertools.combinations(self.pool_of_nodes, num_of_subset))))
                return nodes_subset
