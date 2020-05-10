import numpy as np
from NodeSelection import *
from matching import *
def calculate_reward(pulled_arm):
    matching = Matching(pulled_arm)
    return matching.matching()


class CMAB_Environment:
    def __init__(self,list_of_nodes):
        self.subset_of_nodes = subset_of_nodes
        node_selector = NodeSelector(list_nodes)
        super_arm_to_pull = node_selector.selectNodes()
    # [A,A,A][A,B,C][A,D,C]

    def round(self,pulled_super_arm):
        reward=[]
        # super_arm_to_pull[pulled_super_arm]
        # should do the matching multiple times then get the mean
        # matching -> sum of weights



    def calculate_reward(self, arm):
        set_of_matching_node = self.super_arm_to_pull[arm]
        return Matching(set_of_matching_node).matching()