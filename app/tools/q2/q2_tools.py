import numpy as np

# calculate minimum number of repetition for MC approach
def calculate_minimum_rep(eps, delta, card_seeds = int):
    if card_seeds == 1:
        return (1/(eps)**2*np.log(2)*np.log(1/delta))
    else :
        return (1/(eps)**2*np.log(card_seeds)*np.log(1/delta))

# returns a list of attribute given a list of nodes.id and object nodes
def check_attributes(list_num_nodes, list_nodes):
    n = len(list_num_nodes)
    attribute = []
    for index in range(n):
        attribute.append(list_nodes[list_num_nodes[index]].special_feature)
    return attribute
