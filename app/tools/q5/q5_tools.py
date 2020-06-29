import math
import sys, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

# to import social netowrk
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'social_network')))
sys.path.append(path2add)
import sn_tools

# to import q2_tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'q2')))
sys.path.append(path2add)
import q2_tools

# to import bipartite
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'bipartite')))
sys.path.append(path2add)
import oracle

"Return list of activated nodes from starting nodes. If starting node is empty, randomize them"
def list_activated_nodes(starting_nodes, budget_specific, message, edges_info_1, nodes_info_1):

    if budget_specific == 0:
        return []

    Prob_matrix = np.zeros((len(nodes_info_1), len(nodes_info_1)))
    for edge in edges_info_1:
        Prob_matrix[edge.head, edge.tail] = edge.proba_activation

    history, list_new_activated_edges = sn_tools.simulate_episode(init_prob_matrix=Prob_matrix,
                                                                n_steps_max=100, budget=budget_specific,
                                                                perfect_nodes=starting_nodes)
    activated_mask = history[0]
    for mask in history:
        activated_mask = np.logical_or(mask, activated_mask)

    temp = list(np.array(nodes_info_1)[activated_mask])
    activated_nodes = []

    for i in range(len(temp)):
        if temp[i].special_feature == message:
            activated_nodes.append(temp[i])

    return activated_nodes

"Return best seeds given the budget"
def select_seeds(budget, n_nodes, edges_info_1, nodes_info_1, message = False):

    dataset = []
    dataset_activation_edges = []

    prob_matrix = np.zeros((n_nodes, n_nodes))
    for edge in edges_info_1:
        prob_matrix[edge.head, edge.tail] = edge.proba_activation

    n_episodes = int(q2_tools.calculate_minimum_rep(0.05, 0.01, budget))

    compt = 0
    # print('The number of episodes is : ', n_episodes)
    for e in range(0, n_episodes):
        history, list_new_activated_edges = sn_tools.simulate_episode(init_prob_matrix = prob_matrix, n_steps_max = 100, budget = budget, perfect_nodes = [])
        dataset.append(history)
        dataset_activation_edges.append(list_new_activated_edges)
        compt += 1

    # We count only nodes activated regardless of their message.
    credits_of_each_node, score_by_seeds_history, nbr_activates = sn_tools.credits_assignment(dataset = dataset, dataset_edges = dataset_activation_edges, list_nodes = nodes_info_1, track = True)
    means = np.nanmean(score_by_seeds_history, axis=1)

    i = 0
    for x in means:
        if math.isnan(x):
            means[i] = 0
        i += 1

    sorted_means = sorted(range(len(means)), key=lambda i: means[i], reverse=True)

    if message in ("A", "B", "C"):
        if message == "A":
            estimated_best_nodes = compare_nodes("A", budget, nodes_info_1, sorted_means)
        elif message == "B":
            estimated_best_nodes = compare_nodes("B", budget, nodes_info_1, sorted_means)
        elif message == "C":
            estimated_best_nodes = compare_nodes("C", budget, nodes_info_1, sorted_means)
    else:
        estimated_best_nodes = sorted_means[:budget]

    return estimated_best_nodes


def compare_nodes(message, budget, nodes_info_1, sorted_means):
    estimated_best_nodes = []
    i=0
    while len(estimated_best_nodes) < budget:
        if nodes_info_1[sorted_means[i]].special_feature == message:
            estimated_best_nodes.append(sorted_means[i])
        if i >= len(nodes_info_1):
            return estimated_best_nodes
        i += 1

    return estimated_best_nodes

# return the vector with all possible allocation budget given a discretized vector
# skips invalid combinations (e.g. if cum_budget = 100, skips A=50 and B=70)
def budget_allocation(discretized_vector, cum_budget):

    budget_list = []
    perm_list = list(itertools.permutations(discretized_vector, 2))
    # selects only A and B that are not over cum budget
    perm_list = list(itertools.filterfalse(lambda x: x[0] + x[1] > cum_budget, perm_list))

    for tuple in perm_list:
        budget_list.append([tuple[0], tuple[1], cum_budget-(tuple[0]+tuple[1])])  # adds C = A - B

    return budget_list

# convert sn_node to oracle_node
def convert_nodes(list_sn_nodes):

    list_bip_nodes = []
    for node in list_sn_nodes:
        list_bip_nodes.append(oracle.Nodes(node.id, node.special_feature, node.features))

    return list_bip_nodes

# create a pull of D nodes for matching purpose
def create_d_nodes(num_nodes_d, n_nodes):

    list_nodes_d = []
    for i in range(num_nodes_d):
        list_nodes_d.append(oracle.Nodes(n_nodes+i, 'D'))

    return list_nodes_d
