import numpy as np
import sys, os
import random

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q5.q5_tools as q5_tools
import social_network.sn as sn
import bipartite.oracle as matching
import bipartite.make_bipartite as bip

# MAIN PARAMETERS
cum_budget = 10
n_nodes = 100
num_nodes_d = 30

# create social network and list of D nodes
edges_info, nodes_info, color_map = sn.create_sn(n_nodes)
nodes_d = q5_tools.create_d_nodes(num_nodes_d, n_nodes)
# sn.draw_sn(edges_info[0], nodes_info[0], color_map)

# discretization
cum_budget_base = 10
scale = cum_budget/cum_budget_base
discretized_vector = (scale*np.array([0, 1, 3, 5, 8, 10])).astype(int)  # scale the discretized base vector accordingly

max_value = 0
matching_value = 0
best_budget = []

# obtain a vector of all possible budget allocation
budget_alloc_vect = q5_tools.budget_allocation(discretized_vector, cum_budget)
print("budget alloc vector: ", budget_alloc_vect)

seeds_A = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message = "A")
print("seeds a: ", seeds_A)
seeds_B = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message = "B")
print("seeds b: ", seeds_B)
seeds_C = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message = "C")
print("seeds c: ", seeds_C)

# main cycle, enumeration
for budget_alloc in budget_alloc_vect:  # for every possible budget allocation

    starting_nodes_A = random.sample(seeds_A, budget_alloc[0])
    starting_nodes_B = random.sample(seeds_B, budget_alloc[1])
    starting_nodes_C = random.sample(seeds_C, budget_alloc[2])
    activated_nodes = []
    print("Starting nodes ", starting_nodes_A, starting_nodes_B, starting_nodes_C)

    activated_nodes = []  # will contain all activated nodes

    # observe, given seeds for each type, the activation of nodes in the SN by spreading the message
    ret = q5_tools.list_activated_nodes(starting_nodes_A, budget_alloc[0], 'A', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)
    ret = q5_tools.list_activated_nodes(starting_nodes_B, budget_alloc[1], 'B', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)
    ret = q5_tools.list_activated_nodes(starting_nodes_C, budget_alloc[2], 'C', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)

    print(activated_nodes)
    activated_nodes = q5_tools.convert_nodes(activated_nodes)  # convert from SN node class to Oracle node class
    activated_nodes.extend(nodes_d)  # add nodes D

    # compute matching value for current activated nodes
    bip_graph_edges = bip.Make_Bipartite(activated_nodes)
    bip_graph_edges.calculate_probability()
    bip_graph_edges.make_bipartite()
    matching_value = matching.Matching(bip_graph_edges).weight_of_matched_list()

    if max_value < matching_value:  # select best budget
        best_budget = budget_alloc

print("Best budget is: ", best_budget)
