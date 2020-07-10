import numpy as np
import sys, os
import random

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q5.q5_tools as q5_tools
import social_network.sn as sn
import q5.sn_environment as sn_env
import q5.gpts_learner as learner
import bipartite.oracle as matching
import bipartite.make_bipartite as bip

# MAIN PARAMETERS
cum_budget = 10
n_nodes = 100
num_nodes_d = 30
T = 100  # iterations for ts

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

# compute elite nodes with max budget, from which we will choose starting nodes for observing influence spreading
seeds_a = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message="A")
print("seeds a: ", seeds_a)
seeds_b = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message="B")
print("seeds b: ", seeds_b)
seeds_c = q5_tools.select_seeds(cum_budget, n_nodes, edges_info[1], nodes_info[1], message="C")
print("seeds c: ", seeds_c)

# create social network learning environment
env = sn_env.SnEnvironment(seeds_a, seeds_b, seeds_c, nodes_d, nodes_info, edges_info)
# create gaussian process thompson sampling learner
gpts_learner = learner.GptsLearner(n_arms=len(budget_alloc_vect))

# main loop, each step, pull arm, compute reward, fit gp regression
for t in range(0, T):
    pulled_arm = gpts_learner.pull_arm()  # returns index of arm pulled
    print("arm pulled: ", pulled_arm)
    reward = env.round(budget_alloc_vect[pulled_arm])   # reward = value of matching
    print("reward: ", reward)
    gpts_learner.update(pulled_arm, reward)     # update the model
    print("means vector: ", gpts_learner.means)

    if (t % (T/5)) == 0:
        q5_tools.print_gp(means=gpts_learner.means, sigmas=gpts_learner.sigmas, x_len=len(budget_alloc_vect))

q5_tools.print_gp(means=gpts_learner.means, sigmas=gpts_learner.sigmas, x_len=len(budget_alloc_vect))

max_budget_idx = np.argmax(gpts_learner.means)

print("Best budget is: ", budget_alloc_vect[max_budget_idx], "with value: ", gpts_learner.means[max_budget_idx])

# for comparison, we are doing here a naive version
max_value = 0
best_budget = []

for budget_alloc in budget_alloc_vect:
    starting_nodes_a = random.sample(seeds_a, budget_alloc[0])
    starting_nodes_b = random.sample(seeds_b, budget_alloc[1])
    starting_nodes_c = random.sample(seeds_c, budget_alloc[2])
    # print("Starting nodes ", starting_nodes_a, starting_nodes_b, starting_nodes_c)

    activated_nodes = []  # will contain all activated nodes

    # observe, given seeds for each type, the activation of nodes in the SN by spreading the message
    ret = q5_tools.list_activated_nodes(starting_nodes_a, budget_alloc[0], 'A', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)
    ret = q5_tools.list_activated_nodes(starting_nodes_b, budget_alloc[1], 'B', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)
    ret = q5_tools.list_activated_nodes(starting_nodes_c, budget_alloc[2], 'C', edges_info[1], nodes_info[1])
    if ret: activated_nodes.extend(ret)

    # print("activated nodes: ", activated_nodes)
    activated_nodes = q5_tools.convert_nodes(activated_nodes)  # convert from SN node class to Oracle node class
    activated_nodes.extend(nodes_d)  # add nodes D

    # compute matching value for current activated nodes
    bip_graph_edges = bip.Make_Bipartite(activated_nodes)
    bip_graph_edges.make_bipartite_q5()

    value_matching = matching.Matching(bip_graph_edges.list_of_edges).weight_of_matched_list()
    if value_matching > max_value:
        max_value = value_matching
        best_budget = budget_alloc

print("Best naive budget is: ", best_budget, "with value: ", max_value.astype(int))

