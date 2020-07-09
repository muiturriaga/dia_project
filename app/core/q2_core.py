import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys, os

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import social_network.sn as sn
import social_network.sn_tools as sn_tools
import q2.q2_tools as q2_tools

# MAIN PARAMETERS
n_nodes = 100
budget = 10

# create and print social network
edges_info, nodes_info, color_map = sn.create_sn(n_nodes)
sn.draw_sn(edges_info[0], nodes_info[0], color_map)

# # 2. Design Algorithm. Monte Carlo Approach. We suppose that we know the probability of activation of each node. We
# are going to compute the score of each node which is calculated as the average number of nodes activated which have
# the special_feature 'A'. Based on video 4.3

dataset = []
dataset_activation_edges = []

prob_matrix = np.zeros((n_nodes, n_nodes))
for edge in edges_info[1]:
    prob_matrix[edge.head, edge.tail] = edge.proba_activation

# Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose,
# we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to
# be spread between nodes of type A.

n_episodes = int(q2_tools.calculate_minimum_rep(0.05, 0.01, budget))

compt = 0
print('The number of episodes is : ', n_episodes)
for e in range(0, n_episodes):
    #    if compt%20 == 0:
    #        print(e)
    history, list_new_activated_edges = sn_tools.simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=100,
                                                                  budget=budget, perfect_nodes=[])
    dataset.append(history)
    dataset_activation_edges.append(list_new_activated_edges)
    compt += 1

# We count only nodes activated regardless of their message.
credits_of_each_node, score_by_seeds_history, nbr_activates = sn_tools.credits_assignment(dataset=dataset,
                                                                                          dataset_edges=dataset_activation_edges,
                                                                                          list_nodes=nodes_info[1],
                                                                                          track=True)
means = np.nanmean(score_by_seeds_history, axis=1)

i = 0
for x in means:
    if math.isnan(x):
        means[i] = 0
    i += 1

estimated_best_nodes = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:budget]

print(estimated_best_nodes)
attributes = q2_tools.check_attributes(estimated_best_nodes, nodes_info[1])
print("Best nodes obtained: ")
for i in range(len(estimated_best_nodes)):
    print("Node {} of type {} activates on average {} nodes".format(estimated_best_nodes[i], attributes[i],
                                                                    means[estimated_best_nodes[i]]))

best_cumulative_reward = 0
for i in estimated_best_nodes:
    best_cumulative_reward += means[i]
print("Best cumulative reward sum of averages: {}".format(best_cumulative_reward))
print(
    "Note the upper value may be higher than the number of nodes because it's the sum of the average of {} Nodes".format(
        len(estimated_best_nodes)))
print(np.nansum(score_by_seeds_history, axis=0))

plt.axhline(y=best_cumulative_reward, color="r")

# We compute the rewards for many experiences, then we sorted it. However, in fact this algorithm, does not learn. It
# just calculates many possibilities and return the best.
list_reward_per_experiments = -np.sort(-np.nansum(score_by_seeds_history, axis=0))
opt = np.round(np.array(best_cumulative_reward), 1)
plt.plot(opt - list_reward_per_experiments, color="g")
plt.xlabel('Experiments')
plt.ylabel('Difference of rewards')
plt.show()
