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
# TO CALCULATE ERROR
percents = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]

# create and print social network
edges_info, nodes_info, color_map = sn.create_sn(n_nodes)
sn.draw_sn(edges_info[0], nodes_info[0], color_map)

# # 2. Design Algorithm. Monte Carlo Approach. We suppose that we know the probability of activation of each node. We
# are going to compute the score of each node which is calculated as the average number of nodes activated which have
# the special_feature 'A'. Based on video 4.3

dataset = []
dataset_activation_edges = []
dataset_clairvoyant = []
dataset_activation_edges_clairvoyant = []

prob_matrix = np.zeros((n_nodes, n_nodes))
for edge in edges_info[1]:
    prob_matrix[edge.head, edge.tail] = edge.proba_activation

# Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose,
# we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to
# be spread between nodes of type A.

########## CALCULATE NUMBER OF EPISODES ##########
# using the Hoeffding bound
n_episodes = int(q2_tools.calculate_minimum_rep(0.05, 0.01, budget))




########## RUN SIMULATION ##########
print('The number of episodes is : ', n_episodes)
for e in range(0, n_episodes):
    history, list_new_activated_edges = sn_tools.simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=100,
                                                                  budget=budget, perfect_nodes=[])
    dataset.append(history)
    dataset_activation_edges.append(list_new_activated_edges)


# We count only nodes activated regardless of their message.
credits_of_each_node, score_by_seeds_history, nbr_activates = sn_tools.credits_assignment(dataset=dataset,
                                                                                          dataset_edges=dataset_activation_edges,
                                                                                          list_nodes=nodes_info[1],
                                                                                          track=False,
                                                                                          budget = budget)


########## GET BEST ARM AND PRINT THE NODES ##########
means = np.nanmean(score_by_seeds_history, axis=1)

means = np.nan_to_num(means)

estimated_best_nodes = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:budget]

attributes = q2_tools.check_attributes(estimated_best_nodes, nodes_info[1])
print("Best nodes obtained: ")
for i in range(len(estimated_best_nodes)):
    print("Node {} of type {} activates on average {} nodes".format(estimated_best_nodes[i], attributes[i],
                                                                    means[estimated_best_nodes[i]]))

########## BEST NUMERICAL RESULT OBTAINED SO FAR AS PSEUDO-CLAIRVOYANT ##########
migliore = np.nansum(score_by_seeds_history, axis=0)
opt = max(migliore)


best_cumulative_reward = 0
for i in estimated_best_nodes:
    best_cumulative_reward += means[i]
print("Best cumulative reward sum of averages: {}".format(best_cumulative_reward))
print(
    "Note the upper value may be higher than the number of nodes because it's the sum of the average of {} Nodes".format(
        len(estimated_best_nodes)))
print(np.nansum(score_by_seeds_history, axis=0))


# We compute the rewards for many experiences, then we sorted it. However, in fact this algorithm, does not learn. It
# just calculates many possibilities and return the best.
list_reward_per_experiments_ordered = np.nansum(score_by_seeds_history, axis=0)
list_reward_per_experiments = np.nansum(score_by_seeds_history, axis=0)
#opt = np.round(np.array(best_cumulative_reward), 1)



########## CALCULATING ERROR ##########
# changing the number of episodes for the MC SIMULATION

opts = []
epis = []
for p in percents:
    dataset = []
    dataset_activation_edges = []
    for e in range(0, round(n_episodes*p)):
        history, list_new_activated_edges = sn_tools.simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=100,
                                                                      budget=budget, perfect_nodes=[])
        dataset.append(history)
        dataset_activation_edges.append(list_new_activated_edges)


    # We count only nodes activated regardless of their message.
    credits_of_each_node, score_by_seeds_history, nbr_activates = sn_tools.credits_assignment(dataset=dataset,
                                                                                              dataset_edges=dataset_activation_edges,
                                                                                              list_nodes=nodes_info[1],
                                                                                              track=False,
                                                                                              budget = budget)

    migliore = np.nansum(score_by_seeds_history, axis=0)
    opt = max(migliore)
    opts.append(opt)
    epis.append(p*n_episodes)


########## PLOTTING RESULTS ##########

plt.figure(1)
plt.axhline(y=opt, color="b")
plt.plot(opt - list_reward_per_experiments_ordered, color="g")
#   plt.title('Episodes {}, Nodes {}, Budget {}'.format(episodes, nodes, budget))
plt.xlabel('Episodes')
plt.ylabel('Difference of rewards')

plt.figure(2)
plt.plot(np.cumsum(opt - list_reward_per_experiments), color="g")
plt.xlabel('Episodes')
plt.ylabel('Regret')

plt.figure(3)
plt.title('Error')
plt.plot(epis, opts, color="g")
plt.xlabel('Number of Episodes')
plt.ylabel('Best result obtained')

plt.show()
