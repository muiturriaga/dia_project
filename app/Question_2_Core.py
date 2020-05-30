# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given. For a first algorithm, we can try to code a Greedy Algorithm.

# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are known.
# The budget is constant over experiences.
# A node can be activated one time in a episode.
# A node has multiple sharing, that means it can activate many nodes.
# Graph directed.
# Reward : A nodes activated.

import os
os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from app.functions.socialNetwork import *
from app.functions.Question_2_functions import *
from networkx.algorithms import bipartite


Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, proba_edges) # A nodes are in red.

if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)
    List_A_num = np.where(np.array(test) == 'A')[0]

## 2. Design Algorithm.
# Monte Carlo Approach. We suppose that we know the probability of activation of each node. We are going to compute the score of each node which is calculated as the average number of nodes activated which have the special_feature 'A'.

Prob_matrix = np.zeros((N_nodes, N_nodes))
for edge in Edges_info[1]:
    Prob_matrix[edge.head, edge.tail] = edge.proba_activation

Dataset = []
Message = 'A'
Dataset_activation_edges = []


def q2_credit_assigned_to_seeds():
    # Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose, we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to be spread between nodes of type A.

    for e in range(0,N_episodes):
        history, list_new_activated_edges = simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = Budget, perfect_nodes = [])
        Dataset.append(history)
        Dataset_activation_edges.append(list_new_activated_edges)

    # We count only nodes activated regardless of their message.
    credits_of_each_node, score_by_seeds_history = estimate_node_message(dataset = Dataset, dataset_edges = Dataset_activation_edges, message = Message, list_nodes = Nodes_info[2])
    means = np.nanmean(score_by_seeds_history, axis=1)

    estimated_best_nodes = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:Budget]

    print(estimated_best_nodes)
    attributes = check_attributes(estimated_best_nodes, Nodes_info[2])
    print("Best nodes obtained: ")
    for i in range(len(estimated_best_nodes)):
        print("Node {} of type {}".format(estimated_best_nodes[i],attributes[i]))

    Best_cumulative_reward = 0
    for i in estimated_best_nodes:
        Best_cumulative_reward += means[i]
    print("Best cumulative reward sum of averages: {}".format(Best_cumulative_reward))
    print("Note the upper value may be higher than the number of nodes because it's the sum of averages")
    print(np.nansum(score_by_seeds_history,axis=0))

    plt.axhline(y=Best_cumulative_reward, color="r")

    plt.plot(np.array(Best_cumulative_reward) - -np.sort(-np.nansum(score_by_seeds_history, axis=0)))
    plt.xlabel('Experiments')
    plt.ylabel('Difference of rewards')
    plt.show()

# Limitations : this approach is not relevant when the number of nodes incresases because it implies that the number of experiments increases exponentially as we want to compute the credit in many situations.

q2_credit_assigned_to_seeds()