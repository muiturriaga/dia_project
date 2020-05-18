# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given. For a first algorithm, we can try to code a Greedy Algorithm.

# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are known.
# The budget is constant over experiences.

import os
os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from app.functions.socialNetwork import *
from app.functions.Question_2_functions import *
from networkx.algorithms import bipartite


# 1. Creation of the graph. Only A nodes are in red.

bool_affichage = True
N_nodes = 100
Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, 0.05)

if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)
    List_A_num = np.where(np.array(test) == 'A')[0]

## 2. Design Algorithm.
# Monte Carlo Approach. We suppose that we know the probability of activation of each node. We are going to compute the score of each node which is calculated as the average number of nodes activated which have the special_feature 'A'.

Budget = int(N_nodes/10)
N_episodes = 1000
Prob_matrix = np.zeros((N_nodes, N_nodes))
for edge in Edges_info[1]:
    Prob_matrix[edge.head, edge.tail] = edge.proba_activation
    # If we consider the graph is undirected, we have to add edges between tail and head.

Dataset = []

Message = 'A'
for e in range(0,N_episodes):
    Dataset.append(simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = Budget, perfect_nodes = [])[0])

[estimated_best_nodes, cumulative_reward] = estimate_node_message(dataset = Dataset, message = Message, list_nodes = Nodes_info[2])

estimated_best_nodes = sorted(range(len(estimated_best_nodes)), key=lambda i: estimated_best_nodes[i], reverse=True)[:Budget]

# Surprisingly, it can be useful to activate node that do not have the same special feature type as the one in the message.
attributes = check_attributes(estimated_best_nodes, Nodes_info[2])
print(estimated_best_nodes, attributes)


SuperDataset = []
for e in range(0,N_episodes):
    SuperDataset.append(simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = Budget, perfect_nodes = estimated_best_nodes)[0])
Best_cumulative_reward = estimate_node_message(dataset = SuperDataset, message = Message, list_nodes = Nodes_info[2])[1]

plt.plot( np.array(Best_cumulative_reward)- np.array(cumulative_reward))
plt.xlabel('Experiments')
plt.ylabel('Difference of rewards')
plt.show()