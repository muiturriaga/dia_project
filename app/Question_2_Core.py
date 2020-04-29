# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given. For a first algorithm, we can try to code a Greedy Algorithm.
# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are known.
# The budget is constant over experiences.

#import os
#os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master/dia_project-master/dia/venv")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from functions.Tools import *
from networkx.algorithms import bipartite
bool_affichage = False


# 1. Creation of the graph. Only A nodes are in red.
N_nodes = 100
Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B', 'C'], N_nodes, 0.5)

if bool_affichage == True:
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print(test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info, Nodes_info, Color_map)


# 2. Design Algorithm.
# Greedy Approach. We suppose that we know the probability of activation of each node. We are going to compute the score of each node which is calculated as the average number of nodes activated which have the special_feature 'A'.
Budget = int(N_nodes/10)
N_episodes = 1000
Prob_matrix = np.zeros((N_nodes, N_nodes))
for edge in Edges_info[1]:
    Prob_matrix[edge.head.id, edge.tail.id] = edge.proba_activation
Dataset = []

# Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose, we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to be spread between nodes of type A.
Message = 'A'
for e in range(0, N_episodes):
    Dataset.append(simulate_episode(init_prob_matrix=Prob_matrix, n_steps_max=100, budget=Budget))

estimated_best_nodes = estimate_node_message(
    dataset=Dataset, message=Message, list_nodes=Nodes_info[2])

estimated_best_nodes = sorted(range(len(estimated_best_nodes)),
                              key=lambda i: estimated_best_nodes[i], reverse=True)[:Budget]

# Surprisingly, it can be useful to activate node that do not have the same special feature type as the one in the message.
attributes = check_attributes(estimated_best_nodes, Nodes_info[2])
print(estimated_best_nodes, attributes)
