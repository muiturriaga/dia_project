import os
import math
os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from app.functions.socialNetwork import *
from app.functions.Classes import *
from app.functions.Question_2_functions import *
from networkx.algorithms import bipartite


Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, proba_edges) # A nodes are in red.

" We need N_nodes, proba_edges available in Tools -at the begining. And Edges_info. "
def list_best_nodes(budget, message, edges_info_1, nodes_info_2):

    if message =='A':
        budget_specific = budget[1]
    elif message == 'B':
        budget_specific = budget[2]
    elif message == 'C':
        budget_specific = budget[3]

    Dataset = []
    Dataset_activation_edges = []

    Prob_matrix = np.zeros((N_nodes, N_nodes))
    for edge in edges_info_1:
        Prob_matrix[edge.head, edge.tail] = edge.proba_activation


    # Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose, we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to be spread between nodes of type A.

    N_episodes = int(calculate_minimum_rep(0.05,0.01,budget_specific))

    compt = 0
    print('The number of episodes is : ' , N_episodes)
    for e in range(0,10):
        history, list_new_activated_edges = simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = budget_specific, perfect_nodes = [])
        Dataset.append(history)
        Dataset_activation_edges.append(list_new_activated_edges)
        compt +=1

    # We count only nodes activated regardless of their message.
    credits_of_each_node, score_by_seeds_history = credits_assignement(dataset = Dataset, dataset_edges = Dataset_activation_edges, list_nodes = nodes_info_2, track = True)
    means = np.nanmean(score_by_seeds_history, axis=1)

    i = 0
    for x in means:
        if math.isnan(x):
            means[i] = 0
        i +=1

    estimated_best_nodes = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:budget_specific]

    # print(estimated_best_nodes)

    history_best, list_new_activated_edges_best = simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = budget_specific, perfect_nodes = estimated_best_nodes)

    # print(history_best)

    list_nodes_activated = estimated_best_nodes
    for step in list_new_activated_edges_best:
        for edges in step :
            if edges[1] not in list_nodes_activated:
                list_nodes_activated.append(edges[1])

    list_nodes_activated = sorted(list_nodes_activated)

    attributes = check_attributes(list_nodes_activated, nodes_info_2)

    # print(list_nodes_activated)
    # print(attributes)

    index_good_message = np.where(np.array(attributes) == message)[0]

    list_nodes_good_message = []

    for index in index_good_message:
        list_nodes_good_message.append(list_nodes_activated[index])

    return list_nodes_good_message