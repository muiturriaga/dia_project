# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given. However now, we do not know the activation probabilities. We can only observe the activation of edges.

# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are not known but activation of edges are known.
# The budget is constant over experiences.

import os
os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from app.functions.socialNetwork import *
from app.functions.Question_3_functions import *
from networkx.algorithms import bipartite


bool_affichage = True
N_nodes = 100
Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, 0.02)

if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)


## 2. Design Algorithm.
# Arms are our nodes. A superarm is a collection of nodes given a certain budget. Our environment is characterized by nodes and a budget. We do not know the activation probabilities of edges.

T = 10
n_experiments = 1000
Lin_social_ucb_rewards_per_experiment = []
Budget = int(N_nodes/10)
arms_features = get_list_features(Nodes_info[2], dim = 19)

List_proba_edges = []
for edge in Edges_info[1]:
    List_proba_edges.append(edge.proba_activation)

List_special_features_nodes = [ node.special_feature for node in Nodes_info[2]]
Env = SocialEnvironment(Edges_info[0], List_proba_edges, N_nodes, 'A', Budget)

for e in range(0,n_experiments):
    Lin_social_ucb_learner = SocialUCBLearner(arms_features = arms_features , budget = Budget)
    for t in range(0,T):
        Pulled_super_arm = Lin_social_ucb_learner.pull_super_arm(Budget)
        List_reward = calculate_reward(Pulled_super_arm, List_special_features_nodes ,Env)
        Lin_social_ucb_learner.update(Pulled_super_arm, List_reward)
        Lin_social_ucb_rewards_per_experiment.append(Lin_social_ucb_learner.collected_rewards)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(Lin_social_ucb_rewards_per_experiment, axis = 0)), 'r')
plt.legend(["LinUCB"])
plt.show()