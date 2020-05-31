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
from app.functions.Classes import *
from app.functions.Question_3_functions import *
from networkx.algorithms import bipartite



Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, proba_edges)

if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)


## 2. Design Algorithm.
# Assumptions :
# Arms are our nodes.
# A superarm is a collection of nodes given a certain budget.
# Our environment is characterized by nodes and a budget.
# We do not know the activation probabilities of edges.

Lin_social_ucb_rewards_per_experiment = []
List_best_super_arms_per_experiment = []
List_best_rewards_per_experiment = []
arms_features = get_list_features(Nodes_info[2], dim = 19)

List_proba_edges = []
for edge in Edges_info[1]:
    List_proba_edges.append(edge.proba_activation)

List_special_features_nodes = [ node.special_feature for node in Nodes_info[2]]

for e in range(0,N_episodes):
    Env = SocialEnvironment(Edges_info[0], List_proba_edges, N_nodes, 'A', Budget, bool_knowledge = False, bool_track = False)
    Lin_social_ucb_learner = SocialUCBLearner(arms_features = arms_features , budget = Budget)
    for t in range(0,T):
        Pulled_super_arm = Lin_social_ucb_learner.pull_super_arm(Budget)
        List_reward = calculate_reward(Pulled_super_arm , Env, Nodes_info[2])
        Lin_social_ucb_learner.update(Pulled_super_arm, List_reward)

        # Save the rewards of each experiments. So it is composed of n_experiments*T numbers. We are basically doing a mean on experiments in order to have the average number of rewards at each step.

    Best_super_arms_experiment = np.argsort(Lin_social_ucb_learner.nbr_calls_arms)[::-1][0:Budget]

    Lin_social_ucb_rewards_per_experiment.append(Lin_social_ucb_learner.collected_rewards)
    List_best_super_arms_per_experiment.append(Best_super_arms_experiment)
    List_best_rewards_per_experiment.append([Lin_social_ucb_learner.collected_rewards_arms[i]/(Lin_social_ucb_learner.nbr_calls_arms[i]) for i in Best_super_arms_experiment])

opt = np.mean(np.array([i for i in List_best_rewards_per_experiment]))
mean = np.mean(Lin_social_ucb_rewards_per_experiment, axis= 0)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt - mean), 'r')
plt.legend(["LinUCB"])
plt.show()