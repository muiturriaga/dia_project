# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given. However now, we do not know the activation probabilities. We can only observe the activation of edges.

# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are not known but activation of edges are known.
# The budget is constant over experiences.

#import os
#os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from functions.socialNetwork import *
from functions.Classes import *
from functions.Question_3_functions import *
from functions.Tools import *
from networkx.algorithms import bipartite



Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], N_nodes, proba_edges)

if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)


## 2. Design Algorithm.
## Learning Process

Lin_social_ucb_rewards_per_experiment = []
List_best_super_arms_per_experiment = []
List_best_rewards_per_experiment = []
arms_features = get_list_features(Nodes_info[2], dim = 19)

List_proba_edges = []
for edge in Edges_info[1]:
    List_proba_edges.append(edge.proba_activation)

List_special_features_nodes = [node.special_feature for node in Nodes_info[2]]
Env = SocialEnvironment(Edges_info[0], List_proba_edges, N_nodes, 'A', Budget, bool_track = False)

np.seterr(over='raise')
print('The number of episodes is : ' , N_episodes)
for e in range(0,N_episodes):#N_episodes
    print("Running episode n {}".format(e))

    Lin_social_ucb_learner = SocialUCBLearner(arms_features = arms_features , budget = Budget)

    for t in range(0,T):
        Pulled_super_arm = Lin_social_ucb_learner.pull_super_arm() #idx of pulled arms
        List_reward = calculate_reward(Pulled_super_arm, Env, Nodes_info[2]) #rewards of just the nodes pulled
        Lin_social_ucb_learner.update(Pulled_super_arm, List_reward)

    # Save the rewards of each experiments. So it is composed of n_experiments*T numbers. We are basically doing a mean on experiments in order to have the average number of rewards at each step.
    # "If" added in construction to avoid dividing by zero (if an arm has not been pulled)
    List_average_reward = [round(Lin_social_ucb_learner.collected_rewards_arms[i]/(Lin_social_ucb_learner.nbr_calls_arms[i]),5) if Lin_social_ucb_learner.nbr_calls_arms[i] != 0 else 0 for i in range(0,N_nodes)]

    np.nan_to_num(List_average_reward) #transform np.NaN values to zeros
#    for i in range(len(List_average_reward)):
#        if math.isnan(List_average_reward[i]) == True:
#            List_average_reward[i] = 0

    Best_super_arms_experiment = np.argsort(List_average_reward)[::-1][0:Budget]

    if Env.bool_track == False:
        Lin_social_ucb_rewards_per_experiment.append([ [Lin_social_ucb_learner.collected_rewards[i][0]] for i in range(len(Lin_social_ucb_learner.collected_rewards))])
    else :
        Lin_social_ucb_rewards_per_experiment.append([ [np.mean(Lin_social_ucb_learner.collected_rewards[i])] for i in range(len(Lin_social_ucb_learner.collected_rewards))])

    List_best_super_arms_per_experiment.append(Best_super_arms_experiment) #append a list with the IDs of best nodes in this experiment
    List_best_rewards_per_experiment.append([List_average_reward[i] for i in Best_super_arms_experiment])


opt = np.mean(np.array([np.mean(np.array(i)) for i in List_best_rewards_per_experiment]), axis = 0)
mean = np.mean(Lin_social_ucb_rewards_per_experiment, axis= 0)
print("opt {} mean {} bool {}".format(opt,mean,Env.bool_track))

plt.figure(0)
plt.title("nodes: {}, time: {}, n_episodes: {}, bool_track: {}".format(N_nodes, T, N_episodes, Env.bool_track))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt - mean ), 'r')
plt.legend(["LinUCB"])
plt.show()
