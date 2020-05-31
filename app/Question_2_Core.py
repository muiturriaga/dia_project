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


if bool_affichage == True :
    test = []
    for nodes in Nodes_info[2]:
        test.append(nodes.special_feature[0])
    print('Nodes of type [A,B,C] : ', test.count('A'), test.count('B'), test.count('C'))
    draw_graph(Edges_info[0], Nodes_info[2], Color_map)
    List_A_num = np.where(np.array(test) == 'A')[0]

## 2. Design Algorithm.
# Monte Carlo Approach. We suppose that we know the probability of activation of each node. We are going to compute the score of each node which is calculated as the average number of nodes activated which have the special_feature 'A'.

Dataset = []
Dataset_activation_edges = []

Prob_matrix = np.zeros((N_nodes, N_nodes))
for edge in Edges_info[1]:
    Prob_matrix[edge.head, edge.tail] = edge.proba_activation


def q2_credit_assigned_to_seeds():
    # Specify a superarm of nodes to study. We want to know if it is useful to activate this superarm. For that purpose, we have to calculate how many nodes it activates in average. And how many are A nodes. Initially our budget has to be spread between nodes of type A.

    for e in range(0,N_episodes):
        history, list_new_activated_edges = simulate_episode(init_prob_matrix = Prob_matrix, n_steps_max = 100, budget = Budget, perfect_nodes = [])
        Dataset.append(history)
        Dataset_activation_edges.append(list_new_activated_edges)

    # We count only nodes activated regardless of their message.
    credits_of_each_node, score_by_seeds_history = credits_assignement(dataset = Dataset, dataset_edges = Dataset_activation_edges, list_nodes = Nodes_info[2], track = True)
    means = np.nanmean(score_by_seeds_history, axis=1)

    i = 0
    for x in means:
        if math.isnan(x):
            means[i] = 0
        i +=1

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

# We compute the rewards for many experiences, then we sorted it. However, in fact this algorithm, does not learn. It just calculates many possibilities and return the best.
    list_reward_per_experiments = -np.sort(-np.nansum(score_by_seeds_history, axis=0))
    opt = np.round(np.array(Best_cumulative_reward),1)
    plt.plot(opt - list_reward_per_experiments)
    plt.xlabel('Experiments')
    plt.ylabel('Difference of rewards')
    plt.show()

q2_credit_assigned_to_seeds()

## Learning Process

Lin_social_ucb_rewards_per_experiment = []
List_best_super_arms_per_experiment = []
List_best_rewards_per_experiment = []
arms_features = get_list_features(Nodes_info[2], dim = 19)

List_proba_edges = []
for edge in Edges_info[1]:
    List_proba_edges.append(edge.proba_activation)

List_special_features_nodes = [ node.special_feature for node in Nodes_info[2]]
Env = SocialEnvironment(Edges_info[0], List_proba_edges, N_nodes, 'A', Budget, bool_knowledge = True, bool_track = False)

for e in range(0,N_episodes):
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