# Our aim is to find the best nodes to activate in order to maximize the social influence once a budget is given.
# However now, we do not know the activation probabilities. We can only observe the activation of edges.

# Assumptions :
# All the nodes have the same budget activation
# The activation probabilities are not known but activation of edges are known.
# The budget is constant over experiences.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import sys, os

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q3.q3_tools as q3_tools
import q3.q3_learners as learners
import social_network.sn as sn

#  MAIN PARAMETER
n_nodes = 100
budget = 10
T = 1000
n_episodes = 10
bool_track = False

edges_info, nodes_info, color_map = sn.create_sn(n_nodes)
sn.draw_sn(edges_info[0], nodes_info[0], color_map)

## 2. Design Algorithm.
## Learning Process

lin_social_ucb_rewards_per_experiment = []
list_best_super_arms_per_experiment = []
list_best_rewards_per_experiment = []
arms_features = q3_tools.get_list_features(nodes_info[1], dim=19)

list_proba_edges = []
for edge in edges_info[1]:
    list_proba_edges.append(edge.proba_activation)

list_special_features_nodes = [node.special_feature for node in nodes_info[1]]
env = learners.SocialEnvironment(edges_info[0], list_proba_edges, n_nodes, 'A', budget, bool_track)

np.seterr(over='raise')
print('The number of episodes is : ', n_episodes)
for e in range(0, n_episodes):  # N_episodes
    print("Running episode n {}".format(e))

    lin_social_ucb_learner = learners.SocialUCBLearner(arms_features=arms_features, budget=budget)

    for t in range(0, T):
        pulled_super_arm = lin_social_ucb_learner.pull_super_arm()  # idx of pulled arms
        list_reward = q3_tools.calculate_reward(pulled_super_arm, env,
                                                nodes_info[1])  # rewards of just the nodes pulled
        lin_social_ucb_learner.update(pulled_super_arm, list_reward)

    # Save the rewards of each experiments. So it is composed of n_experiments*T numbers. We are basically doing a
    # mean on experiments in order to have the average number of rewards at each step. "If" added in construction to
    # avoid dividing by zero (if an arm has not been pulled)
    list_average_reward = [
        round(lin_social_ucb_learner.collected_rewards_arms[i] / (lin_social_ucb_learner.nbr_calls_arms[i]), 5) if
        lin_social_ucb_learner.nbr_calls_arms[i] != 0 else 0 for i in range(0, n_nodes)]

    np.nan_to_num(list_average_reward)  # transform np.NaN values to zeros
    #    for i in range(len(List_average_reward)):
    #        if math.isnan(List_average_reward[i]) == True:
    #            List_average_reward[i] = 0

    best_super_arms_experiment = np.argsort(list_average_reward)[::-1][0:budget]

    if not env.bool_track:
        lin_social_ucb_rewards_per_experiment.append([[lin_social_ucb_learner.collected_rewards[i][0]] for i in
                                                      range(len(lin_social_ucb_learner.collected_rewards))])
    else:
        lin_social_ucb_rewards_per_experiment.append([[np.mean(lin_social_ucb_learner.collected_rewards[i])] for i in
                                                      range(len(lin_social_ucb_learner.collected_rewards))])

    list_best_super_arms_per_experiment.append(
        best_super_arms_experiment)  # append a list with the IDs of best nodes in this experiment
    list_best_rewards_per_experiment.append([list_average_reward[i] for i in best_super_arms_experiment])

opt = np.mean(np.array([np.mean(np.array(i)) for i in list_best_rewards_per_experiment]), axis=0)
mean = np.mean(lin_social_ucb_rewards_per_experiment, axis=0)
print("opt {} mean {} bool {}".format(opt, mean, env.bool_track))

plt.figure(0)
plt.title("nodes: {}, time: {}, n_episodes: {}, bool_track: {}".format(n_nodes, T, n_episodes, env.bool_track))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt - mean), 'r')
plt.legend(["LinUCB"])
plt.show()
