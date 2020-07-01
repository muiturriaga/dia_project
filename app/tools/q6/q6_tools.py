import sys
import os
import numpy as np

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q3.q3_tools as q3_tools
import q3.q3_learners as learners

n_episodes = 10
T = 1000
bool_track = False


# Return best seeds given the budget. Activation probabilities are estimated with UCB approach
def select_seeds(budget, n_nodes, edges_info, nodes_info, message):
    lin_social_ucb_rewards_per_experiment = []
    list_best_super_arms_per_experiment = []
    list_best_rewards_per_experiment = []
    arms_features = q3_tools.get_list_features(nodes_info[1], dim=19)

    list_proba_edges = []
    for edge in edges_info[1]:
        list_proba_edges.append(edge.proba_activation)

    list_special_features_nodes = [node.special_feature for node in nodes_info[1]]
    env = learners.SocialEnvironment(edges_info[0], list_proba_edges, n_nodes, message, budget, bool_track)

    np.seterr(over='raise')
    lin_social_ucb_learner = learners.SocialUCBLearner(arms_features=arms_features, budget=budget)

    for t in range(0, T):
        pulled_super_arm = lin_social_ucb_learner.pull_super_arm()  # idx of pulled arms
        list_reward = q3_tools.calculate_reward(pulled_super_arm, env,
                                                nodes_info[1])  # rewards of just the nodes pulled
        lin_social_ucb_learner.update(pulled_super_arm, list_reward)

    # Save the rewards of each experiments. So it is composed of n_experiments*T numbers. We are basically doing a mean on experiments in order to have the average number of rewards at each step.
    # "If" added in construction to avoid dividing by zero (if an arm has not been pulled)
    list_average_reward = [
        round(lin_social_ucb_learner.collected_rewards_arms[i] / (lin_social_ucb_learner.nbr_calls_arms[i]), 5) if
        lin_social_ucb_learner.nbr_calls_arms[i] != 0 else 0 for i in range(0, n_nodes)]

    np.nan_to_num(list_average_reward)  # transform np.NaN values to zeros
    #    for i in range(len(List_average_reward)):
    #        if math.isnan(List_average_reward[i]) == True:
    #            List_average_reward[i] = 0

    return np.argsort(list_average_reward)[::-1][0:budget]


