import sys
import os
import numpy as np

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q3.q3_tools as q3_tools
import q3.q3_learners as learners
import bipartite.cmab_env as cmab
import bipartite.learner as learner
import bipartite.make_bipartite as m_b

n_episodes = 10
T_seeds = 1000
bool_track = False


# Return best seeds given the budget. Activation probabilities are estimated with UCB approach
def select_seeds(budget, n_nodes, edges_info, nodes_info, message):

    arms_features = q3_tools.get_list_features(nodes_info[1], dim=19)

    list_proba_edges = []
    for edge in edges_info[1]:
        list_proba_edges.append(edge.proba_activation)

    list_special_features_nodes = [node.special_feature for node in nodes_info[1]]
    env = learners.SocialEnvironment(edges_info[0], list_proba_edges, n_nodes, message, budget, bool_track)

    np.seterr(over='raise')
    lin_social_ucb_learner = learners.SocialUCBLearner(arms_features=arms_features, budget=budget)

    for t in range(0, T_seeds):
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

    return np.argsort(list_average_reward)[::-1][0:budget].tolist()


def estimating_weight(list_of_nodes, T):
    prob = m_b.Make_Bipartite(list_of_nodes)
    list_of_edges = prob.make_bipartite_q5()

    env = cmab.CMAB_Environment(list_of_edges)

    cmab_learner = learner.Learner(list_of_edges, env.Ti, env.mu_bar)

    for t in range(0, T):
        cmab_learner.t_total = t + 1
        pulled_super_arm = cmab_learner.pull_super_arm()
        results = env.round(pulled_super_arm)

        cmab_learner.update(results, True)  # when do we use cmab_learner.pull_super_arm
        prob.set_p(env.mu_bar)
        list_of_edges = prob.make_bipartite_q5()
        env.list_of_all_arms = list_of_edges

        results.weight_of_matched_list()

    return env.matching_result
