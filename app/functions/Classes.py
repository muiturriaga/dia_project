import numpy as np

class SocialEnvironment():
    def __init__(self,list_num_edges,  list_proba_edges, n_nodes, message,  budget, bool_track):
        # Probabilities of edges are supposed to be unknown, so we can  not use them directly. However we can estimate it with round.
        self.estimated_p = [0]*len(list_proba_edges)
        self.message = message
        self.age = 1
        self.n_nodes = n_nodes
        self.list_num_edges = list_num_edges
        self.bool_track = bool_track
        self.budget = budget

        # Then we calculate the proba matrix of our network.
        prob_matrix = np.zeros((n_nodes, n_nodes))
        index = 0
        for num_edges in list_num_edges:
            prob_matrix[num_edges[0], num_edges[1]] = list_proba_edges[index]
            index +=1
        self.p = prob_matrix


class SocialUCBLearner():
    def __init__(self, arms_features, budget):
        self.arms = arms_features
        self.dim = arms_features.shape[1]
        self.collected_rewards = []
        self.pulled_arms = []
        self.c = 2.0
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T
        self.beta = np.dot(np.linalg.inv(self.M), self.b)
        self.collected_rewards_arms = np.zeros(len(arms_features))
        self.nbr_calls_arms = np.zeros(len(arms_features))

    def compute_cbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = []
        for arm in self.arms:
            arm = np.atleast_2d(arm).T
            ucb = np.dot(self.theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb[0][0])
        return np.array(ucbs)


    def pull_super_arm(self, budget):
        ucbs  = self.compute_cbs()
       # print(ucbs)
        super_ucbs = ucbs.argsort()[-budget:][::-1]
        return super_ucbs

    def update_estimation(self, pulled_super_arm_idx, list_reward):
        i = 0
        for arm_idx in pulled_super_arm_idx:
            arm = np.atleast_2d(self.arms[arm_idx]).T # Vectors of features.
            self.M += np.dot(arm, arm.T)
            self.b += (list_reward[i]) * arm
            self.collected_rewards_arms[arm_idx] += list_reward[i]
            self.nbr_calls_arms[arm_idx] += 1
            i += 1

    def update(self, pulled_super_arm_idx, list_reward):
        self.pulled_arms.append(list(pulled_super_arm_idx))
        self.collected_rewards.append(list_reward)
        self.update_estimation(pulled_super_arm_idx, list_reward)