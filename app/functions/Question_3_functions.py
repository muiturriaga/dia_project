from Tools import *
import numpy as np

def get_list_features(list_of_nodes, dim):
    list_features = [[0]*dim]*len(list_of_nodes)
    i = 0
    for nodes in list_of_nodes:
        list_gender = (nodes.features['gender'])
        list_age = (nodes.features['age'])
        list_interests = (nodes.features['interests'])
        list_location = (nodes.features['location'])
        list_features[i]= list_gender+list_age + list_interests + list_location
        i +=1
    return np.array(list_features)

def get_edges_cascade(pull_node, list_edges_activated):
    list_new_nodes = [pull_node]
    list_cascade_edges = []
    while(len(list_new_nodes) != 0):
        tamp = []
        for node_head in list_new_nodes:
            for edge in list_edges_activated:
                if node_head == edge[0]:
                    list_cascade_edges.append(edge)
                    tamp.append(edge[1])
        list_new_nodes = tamp
    return list_cascade_edges



def credit_nodes(list_A_node_activated, list_edges_activated, pulled_super_arm):
    list_credit = [0]*len(pulled_super_arm)
    for i in range(len(pulled_super_arm)):
        pull_node = pulled_super_arm[i]

        if pull_node in list_A_node_activated:
            list_credit[i] += 1

        tamp = get_edges_cascade(pull_node, list_edges_activated)
        if len(tamp) != 0:
            list_cascade_node = [num[1] for num in tamp] # get the inherited nodes from the cascade
            for node in list_cascade_node:
                if node in list_A_node_activated:
                    list_credit[i] +=1
    return list_credit



def calculate_reward(pulled_super_arm, list_special_features_nodes, env):
    list_rewards_super_arm = [0] * len(pulled_super_arm)
    env.update_proba(pulled_super_arm)

    prob_matrix = np.zeros((env.n_nodes, env.n_nodes))

    index = 0
    for num_edges in env.list_num_edges:
        prob_matrix[num_edges[0], num_edges[1]] = env.estimated_p[index]/(env.age)
        index +=1

    [episode , list_edges_activated]= simulate_episode(init_prob_matrix = prob_matrix, n_steps_max = 100, budget = env.budget, perfect_nodes =  pulled_super_arm) # Simulate an episode.

    credit = 0
    list_A_node_activated = []

    for index_steps in range(len(episode)): # What are the nodes activated at each step ?
        idx_w_active = np.argwhere(episode[index_steps] == 1).reshape(-1)
        for num_node in idx_w_active:
            if list_special_features_nodes[num_node] == env.message:
                list_A_node_activated.append(num_node)

    # print('I pull the super_arm : \n', pulled_super_arm)
    # print('\n', ' Edges activated in my episode are : \n', list_edges_activated)
    # print('\n', ' Nodes A activated in my episode are : \n', list_A_node_activated)
    # print('\n', check_attributes(list_A_node_activated, Nodes_info[2]))

    # We go up the tree of successive edges in order to find the node which has activated an A node. Then we give it a credit.
    list_rewards_super_arm = credit_nodes(list_A_node_activated, list_edges_activated, pulled_super_arm)
    return list_rewards_super_arm


class SocialEnvironment():
    def __init__(self,list_num_edges,  list_proba_edges, n_nodes, message,  budget):
        self.p = list_proba_edges # Probabilities of edges are supposed to be unknown, so we can  not use them directly. However we can estimate it with round.
        self.estimated_p = [0]*len(list_proba_edges)
        self.message = message
        self.age = 1
        self.n_nodes = n_nodes
        self.list_num_edges = list_num_edges
        self.budget = budget

    def update_proba(self, pulled_super_arm):
        for num_node in pulled_super_arm:
            index = 0
            for edges_num in self.list_num_edges:
                if num_node == edges_num[0]:
                    if np.random.random() < self.p[index]:
                        self.estimated_p[index] += 1
                index +=1
        self.age +=1


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
        super_ucbs = ucbs.argsort()[-budget:][::-1]
        return super_ucbs

    def update_estimation(self, pulled_super_arm_idx, list_reward):
        i = 0
        for arm_idx in pulled_super_arm_idx:
            arm = np.atleast_2d(self.arms[arm_idx]).T
            self.M += np.dot(arm, arm.T)
            self.b += list_reward[i] * arm
            i += 1

    def update(self, pulled_super_arm_idx, list_reward):
        self.pulled_arms.append(pulled_super_arm_idx)
        self.collected_rewards.append(list_reward)
        self.update_estimation(pulled_super_arm_idx, list_reward)
