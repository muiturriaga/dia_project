from app.functions.Tools import *

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

def calculate_reward(pulled_super_arm, list_special_features_nodes, env):
    list_rewards_super_arm = [0] * len(pulled_super_arm)
    env.update_proba(pulled_super_arm) # We update the probability of edges.

    prob_matrix = np.zeros((env.n_nodes, env.n_nodes)) # Then we calculate the proba matrix.
    index = 0
    for num_edges in env.list_num_edges:
        Prob_matrix[num_edges[0], num_edges[1]] = env.estimated_p[index]/(env.age)
        index +=1

    episode = simulate_episode(init_prob_matrix = prob_matrix, n_steps_max = 100, budget = env.budget, perfect_nodes =  []) # Simulate an episode.
    credit = 0
    list_A_node_activated = []
    for index_steps in range(len(episode)): # What are the nodes activated at each step ?
        idx_w_active = np.argwhere(episode[index_steps] == 1).reshape(-1)
        for num_node in idx_w_active: # If they are of type 'message', we give credits to all of them. We can criticize this point. A better option is to find what is the root which has activated an A node.
            if list_special_features_nodes[num_node] == env.message:
                list_A_node_activated.append(num_node)
                credit += 1
    return list_A_node_activated, episode
    # Upgrade credits of initial nodes.
    for i  in range(env.budget):
        list_rewards_super_arm[i] += credit
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
        for num_node in pulled_super_arm: # pulled_super_arm = [13,12,4,9 ... ]
            index = 0
            for edges_num in self.list_num_edges: # list_num_edges = [ (1,3) , (3,4) ...]
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
        self.pulled_super_arms = []
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

    def pull_super_arm(self, budget): #We change pull_arm in pull_super_arm.
        ucbs  = self.compute_cbs()
        super_ucbs = ucbs.argsort()[-budget:][::-1]
        return super_ucbs

    def update_estimation(self, pulled_arm_idx, reward):
        for arm_idx in pulled_arm_idx:
            arm = np.atleast_2d(self.arms[arm_idx]).T
            self.M += np.dot(arm, arm.T)
            self.b += reward * arm

    def update(self, pulled_arm_idx, reward):
        self.pulled_super_arms.append(pulled_arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(pulled_arm_idx, reward)