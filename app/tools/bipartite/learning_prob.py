import numpy as np
from copy import copy


def simulate_episode(init_prob_matrix, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.random.binomial(1, 0.1, size=n_nodes)
    history = np.array([initial_active_nodes])# data set that we want to exploit to estimate probs
    active_nodes = initial_active_nodes # store all active nodes in our episode
    newly_active_nodes = active_nodes# store that activaeted in this time step
    t = 0

    while t < n_steps_max and np.sum(newly_active_nodes) > 0: # we terminate either when we reach the numbr of states, or once we do not have nodes remained to be activated
        p = (prob_matrix.T * active_nodes).T # set the value of the variable p that select from  the probability P matrix only the rows related
        # to the active nodes
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])# sampling a value from random distribution from zero to one and comparing it to
        # the probability of each edge.
        prob_matrix=prob_matrix*((p!=0)==activated_edges)# remove from prob matrix all the values of the probabilities related to the previously activated nodes
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1-active_nodes)# compute  the value of the newly activated nodes by mean of the
        # sum and we sum the values of the activated edges matrix in zero axis
        active_nodes = np.array(active_nodes + newly_active_nodes) # update the value of the active nodes by adding the newly active nodes.
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history


def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob = np.ones(n_nodes) * 1.0 / (n_nodes - 1)
    credits = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)# in which row (time step)the target node has been activated.
        ###maybe we dont need this as the number of rows are 1.
        if len(idx_w_active) > 0 and idx_w_active > 0:
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
            credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)# assign the credits uniformally to all the nodes active in previous step
        for v in range(0, n_nodes):# then we iterate their occurance on each nodes in the each episod
            if (v != node_index):# if the node v is different than target node
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_v_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
                    occurr_v_active[v] += 1
    estimated_prob = credits / occurr_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob


n_nodes = 5
n_episodes = 100
prob_matrix = np.random.uniform(0.0, 0.1, (n_nodes, n_nodes))
node_index = 4
dataset = []

for e in range(0, n_episodes):
    dataset.append(simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=10))


estimated_prob = estimate_probabilities(dataset=dataset, node_index=node_index, n_nodes=n_nodes)

print("True P Matrix: ", prob_matrix[:, 4])
print("Estimated P Matrix: ", estimated_prob)
