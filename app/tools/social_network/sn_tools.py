import numpy as np
import random

# Simulate an information propagation episode over the social network
def simulate_episode(init_prob_matrix, n_steps_max, budget, perfect_nodes):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]

    if len(perfect_nodes) == 0:
        initial_active_nodes = np.zeros(n_nodes)
        for i in range(budget):
            initial_active_nodes[i] = 1
        random.shuffle(initial_active_nodes)

    else:
        initial_active_nodes = np.zeros(n_nodes)
        for i in perfect_nodes:
            initial_active_nodes[i] = 1

    # print('Initial active nodes are : ' , initial_active_nodes.nonzero()[0], '\n')

    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes

    t = 0
    list_new_activated_edges = []
    while(t < n_steps_max and np.sum(newly_active_nodes) > 0):
        # Extract only rows where nodes are activated.
        p = (prob_matrix.T * active_nodes).T
        list_new_activated_edges_inside = []
        # Draw random numbers and catch if we can activate them or not. If M_ij is True it means node i activate node j.
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        # If edges were not activated, they disappear.
        prob_matrix = prob_matrix * ((p != 0) == activated_edges)
        # Return nodes where edges have been activated.
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1-active_nodes)


        for i in np.argwhere(activated_edges == True):
            if (list(i) not in list_new_activated_edges_inside) & (active_nodes[i[1]] == 0):
                list_new_activated_edges_inside.append(list(i))

        list_new_activated_edges.append(list_new_activated_edges_inside)
        active_nodes = np.array(active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1

    return history, list_new_activated_edges

# Track is a boolean. If track = True then we assign rewards at a specific node by tracing the root. If track = False we give the same rewards to all the nodes.
def credits_assignment(dataset, dataset_edges, list_nodes, track = False, budget = 1):
    n_nodes = len(list_nodes)
    list_credit = np.zeros(n_nodes)
    score_by_seeds_history = np.zeros([n_nodes,len(dataset)])
    #score_by_seeds_history[:] = np.NaN

    for i in range(0,len(dataset)):
        episode = dataset[i]
        list_activated = np.zeros(n_nodes)
        history_edges = dataset_edges[i]
        list_credit_steps = np.zeros(n_nodes)

        # Catch the initial nodes
        idx_initial = np.argwhere(episode[0] == 1).reshape(-1)


        for step in history_edges[::-1]:
            # Catch new nodes activated at each step. element[0] = head and element[1] = tail. If it is a terminal node, we add 1. Else we add the number of nodes of the tail + 1 corresponding to the current node.
            for element in step:
                list_credit_steps[element[0]] += (list_credit_steps[element[1]]+1)
                list_activated[element[0]] = 1
                list_activated[element[1]] = 1


        # print(history_edges[::-1])
        # print(list_credit_steps)
        # print(list_activated)

        nbr_activates = max(np.sum(list_activated),0)
        # Upgrade credits of initial nodes.
        if track == True :
            for id in idx_initial:
                score_by_seeds_history[id][i] = list_credit_steps[id]
                list_credit[id] += list_credit_steps[id]

        # If track = False, we assign to initial nodes, the sum of activated nodes.
        elif track == False :
            for id in idx_initial:
                score_by_seeds_history[id][i] = nbr_activates/budget
                list_credit[id] = nbr_activates/budget
    #print(score_by_seeds_history)

    return list_credit, score_by_seeds_history, nbr_activates
