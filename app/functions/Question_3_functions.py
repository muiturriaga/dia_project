from .Tools import *
import numpy as np

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


    # Find nodes who activate an A node.
def credit_nodes(list_A_node_activated, list_edges_activated, pulled_super_arm):
    list_credit = [0]*len(pulled_super_arm)
    for i in range(len(pulled_super_arm)):
        pull_node = pulled_super_arm[i]

        if pull_node in list_A_node_activated:
# If it is an initial node, we add 1 to his credit because it is possible that it has no edges.
            list_credit[i] += 1

        tamp = get_edges_cascade(pull_node, list_edges_activated)
        if len(tamp) != 0:
            list_cascade_node = [num[1] for num in tamp] # get the inherited nodes from the cascade
            for node in list_cascade_node:
                if node in list_A_node_activated: # add a credit if it activates an A node.
                    list_credit[i] +=1

    return list_credit


def calculate_reward(pulled_super_arm, env, list_nodes_info):
    list_rewards_super_arm = [0] * len(pulled_super_arm)

    # Simulate an episode.
    [episode , list_edges_activated]= simulate_episode(init_prob_matrix = env.p, n_steps_max = 100, budget = env.budget, perfect_nodes =  pulled_super_arm)

    # We count only nodes activated regardless of their message. If track = True then we assign rewards at a specific node by tracing the root. If track = False we give the same rewards to all the nodes.
    credits_of_each_node, score_by_seeds_history, nbr_activates = credits_assignement(dataset = [episode], dataset_edges = [list_edges_activated], list_nodes = list_nodes_info, track = env.bool_track)

    # print(credits_of_each_node, '\n')

    if env.bool_track == False:
        i = 0
        for node in pulled_super_arm:
            list_rewards_super_arm[i] = (credits_of_each_node[node])/(len(list_nodes_info)) #practilly is the % of nodes activated in the  network
            i +=1
    else:
        i = 0
        for node in pulled_super_arm:
            list_rewards_super_arm[i] = (credits_of_each_node[node]/len(list_nodes_info))  
            i +=1


    return list_rewards_super_arm
