import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .socialNetwork import Node, Edge
from networkx.algorithms import bipartite
import random

#Create bipartite graph
def create_graph(message_left, message_right, n_nodes, threshold):
    color_map = [''] * n_nodes

    # Initialisation of nodes.
    list_Nodes = []
    for i in range(n_nodes):
        list_Nodes.append(Node(i))

    # Assignments of features.
    set_right_num = []
    set_right_nodes = []
    set_left_num = []
    set_left_nodes = []
    for node in list_Nodes:
        if node.special_feature in message_right:
            set_right_num.append(node.id)
            set_right_nodes.append(node)
            color_map[node.id] = 'b'
        else:
            set_left_num.append(node.id)
            set_left_nodes.append(node)
            color_map[node.id] = 'r'

    list_Edges = []
    set_edges = []
    set_nodes = set_right_nodes + set_left_nodes
    set_nodes_num = [i.id for i in set_nodes]
    for node1 in set_nodes:
        for node2 in set_nodes:
            # Create an edge iif a number generated is inferior to our threshold.
            if np.random.random() < threshold and node1.id != node2.id:
                list_Edges.append(Edge(node1, node2))
                set_edges.append((node1.id, node2.id))

    edges_info = [set_edges, list_Edges]
    nodes_info = [[set_right_num, set_right_nodes], [set_left_num, set_left_nodes], list_Nodes]
    return edges_info, nodes_info, color_map


def draw_graph(edges_info, nodes_info, color_map):
    G_project = nx.Graph()
    for node in nodes_info:
        G_project.add_node(node.id)

    G_project.add_edges_from(edges_info)

    nx.draw_networkx(G_project, labels=None, node_color=color_map)
    nx.spring_layout(G_project)
    plt.show()


## Question 2.

def simulate_episode(init_prob_matrix, n_steps_max, budget, perfect_nodes):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]

    if len(perfect_nodes) == 0:
        # We activates a number of nodes equal to the budget considering 1 node = 1 unit of budget.
        initial_active_nodes = np.zeros(n_nodes)
        for i in range(budget):
            initial_active_nodes[i] = 1
        random.shuffle(initial_active_nodes)
        # print('Initial active nodes are : ' , initial_active_nodes.nonzero()[0], '\n')
    else :
        initial_active_nodes = np.zeros(n_nodes)
        for i in perfect_nodes:
            initial_active_nodes[i] = 1

    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes

    t = 0
    while(t < n_steps_max and np.sum(newly_active_nodes) > 0):
        # Extract only rows where nodes are activated.
        p = (prob_matrix.T * active_nodes).T
        # Draw random numbers and catch if we can activate them or not. If M_ij is True it means node i activate node j.
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        # If edges were not activated, they disappear.
        prob_matrix = prob_matrix * ((p != 0) == activated_edges)
        # Return nodes where edges have been activated.
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1-active_nodes)
        active_nodes = np.array( active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history

# Find episode with initial nodes in order to verify the results
def find_episode(list_num_nodes, dataset):
    initial_nodes = [0]*len(dataset[0][0])
    for i in list_num_nodes:
        initial_nodes[i] = 1
    list_episode = []
    for i in range(len(dataset)):
        if list(dataset[i][0]) == initial_nodes:
            list_episode.append(i)

    sub_dataset = [Dataset[i] for i in list_episode]
    return list_episode,sub_dataset


def estimate_node_message(dataset, message, list_nodes):
    n_nodes = len(list_nodes)
    list_credit = np.zeros(n_nodes)
    time_credit = [0]*(len(dataset)+1)
    i = 1

    for episode in dataset:
        credit = 0
        # What are the initial nodes ?
        idx_initial = np.argwhere(episode[0] == 1).reshape(-1)
        for index_steps in range(len(episode)):
            # What are the nodes activated at each step ?
            idx_w_active = np.argwhere(episode[index_steps] == 1).reshape(-1)
            for num_node in idx_w_active:
                # If they are of type 'message', we give them credits. More a node have credits, more it is able to activate message_nodes.
                if list_nodes[num_node].special_feature == message:
                    credit += 1
        # Upgrade credits of initial nodes.
        for node_index in idx_initial:
            list_credit[node_index] += credit/len(dataset)

        time_credit[i] = credit/len(dataset) + time_credit[i-1]
        i +=1
    return [list_credit, time_credit] # Return the number of nodes approximately activated at each episode.


def check_attributes(list_num_nodes, list_nodes):
    n = len(list_num_nodes)
    attribute = []
    for index in range(n):
        attribute.append(list_nodes[list_num_nodes[index]].special_feature)
    return attribute

