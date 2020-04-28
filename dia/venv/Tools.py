import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Models.Graph import *
from networkx.algorithms import bipartite
import random


def create_graph(message_left, message_right, N_nodes, treshold):
    color_map = [''] * N_nodes

    # Initialisation of nodes.
    list_Nodes = []
    for i in range(N_nodes):
        list_Nodes.append( Nodes(i) )

    # Assignments of features.
    set_right_num = []
    set_right_nodes = []
    set_left_num = []
    set_left_nodes = []
    for node in list_Nodes:
        if node.special_feature[0] in message_right:
            set_right_num.append(node.num)
            set_right_nodes.append(node)
            color_map[node.num] = 'b'
        else:
            set_left_num.append(node.num)
            set_left_nodes.append(node)
            color_map[node.num] = 'r'


    list_Edges = []
    set_edges = []
    set_nodes = set_right_nodes + set_left_nodes
    set_nodes_num = [i.num for i in set_nodes]
    for node1 in set_nodes:
        for node2 in set_nodes:
            if np.random.random() < treshold and node1.num != node2.num:
                list_Edges.append(Edges(node1, node2))
                set_edges.append((node1.num, node2.num))

    edges_info = [set_edges, list_Edges]
    nodes_info = [[set_right_num, set_right_nodes], [set_left_num, set_left_nodes], list_Nodes]
    return edges_info, nodes_info, color_map


def draw_graph(edges_info, nodes_info, color_map):
    G_project = nx.Graph()
    for node in nodes_info[2]:
        G_project.add_node(node.num)

    G_project.add_edges_from(edges_info[0])

    nx.draw_networkx(G_project, labels = None, node_color = color_map)
    nx.spring_layout(G_project)
    plt.show()

# Question 2.
def simulate_episode(init_prob_matrix, n_steps_max, budget):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]

    # We activates a number of nodes equal to the budget considering 1 node = 1 unit of budget.
    initial_active_nodes = np.zeros(n_nodes)
    for i in range(budget):
        initial_active_nodes[i] = 1
    random.shuffle(initial_active_nodes)

    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t = 0
    while(t<n_steps_max and np.sum(newly_active_nodes) > 0):
        p = (prob_matrix.T* active_nodes).T
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        prob_matrix = prob_matrix* ((p!=0)==activated_edges)
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1-active_nodes)
        active_nodes = np.array(active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis = 0)
        t+=1
    return history

# We want to estimate the probability of edges of node_index.
def estimate_node_message(dataset, message, list_nodes):
    n_nodes = len(list_nodes)
    list_credit = np.zeros(n_nodes)

    for episode in dataset:
        credit = 0
        # What are the initial nodes ?
        idx_initial = np.argwhere(episode[0] == 1).reshape(-1)
        for index_steps in range(len(episode)):
            # What are the nodes activated at each step ?
            idx_w_active = np.argwhere(episode[index_steps] == 1).reshape(-1)
            for num_node in idx_w_active:
                if list_nodes[num_node].special_feature == message:
                    credit +=1
        # Upgrade credits of initial nodes.
        for node_index in idx_initial:
            list_credit[node_index] += credit/len(dataset)
    return list_credit

def check_attributes(list_num_nodes, list_nodes):
    n = len(list_num_nodes)
    attribute = []
    for index in range(n):
        attribute.append(list_nodes[list_num_nodes[index]].special_feature[0])
    return attribute