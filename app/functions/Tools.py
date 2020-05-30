import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .socialNetwork import Node, Edge
from networkx.algorithms import bipartite
import random

# Multiple file variables
bool_affichage = True
N_nodes = 100
proba_edges = 0.03
Budget = int(N_nodes/20)
N_episodes = 1000
T = 1000
n_experiments = 100



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

def check_attributes(list_num_nodes, list_nodes):
    n = len(list_num_nodes)
    attribute = []
    for index in range(n):
        attribute.append(list_nodes[list_num_nodes[index]].special_feature)
    return attribute


def simulate_episode(init_prob_matrix, n_steps_max, budget, perfect_nodes):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]

    if len(perfect_nodes) == 0:
        initial_active_nodes = np.zeros(n_nodes)
        for i in range(budget):
            initial_active_nodes[i] = 1
        random.shuffle(initial_active_nodes)

    else :
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
        active_nodes = np.array( active_nodes + newly_active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history, list_new_activated_edges


def estimate_node_message(dataset, dataset_edges, list_nodes):
    n_nodes = len(list_nodes)
    list_credit = np.zeros(n_nodes)
    score_by_seeds_history = np.empty([n_nodes,len(dataset)])
    score_by_seeds_history[:] = np.NaN

    for i in range(0,len(dataset)):
        episode = dataset[i]
        history_edges = dataset_edges[i]
        list_credit_steps = np.zeros(n_nodes)

         # Catch the initial nodes
        idx_initial = np.argwhere(episode[0] == 1).reshape(-1)

        for step in history_edges[::-1]:
            # Catch new nodes activated at each step. element[0] = head and element[1] = tail.
            for element in step:
                list_credit_steps[element[0]] += max(list_credit_steps[element[1]], 1)

        # Upgrade credits of initial nodes.
        for id in idx_initial:
            score_by_seeds_history[id][i] = list_credit_steps[id]
            list_credit[id] += list_credit_steps[id]

    return list_credit, score_by_seeds_history