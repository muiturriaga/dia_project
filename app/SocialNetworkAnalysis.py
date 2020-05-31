from networkx.algorithms import bipartite
from functions.socialNetwork import *
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import os
#os.chdir("C:/Users/Loick/Documents/Ecole Nationale des Ponts et Chaussées/2A/Erasmus Milan/Data Analysis/dia_project-master/dia_project-master/dia/venv")
#set()


# Project
# Node

N_nodes = 25
G_project = nx.Graph()
color_map = [''] * N_nodes

# Initialisation of nodes.
list_Nodes = []
for i in range(N_nodes):
    # We give more weights on C points, else there is too much D points.
    list_Nodes.append(Node(i))

# Assignments of features.
set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_Nodes:
    if node.special_feature == 'C':
        set_right_num.append(node.id)
        set_right_nodes.append(node)
        color_map[node.id] = 'b'
        G_project.add_node(node.id, bipartite=1)
    else:
        set_left_num.append(node.id)
        set_left_nodes.append(node)
        color_map[node.id] = 'r'
        G_project.add_node(node.id, bipartite=0)


list_Edges = []
set_edges = []
set_nodes = set_right_nodes + set_left_nodes
set_nodes_num = [i.id for i in set_nodes]
for node1 in set_nodes:
    for node2 in set_nodes:
        # Here is the point, what is the adequate probability for add edges ?
        if np.random.random() < 0.1 and node1.id != node2.id:
            list_Edges.append(Edge(node1, node2))
            set_edges.append((node1.id, node2.id))

G_project.add_edges_from(set_edges)

nx.draw_networkx(G_project, labels=None, node_color=color_map)
nx.spring_layout(G_project)
plt.show()

print("number of A or B nodes ", len(set_left_num))
print("number of C nodes ", len(set_right_num))

# for i in set_nodes:
#     print(i.special_feature)

# Question 2.
# 1. Create a sub-graph independent of the message.


def active_edges(Graph, List_of_Edges_Graph, active_print: bool):

    # Activation of some initial nodes
    N_nodes = len(Graph.nodes)
    num_init = int(len(set_nodes_num)/9)
    list_seeds = random.sample(set_nodes_num, num_init)
    list_active = [i for i in list_seeds]
    list_new_active = [i for i in list_active]
    list_active_edges_num = []
    list_active_edges = []

    if active_print:
        print('List of seeds ', list_seeds)
        print('List of active nodes ', list_new_active)

    while len(list_new_active) != 0:
        list_add_edge = []
        # Activation probability
        for edge in List_of_Edges_Graph:
            # If the edge connects one active and one non active node.

            if edge.head.id in list_new_active and edge.tail.id not in list_active:
                p = round(edge.measure_similarity_distance(), 5)
                # Activate or not the edge depending of his probability of activation.
                if np.random.binomial(1, p) == 1:
                    list_add_edge.append(edge.tail.id)
                    list_active_edges.append(edge)
                    vect_num = [edge.head.id, edge.tail.id]
                    list_active_edges_num.append(vect_num)

        list_new_active = list(set(list_add_edge))
        list_active += list_new_active

        if active_print:
            print('Now the current list is ', list_new_active, '\n')

    return list_active_edges_num, list_active_edges


plt.figure()
G_activate = nx.Graph()
G_activate.add_nodes_from(G_project.nodes)
[list_activated_edges_num, list_activated_edges] = active_edges(G_project, list_Edges, False)
G_activate.add_edges_from(list_activated_edges_num)
nx.draw_networkx(G_activate, labels=None, node_color=color_map)
nx.spring_layout(G_activate)
plt.show()

print('Number of edges for the activate graph', len(G_activate.edges))
print('Number of edges for the project graph', len(G_project.edges))


# Were nodes A activated by an A node ?
def filter_node_by_message(List_activated_edges, message):
    list_activated_matching = [[], [], []]
    list_activated_matching_num = [[], [], []]
    for edge in List_activated_edges:
        # Check if the head have the same special feature than the tail. In that case, save the tail and the head for the matching.
        if edge.head.special_feature == edge.tail.special_feature and message == edge.tail.special_feature:
            # print(ord(edge.tail.special_feature[0])-65)
            num = (ord(edge.tail.special_feature[0])-65)
            list_activated_matching_num[num].append(edge.tail.id)
    unique_result = [list(set(i)) for i in list_activated_matching_num]
    return unique_result


# Three cascades
activate_matching = []
for message_factor in ['A', 'B', 'C']:
    # Activation for each cascades
    [list_activated_edges_num, list_activated_edges] = active_edges(G_project, list_Edges, False)
    # Check special_feature activation for each node
    activate_matching.append(filter_node_by_message(
        list_activated_edges, message_factor)[(ord(message_factor)-65)])

print(activate_matching)


G_matching = nx.Graph()
set_matching_left = activate_matching[0] + activate_matching[1]
set_matching_right = activate_matching[2]

color_map_matching = []
for i in set_matching_right:
    color_map_matching.append(color_map[i])
G_matching.add_nodes_from(set_matching_right, bipartite=1)
for i in set_matching_left:
    color_map_matching.append(color_map[i])
G_matching.add_nodes_from(set_matching_left, bipartite=0)


print("Number of A or B nodes in the matching :", len(set_matching_left))
print("Number of C nodes in the matching :", len(set_matching_right))


# Add other nodes D.
if len(set_matching_right) < len(set_matching_left):
    n_diff = abs(len(set_matching_right) - len(set_matching_left))
    print(n_diff)
    for i in range(1, n_diff+1):
        set_matching_right.append(i + N_nodes)
        color_map_matching.append('g')
        G_matching.add_node((i+N_nodes), bipartite=1)

plt.figure()
# What are the edges we have to plot ?
list_edges_matching = [[i, j] for i in set_matching_left for j in set_matching_right]
G_matching.add_edges_from(list_edges_matching)

# Separate by group
X, Y = nx.bipartite.sets(G_matching)
pos = {}
# Update position for node from each group
pos.update((node, (2, index)) for index, node in enumerate(X))
pos.update((node, (1, index)) for index, node in enumerate(Y))

nx.draw(G_matching, pos=pos,  node_color=color_map_matching)
plt.show()
