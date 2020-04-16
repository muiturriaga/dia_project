import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

## First steps
# Asymetric Digraph
G_asymmetric = nx.DiGraph()
G_asymmetric.add_edge('A','B')
G_asymmetric.add_edge('A','D')
G_asymmetric.add_edge('C','A')
G_asymmetric.add_edge('D','E')
nx.spring_layout(G_asymmetric)
nx.draw_networkx(G_asymmetric)
plt.show()

# Shortest path
nx.shortest_path(G_asymmetric, 'A', 'E')

## First file
# Extract nodes and edges
f = open("soc-Epinions1.txt", "r")
start = 4
f_line = f.readlines()[start:]

max_nodes = 100
N_nodes = min(max_nodes , 75879)
l_nodes = [i for i in range(N_nodes)]

l_nodes =  []
for l in f_line:
    if max(list(map(int,l.split()))) < max_nodes:
        l_nodes.append( list(map(int,l.split())) )

##
# Create Graph
G_data = nx.DiGraph()
G_data.add_nodes_from(l_nodes)
G_data.add_edges_from(l_nodes)
nx.draw_networkx(G_data)
nx.spring_layout(G_data)
plt.show()

## Project
# Nodes
import random
from networkx.algorithms import bipartite

class Nodes():
    def __init__(self, features, special_f, num):
        self.num = num
        self.features = features
        self.special_f = special_f

class Edges():
    def __init__(self, Node_1, Node_2):
        # Features of the edge.
        self.features = [np.abs(Node_1.features[i] - np.abs(Node_2.features[i])) for i in range(3)]
        self.nodes = [Node_1.num, Node_2.num]

        # Activation probability
        self.theta = np.random.dirichlet(np.ones(3), size = 1)
        self.proba_activation = np.dot(self.theta, self.features)


G_project = nx.Graph()

# Feature of the node.
list_special_f = ['A', 'B', 'C']
list_f = [[0,1] , [0,1] , [0,1]  ]
N_nodes = 25
color_map = [''] * N_nodes

# Initialisation of nodes.
list_Nodes = []
for i in range(N_nodes):
    # We give more weights on C points, else there is too much D points.
    special_f = np.random.choice(list_special_f, 1, p= [0.28, 0.28, 0.44])
    features = [random.choice(list_f[j]) for j in range(3)]
    list_Nodes.append( Nodes( features = features, special_f = special_f, num = i) )

# Assignments of features.
set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_Nodes:
    if node.special_f == 'C':
        set_right_num.append(node.num)
        set_right_nodes.append(node)
        color_map[node.num] = 'b'
        G_project.add_node(node.num, bipartite = 1)
    else:
        set_left_num.append(node.num)
        set_left_nodes.append(node)
        color_map[node.num] = 'r'
        G_project.add_node(node.num, bipartite = 0)

# Add other nodes D.

if len(set_right_nodes) < len(set_left_nodes):
    n_diff = abs(len(set_right_num) - len(set_left_num))
    for i in range(1,n_diff+1):
        set_right_num.append(i + N_nodes)
        set_right_nodes.append(Nodes( features = features, special_f = 'D', num = (i + N_nodes)))
        color_map.append('g')
        G_project.add_node((i+N_nodes), bipartite = 1)


list_Edges = []
set_edges = []
for node_r in set_right_nodes:
    for node_l in set_left_nodes:

        # Here is the point, what is the adequate probability for matching ?
        if (np.random.random() < 0.8) :
            list_Edges.append(Edges(node_r, node_l))
            set_edges.append((node_r.num, node_l.num))

G_project.add_edges_from(set_edges)

nx.draw_networkx(G_project, labels = None, node_color = color_map)
nx.spring_layout(G_project)
plt.show()

##
X,Y = bipartite.sets(G_project)
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
nx.draw(G_project, pos=pos, node_color = color_map)
plt.show()

## Question 2.
# 1. Create a subgraph independent of the message.

import random

def active_edges(Graph, List_of_Edges_Graph, active_print):

    # Activation of some initial nodes
    N_nodes = len(Graph.nodes)
    num_init = int(len(set_left_num)/2)
    list_seeds = random.sample(set_left_num,num_init)
    list_active = [i for i in list_seeds]
    list_new_active = list_active
    list_active_edges_num = []
    list_active_edges = []

    if active_print == True:
        print('List of seeds ', list_seeds)
        print('List of active nodes ', list_new_active)

    while len(list_new_active) != 0:
        list_add_edge = []
        # Activation probability
        for edge in List_of_Edges_Graph :
            # If the edge connects one active and one non active node.
            if ( tuple(edge.nodes)[0] in list_new_active and tuple(edge.nodes)[1] not in list_active ) or ( tuple(edge.nodes)[1] in list_new_active and tuple(edge.nodes)[0] not in list_active ):

                p = round(edge.proba_activation[0],5)
                # Activate or not the edge depending of his probability of activation.
                if np.random.binomial(1,p) == 1:
                    # What is the node not activated ?
                    if edge.nodes[0] not in list_new_active :
                        list_add_edge.append(edge.nodes[0])
                    else :
                        list_add_edge.append(edge.nodes[1])
                        list_active_edges.append(edge)
                    list_active_edges_num.append(edge.nodes)

        list_new_active = list(set(list_add_edge))
        list_active+=list_new_active

        if active_print == True:
            print('Now the current list is ', list_new_active, '\n')

    return list_active_edges_num, list_active_edges

plt.figure()
G_activate = nx.Graph()
G_activate.add_nodes_from(G_project.nodes)
list_activated_edges_num = active_edges(G_project, list_Edges, False)[0]
G_activate.add_edges_from(list_activated_edges_num)
nx.draw_networkx(G_activate, labels = None, node_color = color_map)
nx.spring_layout(G_activate)
plt.show()

print('Number of edges for the activate graph', len(G_activate.edges))
print('Number of edges for the project graph', len(G_project.edges))

# 2. Design the matching application.
def get_node_special_feature(num_node, list_node):
    for node in list_node:
        if node.num == num_node:
            return node.special_f[0]

def matching_application_message(list_nodes, list_activated, message):
    list_nodes_in_matching = []
    for edge in list_activated:
        spec_f_node1 = get_node_special_feature(edge.nodes[0], list_nodes)
        spec_f_node2 = get_node_special_feature(edge.nodes[1], list_nodes)
        print(spec_f_node1, spec_f_node2)
        # Check the special feature of the nodes.
        if spec_f_node1 == spec_f_node2:

            list_nodes_in_matching.append(edge.nodes[0], edge.nodes[1])
    return list_nodes_in_matching