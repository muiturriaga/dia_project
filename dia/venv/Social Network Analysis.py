import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

## Project
# Nodes
import random
from numpy.random import seed
from numpy.random import randint
from networkx.algorithms import bipartite
from Models.Models import Features
from Models.Models import Gender
from Models.Models import Interests
from Models.Models import SpecialFeatures



class Nodes():
    def __init__(self, num):
        self.num = num
        self.features: Features = self.create_features(num)

    def create_features(self, node_number):
        seed(node_number)
        gender = Gender(randint(0, 2, 1, int))
        age = randint(12, 99)
        # location = Location
        location = "location"
        interests = []
        for numbers in randint(0, 6, 4):
            interests.append (Interests(numbers))
        special_feature = SpecialFeatures(randint(0, 2, 1, int))
        selected_features = Features(gender, age, interests, location, special_feature)
        return selected_features

class Edges():
    def __init__(self, Node_1: Nodes, Node_2: Nodes):
        self.features = [np.abs(Node_1.features[i] - np.abs(Node_2.features[i])) for i in range(3)]
        self.theta = np.random.dirichlet(np.ones(len(features)), size=1)
        self.proba_activation = np.dot(self.theta, self.features)

node = Nodes(1)
if node.features.special_feature == "C":
    print("C")
if node.features.special_feature == "A":
    print("A")
if node.features.special_feature.row == "B":
    print("B")
print(node.features.special_feature)
G_project = nx.Graph()

N_nodes = 1000

list_Nodes = []
for i in range(N_nodes):

    list_Nodes.append(Nodes(i))

set_right_num = []
set_right_nodes = []
set_left_num = []
set_left_nodes = []
for node in list_Nodes:
    if node.features.special_feature == 'C':
        set_right_num.append(node.num)
        set_right_nodes.append(node)
    else:
        set_left_num.append(node.num)
        set_left_nodes.append(node)

list_Edges = []
set_edges = []
for node_r in set_right_nodes:
    for node_l in set_left_nodes:
        if (np.random.random() < 0.2):
            list_Edges.append(Edges(node_r, node_l))
            set_edges.append((node_r.num, node_l.num))

G_project.add_nodes_from(set_right_num, bipartite=1)
G_project.add_nodes_from(set_left_num, bipartite=0)
G_project.add_edges_from(set_edges)

nx.draw_networkx(G_project, labels=None)
nx.spring_layout(G_project)
plt.show()
