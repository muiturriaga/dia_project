import numpy as np
import math
import random
import time
import networkx as nx
import matplotlib.pyplot as plt

# Considering these are our models
#
#
# Gender:
#       Female, Male, Other
#       We represent this with an array with 3 elements [female, male, other]
#
# Age:
#       Teenage, Young, Old
#       We represent this with an array with 3 elements [Teenage, Young, Old]
#
# Interests:
#        Makeup, Science, Sport, Books, Politics, Technology
#        We represent this with an array with 6 elements [Makeup, Science, Sport, Books, Politics, Technology]
#
# Location:
#        Europe, Asia, North_America, Africa, South_America, Antarctica, Australia
#        We represent this with an array with 7 elements
#        [Europe, Asia, North_America, Africa, South_America, Antarctica, Australia]
#
# Special_feature_f*
#        A, B, C
#
#
# Element 0 means that the attribute does not belong to that feature, while 1 indicates the selected
# features for the Node.
# For example if the vector of gender is like this [1, 0, 0], it means that Node is a male and the
# attributes female and other doesn't belong to that node

k1 = 0.25
k2 = 0.15
k3 = 0.15
k4 = 0.4
noise = 0.05
normalizing_factor = {"gender": k1,
                      "age": k3,
                      "interests": k4,
                      "location": k2
                      }

class Feature:
    def __init__(self, gender, age, interests, location):
        self.gender = gender
        self.age = age
        self.interests = interests
        self.location = location

    def features_dictionary(self):
        return {"gender": self.gender,
                "age": self.age,
                "interests": self.interests,
                "location": self.location
                }

class Node:
    def __init__(self, num, special_feature=None, features=None):
        self.id = num
        self.activated = False # useful for spreading messages
        self.has_share = False # useful for spreading messages
        self.total_neighbor_activated_cascade = 0
        self.activated_by_same_type = False
        if not special_feature:
            self.special_feature = np.random.choice(['A', 'B', 'C'], 1, p=[0.33, 0.33, 0.34])[0] #[0] to have the feature equal to a str "A", not =["A"] ##faster and lighter
        else:
            self.special_feature = special_feature
        if not features:
            self.features: dict = self.create_features(num)
        else:
            self.features = features

    def create_features(self, node_number):
        np.random.seed(node_number)

        # creating an empty array for each feature
        gender = np.zeros(3)
        age = np.zeros(3)
        location = np.zeros(7)

        # assign a random attribute for each feature except for interests which can
        # assume multiple attributes
        gender[np.random.randint(3)] = 1
        age[np.random.randint(3)] = 1
        location[np.random.randint(7)] = 1

        interests = np.random.randint(2, size=6)
        # .tolist() transform the N dimanesional array (this case n=1) to a simple array, actually the same but more manageable
        selected_features: Feature = Feature(gender.tolist(), age.tolist(), interests.tolist(), location.tolist())
        return selected_features.features_dictionary()

class Edge:
    def __init__(self, Node_1: Node, Node_2: Node):
        self.head = Node_1.id
        self.tail = Node_2.id

        self.feature_distance = self.calculate_features_distance(Node_1, Node_2)
        self.similarity_distance = self.measure_similarity_distance()
        self.theta = np.random.dirichlet(np.ones(len(self.feature_distance)), size=1).tolist() #tolist() to avoid numpy.ndarray
        # probability of activation based on the similarity_distance
        self.proba_activation = math.exp(-self.similarity_distance**2)

    def calculate_features_distance(self, node1, node2):
        features_distance = {}
        head = node1.features
        tail = node2.features
        for feature in head:
            # Compute the sum of the difference as it is written in the project.
            diff = np.dot(head[feature], tail[feature])
            features_distance[feature] = diff
        return features_distance

    def measure_similarity_distance(self):
        fe = 0
        for key, value in normalizing_factor.items():
            if key == "interests":
                fe += value*(1 - self.feature_distance[key]/7)
            else:
                fe += value*(1 - self.feature_distance[key])
        fe += noise
        # fe is equal to 1 in the perfect case where nodes are perfectly the same. In the other dual case, fe is equal to 0.
        return float(fe)

# Create Social Network graph
# Return: edges_info -> edges_info[0] = set of edges in the form of node.id pair (i,j)
#                       edges_info[1] = list of objects Edge
#         nodes_info -> nodes_info[0] = set of nodes id
#                       nodes_info[1] = list of objects Node
#         color_map -> array of color for each nodes, positional using node.id, A=red, B=blue, C=yellow
def create_sn(n_nodes):
    threshold = np.log(n_nodes)/(2*n_nodes)
    color_map = [''] * n_nodes

    # Initialisation of nodes.
    list_nodes = []
    for i in range(n_nodes):
        list_nodes.append(Node(i))

    set_nodes = []
    for node in list_nodes:
        set_nodes.append(node.id)
        if node.special_feature == 'A':
            color_map[node.id] = 'r'
        elif node.special_feature == 'B':
            color_map[node.id] = 'b'
        else:
            color_map[node.id] = 'y'

    # Initialisation of edges.
    list_edges = []
    set_edges = []
    for node1 in list_nodes:
        for node2 in list_nodes:
            # Create an edge iif a number generated is inferior to our threshold.
            if np.random.random() < threshold and node1.id != node2.id:
                list_edges.append(Edge(node1, node2))
                set_edges.append((node1.id, node2.id))

    edges_info = [set_edges, list_edges]
    nodes_info = [set_nodes, list_nodes]

    return edges_info, nodes_info, color_map

# Draw social network graph
# edges_info = set of edges returned by create_sn, in the form of (i,j)
# nodes_info = set of nodes returned by create_sn, set of nodes id
def draw_sn(edges_info, nodes_info, color_map):
    graph = nx.Graph()
    graph.add_nodes_from(nodes_info)
    graph.add_edges_from(edges_info)

    nx.draw_networkx(graph, labels=None, node_color=color_map)
    nx.spring_layout(graph)
    plt.show()
