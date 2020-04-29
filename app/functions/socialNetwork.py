import numpy as np
import math
import random
import json
import time
from typing import Dict, Any

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
    def __init__(self, num, special_feature=None, old_features_dict = None):
        self.id = num
        self.special_feature = special_feature if special_feature is not None else np.random.choice(
            ['A', 'B', 'C'], 1, p=[0.28, 0.28, 0.44])[0] #[0] to have the feature equal to a str "A", not =["A"] ##faster and lighter
        if old_features_dict == None:
            self.features: dict = self.create_features(num)
        else:
            self.features = old_features_dict

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

        interests = np.random.randint(2, size=7)
        # .tolist() transform the N dimanesional array (this case n=1) to a simple array, actually the same but more manageable
        selected_features: Feature = Feature(gender.tolist(), age.tolist(), interests.tolist(), location.tolist())
        return selected_features.features_dictionary()

class Edge:
    def __init__(self, Node_1: Node, Node_2: Node, old = False):
        # It's not a good idea to save again the entire node as head or tail.
        self.head: Node = Node_1
        self.tail: Node = Node_2
        if old == False:
            self.feature_distance = self.calculate_features_distance()
            self.similarity_distance = self.measure_similarity_distance()
            self.theta = np.random.dirichlet(np.ones(len(self.feature_distance)), size=1).tolist() #tolist() to avoid numpy.ndarray
            # I don't know what to write here
            self.proba_activation = 0
        else: #load existent Edge
            self.feature_distance = old.get('feature_distance')
            self.similarity_distance = float(old.get('similarity_distance'))
            self.theta = old.get('theta')
            self.proba_activation = old.get('proba_activation')


    def calculate_features_distance(self):
        features_distance = {}
        head = self.head.features
        tail = self.tail.features
        # Map to convert into float type, instead of using int32 of numpy
        # Actually it can be also int instead of float, Tara please confirm it.
        for feature in head:
            features_distance[feature] = float(np.dot(head[feature], tail[feature]))
        return features_distance

    def measure_similarity_distance(self):
        fe = 0
        for key, value in normalizing_factor.items():
            if key == "interests":
                fe += value*(1 - self.feature_distance[key]/6)
            else:
                fe += value*(1 - self.feature_distance[key])
        fe += noise
        return float(fe)

class Graph:
    def __init__(self, numberNodes, probConnection, old = False):
        self.numberNodes = numberNodes
        self.probConnection = probConnection
        self.list_nodes = []
        self.list_edges = []
        if old == False:
            self.adjacent_matrix = [[] for i in range(self.numberNodes)]
            self.create_nodes()
            self.connect_graph()
        else: #load existent graph
            self.load_from_json(old)

    def create_nodes(self):
        for i in range(self.numberNodes):
            self.list_nodes.append(Node(i))

    #Generate a condenced adjacent matrix.
    #self.adjacent_matrix[XX] will be a list of all the EDGES connected to XX
    def connect_graph(self):
        k = 0
        for i in range(self.numberNodes):
            for j in range(i + 1, self.numberNodes): ## i + 1 to not have repeated edges as we have undirected socialnetwork (half-time computing)
                if random.random() <= self.probConnection:
                    self.list_edges.append(Edge(self.list_nodes[i], self.list_nodes[j]))
                    self.adjacent_matrix[i].append(k) ## will be used as a pointer
                    self.adjacent_matrix[j].append(k)
                    k += 1

    def turn_self_dict(self):
        for i in range(self.numberNodes):
            self.list_nodes[i] = self.list_nodes[i].__dict__
        self.list_nodes = {i : self.list_nodes[i] for i in range(len(self.list_nodes))}
        for i in range(len(self.list_edges)):
            self.list_edges[i].head = self.list_edges[i].head.__dict__
            self.list_edges[i].tail = self.list_edges[i].tail.__dict__
            self.list_edges[i] = self.list_edges[i].__dict__
        self.list_edges = {i : self.list_edges[i] for i in range(len(self.list_edges))}
        self.adjacent_matrix = {i: self.adjacent_matrix[i] for i in range(len(self.adjacent_matrix))}
        return self.__dict__

    def load_from_json(self, jsonfile):
        # recovering list of nodes from json
        for item in jsonfile.get('list_nodes').items():
            self.list_nodes = Node(item[1].get('id'), item[1].get('special_feature'), item[1].get('features'))
        # recovering adjacent matrix from json
        self.adjacent_matrix = []
        for item in jsonfile.get('adjacent_matrix').items():
            self.adjacent_matrix.append(item[1])
        # recovering list of edges from json
        for k, v in jsonfile.get('list_edges').items():
            head = v.get('head')
            node_head = Node(head.get('id'), head.get('special_feature'), head.get('features'))
            tail = v.get('tail')
            node_tail = Node(tail.get('id'), tail.get('special_feature'), tail.get('features'))
            self.list_edges.append(Edge(node_head, node_tail, v))



## Just a caller function
def load_social_network(jsonfile):
    SN = Graph(jsonfile.get('numberNodes'), jsonfile.get('probConnection'), jsonfile)
    return SN

## Check if a SN file exist, if not it will create it and return a SN
## The idea is to call this funcion from the others files
def import_social_network(numberNodes, probConnection):
    SN = None
    try:
        file = open("functions/graphs_files/graph_nodes{}_probconnection{}.txt".format(numberNodes, probConnection))
        data = json.load(file)
        SN = load_social_network(data)
        file.close()
        print("Loading social network file,\n\nNumber of nodes: {}\nProability of having an edge between two nodes: {}\nThis may take a while ...".format(numberNodes, probConnection))

    except:
        print("Creating social network,\n\nNumber of nodes: {}\nProability of having an edge between two nodes: {}\nThis may take a while ...".format(numberNodes, probConnection))
        with open("functions/graphs_files/graph_nodes{}_probconnection{}.txt".format(numberNodes, probConnection), 'w') as outfile:
            SN = Graph(numberNodes, probConnection)
            json.dump(SN.turn_self_dict(), outfile)
    finally:
        print("\nReturning Social Network | classtype: {}".format(type(SN)))
        return SN
