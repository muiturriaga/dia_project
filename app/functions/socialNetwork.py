import numpy as np
import math
import random
import json
import time
from queue import Queue
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
#        We represent this with an array with 7 elements [Makeup, Science, Sport, Books, Politics, Technology]
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
        self.activated = False # useful for spreading messages
        self.has_share = False # useful for spreading messages
        self.total_neighbor_activated_cascade = 0
        self.activated_by_same_type = False
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
        self.head = Node_1.id
        self.tail = Node_2.id
        if old == False:
            self.feature_distance = self.calculate_features_distance(Node_1, Node_2)
            self.features = self.get_features(Node_1, Node_2)
            self.similarity_distance = self.measure_similarity_distance()
            self.theta = np.random.dirichlet(np.ones(len(self.feature_distance)), size=1).tolist()[0]

            self.proba_activation = np.dot(self.theta, [ (1 - self.feature_distance['gender']), (1 -  self.feature_distance['age']), (1 - self.feature_distance['interests']), (1 - self.feature_distance['location']) ])
            # If our feature_distance is [0,0,0,0] we want to have a probability of  to activate the edge. That is why there is "1-feature(edge)". Theta is a weight vector which change randomly the weight between the features.

        else:
            self.feature_distance = old.get('feature_distance')
            self.features = old.get('features')
            self.similarity_distance = float(old.get('similarity_distance'))
            self.theta = old.get('theta')
            self.proba_activation = old.get('proba_activation')

        self.features_1 = Node_1.features
        self.features_2 = Node_2.features

    def get_features(self, node1, node2):
        gender_1 = np.array([int(i) for i in node1.features['gender']])
        gender_2 = np.array([int(i) for i in node2.features['gender']])
        list_gender = list(gender_1 & gender_2)

        age_1 = np.array([int(i) for i in node1.features['age']])
        age_2 = np.array([int(i) for i in node2.features['age']])
        list_age = list(age_1 & age_2)

        interests_1 = np.array([int(i) for i in node1.features['interests']])
        interests_2 = np.array([int(i) for i in node2.features['interests']])
        list_interests = list(interests_1 & interests_2)

        location_1 = np.array([int(i) for i in node1.features['location']])
        location_2 = np.array([int(i) for i in node2.features['location']])
        list_location = list(location_1 & location_2)

        return (list_gender + list_interests + list_age + list_location)

    def calculate_features_distance(self, node1, node2):
        features_distance = {}
        head = node1.features
        tail = node2.features
        for feature in head:
            sum_diff = 0 # Compute the sum of the difference as it is written in the project.
            for i in range(len(head[feature])):
                sum_diff += abs(head[feature][i] - tail[feature][i])
            features_distance[feature] = round(sum_diff/len(head[feature]),2)
        return features_distance
        # [1,1,1,1] means we are really dissimilar while [0,0,0,0] implies the nodes have the same features.

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
        if old == False: #create a new graph
            self.adjacent_matrix = [[] for i in range(self.numberNodes)]
            self.create_nodes()
            self.connect_graph()
        else: #load existent graph
            self.load_from_json(old)

    ## Create all the random nodes needed
    def create_nodes(self):
        for i in range(self.numberNodes):
            self.list_nodes.append(Node(i))

    ## Generate a condenced adjacent matrix.
    ## This matrix is kinda matrix[node_id] = list_of_edges_ids
    ## self.adjacent_matrix[XX] will be a list of all the EDGES connected to XX
    def connect_graph(self):
        k = 0
        for i in range(self.numberNodes):
            for j in range(i + 1, self.numberNodes): ## i + 1 to not have repeated edges as we have undirected socialnetwork (half-time computing)
                if random.random() <= self.probConnection:
                    self.list_edges.append(Edge(self.list_nodes[i], self.list_nodes[j]))
                    self.adjacent_matrix[i].append(k) # will be used as a pointer
                    self.adjacent_matrix[j].append(k)
                    k += 1

    ## Function transforming the Graph class into a JSON serializable
    def turn_self_dict(self):
        for i in range(self.numberNodes):
            self.list_nodes[i] = self.list_nodes[i].__dict__
        self.list_nodes = {i : self.list_nodes[i] for i in range(len(self.list_nodes))}
        for i in range(len(self.list_edges)):
            # self.list_edges[i].head = self.list_edges[i].head.__dict__
            # self.list_edges[i].tail = self.list_edges[i].tail.__dict__
            self.list_edges[i] = self.list_edges[i].__dict__
        self.list_edges = {i : self.list_edges[i] for i in range(len(self.list_edges))}
        self.adjacent_matrix = {i: self.adjacent_matrix[i] for i in range(len(self.adjacent_matrix))}
        return self.__dict__

    ## Function used to generate a graph from JSON file
    def load_from_json(self, jsonfile):
        # recovering list of nodes from json
        for item in jsonfile.get('list_nodes').items():
            self.list_nodes.append(Node(item[1].get('id'), item[1].get('special_feature'), item[1].get('features')))
        # recovering adjacent matrix from json
        self.adjacent_matrix = []
        for item in jsonfile.get('adjacent_matrix').items():
            self.adjacent_matrix.append(item[1])
        # recovering list of edges from json
        for k, v in jsonfile.get('list_edges').items():
            node_head_id = v['head']
            node_tail_id = v['tail']
            self.list_edges.append(Edge(self.list_nodes[node_head_id], self.list_nodes[node_tail_id], v))



    ## Function returning a list of all the nodes of the type desired
    def return_nodes_of_type(self, typeDesired):
        retList = []
        for node in self.list_nodes:
            if node.special_feature == typeDesired:
                retList.append(node)
        return retList


## Just a caller function
def load_social_network(jsonfile):
    SN = Graph(jsonfile.get('numberNodes'), jsonfile.get('probConnection'), jsonfile)
    return SN

## Check if a SN file exist, if not it will create it and return a SN
## The idea is to call this funcion from the others files
def import_social_network(numberNodes, probConnection):
    SN = None
    try:
        file = open("app/functions/graphs_files/graph_nodes{}_probconnection{}.txt".format(numberNodes, probConnection))
        data = json.load(file)
        SN = load_social_network(data)
        file.close()
        print("Loading social network file,\n\nNumber of nodes: {}\nProability of having an edge between two nodes: {}\nThis may take a while ...".format(numberNodes, probConnection))

    except:
        print("Creating social network,\n\nNumber of nodes: {}\nProability of having an edge between two nodes: {}\nThis may take a while ...".format(numberNodes, probConnection))
        with open("app/functions/graphs_files/graph_nodes{}_probconnection{}.txt".format(numberNodes, probConnection), 'w') as outfile:
            SN = Graph(numberNodes, probConnection)
            json.dump(SN.turn_self_dict(), outfile)

    finally:
        if type(SN) != Graph:
            print("Error in social returning classtype: {}\n\n".format(type(SN)))
        else:
            print("\nReturning Social Network | classtype: {}\n\n".format(type(SN)))
        return SN

## Function to spread a message over the network
## This function require the SN graph, the type of Message (A,B or C)
## and also a list of initial nodes for spreading
## also implemented multipletimeinfluenced=False, which means:
## if a node is no activated at the first attempt this node will be blocked,
## and in the future no other node will be able to activate it, but if we set it
## ==True, every attempt of activation will be completely valid.
## multipletimeinfluenced == False => one attempt to be activated
## multipletimeinfluenced == True  => multiple attempts to activate the node
## multipletimesharing == False => a node can spread the message one time
## multipletimesharing == True  => a node can spread the message multiple times
## so we'll have 4 possible cases for the 2 booleans vars
def spread_message_SN(SN : Graph, typeMessage, list_seeds_nodes, multipletimeinfluenced = False, multipletimesharing = False):
    queue = Queue()
    activated_nodes = []
    total_messages = 0
    # activating seeds
    for node_id in list_seeds_nodes:
        queue.put(node_id)
        activated_nodes.append(node_id)
        SN.list_nodes[node_id].activated = True
    # spreading from the queue
    while not queue.empty():
        node_id = queue.get()
        for edge_id in SN.adjacent_matrix[node_id]:
            edge = SN.list_edges[edge_id]
            from_id = node_id
            to_id = edge.tail if (from_id == edge.head) else edge.head
            toNode = SN.list_nodes[to_id]
            ##FIRST CASE multi_influenced==False, multi_sharing==False.
            if multipletimeinfluenced == False and multipletimesharing == False:
                if toNode.activated == False: #send message if not already activated
                    print("spreading from node: {} to node: {}".format(from_id, to_id))
                    total_messages += 1
                    if random.random() <= edge.proba_activation: #proving activate
                        toNode.activated = True
                        queue.put(toNode.id)
                        activated_nodes.append(toNode.id)
                    ##NOT FINISHED
            ##SECOND CASE multi_influenced==False, multi_sharing==True.
            elif multipletimeinfluenced == False and multipletimesharing == True:
                None #Constructing
            ##THIRD CASE multi_influenced==True, multi_sharing==False.
            elif multipletimeinfluenced == True and multipletimesharing == False:
                None #Constructing
            ##FOURTH CASE multi_influenced==True, multi_sharing==True.
            elif multipletimeinfluenced == True and multipletimesharing == True:
                None #Constructing
    print("\n\nSome metrics that may be useful: ")
    print("Message spreaded of type: {}".format(typeMessage))
    print("Total number of nodes: {}".format(len(SN.list_nodes)))
    print("Total number of seeds: {}".format(len(list_seeds_nodes)))
    print("Total number edges: {}".format(len(SN.list_edges)))
    print("Total nodes activated including seeds: {}".format(len(activated_nodes)))
    print("Total nodes activated without seeds: {}".format(len(activated_nodes)-len(list_seeds_nodes)))
    print("Total messages sent: {}".format(total_messages))

            ##constructing



## YOU CAN TRY THE CODE HERE
# erase = '\x1b[1A\x1b[2K'
# SN = import_social_network(1000,0.1)
# #spread_message_SN(SN, "A", [1,3,4,5,6,7,8])
