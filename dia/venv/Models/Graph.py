import numpy as np
import math
from Models.Models import Features

def create_features(node_number):
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

    selected_features: Features = Features(gender, age, interests, location)
    return selected_features.features_dictionary()

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
class Nodes:
    def __init__(self, num, special_feature = None):
        self.num = num
        self.special_feature = special_feature if special_feature is not None else np.random.choice(['A', 'B' , 'C'], 1, p=[0.28, 0.28, 0.44])
        self.features: dict = create_features(num)


class Edges:
    def __init__(self, Node_1: Nodes, Node_2: Nodes):
        self.head: Nodes = Node_1
        self.tail: Nodes = Node_2
        self.feature_distance = self.calculate_features_distance()
        self.similarity_distance = self.measure_similarity_distance()
        self.theta = np.random.dirichlet(np.ones(len(self.feature_distance)), size=1)
        # I don't know what to write here
        self.proba_activation = 0

    def calculate_features_distance(self):
        features_distance = {}
        head = self.head.features
        tail = self.tail.features
        for feature in head:
            features_distance[feature] = np.dot(head[feature], tail[feature])
        return features_distance

    def measure_similarity_distance(self):
        fe = 0
        for key, value in normalizing_factor.items():
            if key == "interests":
                fe += value*(1 - self.feature_distance[key]/6)
            else:
                fe += value*(1 - self.feature_distance[key])
        fe += noise
        return fe
