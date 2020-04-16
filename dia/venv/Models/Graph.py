from Models.Models import Gender
from Models.Models import Location
from Models.Models import Interests
from Models.Models import SpecialFeatures
from Models.Models import Features
from Models.Models import Age
from Models.Models import Location

from numpy.random import seed
from numpy.random import randint
import math


def create_features(node_number):
    seed(node_number)
    gender = Gender(randint(0, 2, 1, int))
    age = Age(randint(0, 2, 1, int))
    location = Location(randint(0, 3, 1 ,int))
    interests = []
    for numbers in randint(0, 6, 4):
        interests.append(Interests(numbers))
    special_feature = SpecialFeatures(randint(0, 2, 1, int))
    selected_features: Features = Features(gender, age, interests, location, special_feature)
    return selected_features.features_dictionary()


class Nodes:
    def __init__(self, num):
        self.num = num
        self.features: dict = create_features(num)


class Edges:
    def __init__(self, Node_1: Nodes, Node_2: Nodes):
        self.head: Nodes = Node_1
        self.tail: Nodes = Node_2
        self.calculate_features_distance()
        self.theta = np.random.dirichlet(np.ones(len(features)), size=1)
        self.proba_activation = np.dot(self.theta, self.features)

    def calculate_features_distance(self):
        features_distance = {}
        for feature in self.head.features:
            if feature == 'interests':
                features_distance[feature] = self.calculate_interests_distance()
            else:
                features_distance[feature] = self.head.features[feature].value - self.tail.features[feature].value
            print("features")

    def calculate_interests_distance(self):
        features_distance = {}
        head_interests = self.head.features['interests']
        tail_interests = self.head.features['interests']
        for head_interest in head_interests:
            for tail_interest in tail_interests:
                if head_interest.name == tail_interest.name:
                    features_distance[head_interest.name] = head_interest.value - tail_interest.value
                    del tail_interests[tail_interest.index]
                else:
                    features_distance[head_interest.name] = interest.value
        for tail_interest in tail_interests:
            features_distance[tail_interest.name] = interest.value
        return features_distance