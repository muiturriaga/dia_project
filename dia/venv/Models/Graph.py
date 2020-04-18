from numpy import np
import math


def create_features(node_number):
    np.random.seed(node_number)

    # creating an empty array for each feature
    gender = np.zeros(3)
    age = np.zeros(3)
    location = np.zeros(7)
    special_feature = np.zeros(3)

    # assign a random attribute for each feature except for interests which can
    # assume multiple attributes
    gender[np.random.randint(3)] = 1
    age[np.random.randint(3)] = 1
    location[np.random.randint(7)] = 1
    special_feature[np.random.randint(3)] = 1
    interests = np.random.randint(2, size=7)

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

