from enum import Enum
from typing import Dict, Any

import numpy as np


class Gender(Enum):
    f = 0
    m = 1
    other = 2


class Age(Enum):
    young = 0
    old = 1
    teenage = 2


class Location(Enum):
    asia = 0
    europe = 1
    africa = 2
    america = 3



class Interests(Enum):
    makeup = 0
    sports = 1
    politics = 2
    science = 3
    education = 4
    sales_and_marketing = 5
    other = 6

    def describe(self):
        # self is the member here
        return self.name, self.value


class SpecialFeatures(Enum):
    A = 0
    B = 1
    C = 2


class Features:
    def __init__(self, gender: Gender, age, interests: [Interests], location: Location,
                 special_feature: SpecialFeatures):
        self.gender = gender
        self.age = age
        self.interests = interests
        self.flattened_interests = self.flatten_interests()
        self.location = location
        self.special_feature = special_feature

    def features_dictionary(self):
        return {"gender": self.gender,
                "age": self.age,
                "interests": self.interests,
                "location": self.location,
                "special_feature": self.special_feature
                }

    def flatten_interests(self):
        interests_dict: Dict[str, int] = {}
        for element in self.interests:
            interests_dict[element.name] = element.value
        return interests_dict
