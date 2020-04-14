from enum import Enum


class Gender(Enum):
    f = 0
    m = 1
    other = 2


class Location:
    def __init__(self, location):
        self.location = location


class Interests(Enum):
    makeup = 0
    sports = 1
    politics = 2
    science = 3
    education = 4
    sales_and_marketing = 5
    other = 6


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
        self.location = location
        self.special_feature = special_feature
