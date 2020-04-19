from typing import Dict, Any
import numpy as np


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

class Features:
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
