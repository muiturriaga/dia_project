import numpy as np
import math
import sys, os

# to import bipartite
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'bipartite')))
sys.path.append(path2add)
import oracle


class CMAB_Environment:
    def __init__(self, list_of_all_arms):
        self.list_of_all_arms = list_of_all_arms
        self.reward_per_round = 0
        self.Ti = np.zeros(len(list_of_all_arms))
        self.Si = np.zeros(len(list_of_all_arms))
        self.matching_result = []
        self.mu_bar = np.ones(len(list_of_all_arms))

    def round(self, super_arm_to_pull):
        # in each round, a set of arms (called a super arm) are played together. (page 2 of the article)
        # In each round, one super arm S ∈ S is played and the outcomes of arms in S are revealed.

        self.matching_result = oracle.Matching(super_arm_to_pull)
        for arm in super_arm_to_pull:
            self.Ti[arm.i] += 1
            if arm.weight != 0:
                self.Si[arm.i] += 1

        return self.matching_result

    def opt(self):
        opt_value = 0
        for arms in self.matching_result.matched_list:
            opt_value = opt_value + self.mu_bar[arms.i]
        return opt_value
