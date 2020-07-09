import numpy as np
import math

from Oracle import *
from EdgeSelector import *


class CMAB_Environment:
    def __init__(self, list_of_all_arms):
        self.list_of_all_arms = list_of_all_arms
        # self.super_arm_to_pull = EdgeSelector(list_of_all_arms).selectNodes(int(len(list_of_all_arms)/16))
        # self.pool_of_nodes = pool_of_nodes

        self.reward_per_round = 0
        self.Ti = np.zeros(len(list_of_all_arms))
        # self.mu_hat=np.ones(len(list_of_all_arms))
        self.Si = np.zeros(len(list_of_all_arms))


        # self.s=np.zeros(len(list_of_all_arms))
        self.matching_result = []
        self.mu_bar = np.ones(len(list_of_all_arms))

    def round(self, super_arm_to_pull):
        # in each round, a set of arms (called a super arm) are played together. (page 2 of the article)
        # In each round, one super arm S ∈ S is played and the outcomes of arms in S are revealed.
        self.matching_result = Matching(super_arm_to_pull)
        for arm in super_arm_to_pull:
            self.Ti[arm.i] += 1
            if arm.weight != 0:
                self.Si[arm.i] += 1

        return self.matching_result

    def opt(self):
        opt_mean = ((0.35 * 35 * 25) + (0.32 * 35 * 15) + (0.28 * 25 * 25) + (0.25 * 25 * 15)) / (61 * 39)
        opt_value=0
        for arms in self.matching_result.matched_list:
            opt_value=opt_value+self.mu_bar[arms.i]
        #return np.mean(self.mu_bar) * len(self.matching_result.matched_list)
        # return np.mean(self.mu_bar) * len(self.matching_result.matched_list)
        return opt_value