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
        # self.super_arm_to_pull = EdgeSelector(list_of_all_arms).selectNodes(int(len(list_of_all_arms)/16))
        # self.pool_of_nodes = pool_of_nodes
        self.s=np.zeros(len(list_of_all_arms))
        self.reward_per_round = 0
        self.Ti = np.zeros(len(list_of_all_arms))
        self.mu_hat=np.ones(len(list_of_all_arms))
        self.mu_bar = np.ones(len(list_of_all_arms))  # first time choosing makes the exploration better

        #self.s=np.zeros(len(list_of_all_arms))
        self.matching_result = []
        #self.mu_bar = np.ones(len(list_of_all_arms))

    def round(self, pulled_super_arm):
        # in each round, a set of arms (called a super arm) are played together. (page 2 of the article)
        # In each round, one super arm S ∈ S is played and the outcomes of arms in S are revealed.

        self.matching_result = oracle.Matching(pulled_super_arm)

        for arm in pulled_super_arm:
            self.Ti[arm.i] += 1

        for arm in self.matching_result.matched_list:
            # #     ##### note: i think the reason that we get some mu_bar goes to zeros is that from bernouli distribution
            # #         # the weight of some arms are already assigned to zero.
            self.s[arm.i] += 1  ### try this s ignoring when the an arm is not in matching (just consider those in the matching)
            self.mu_hat[arm.i] = (arm.constant + ((self.s[arm.i] - 1) * self.mu_hat[arm.i])) / self.s[
                arm.i]  ### he
        return self.matching_result

    ### what are we maximizing ???? and where are we using this?
    def opt(self,mean):
       # opt_reward = np.max(self.matching_result.matched_list)
        # return self.opt_reward*len(self.matching_result.matched_list)
        return mean * len(self.matching_result.matched_list)

    # def opt(self):
    # weights_list = self.matching_result.Edge.weights  # of course this line is not correct in programming
    #   # PoV but we should be able to  extract and put weights of the matched edges in a list
    #   weight_list=self.matching_result
    #  self.opt_reward.append(n)
    # self.opt_reward(p.max(weight_list) * len(weight_list))
    # weight_list=xx.
    # self.opt_reward.append(n)
    # return self.opt_reward # this is the best possible reward that can be taken from
