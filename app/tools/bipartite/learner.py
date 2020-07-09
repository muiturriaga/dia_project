import numpy as np
from EdgeSelector import *
from CMAB_Environment import *
import heapq


class Learner:
    def __init__(self, list_of_arms, Ti, mu_bar):
        self.reward_list = []
        self.list_of_arms = list_of_arms
        self.t = 1
        self.collected_rewards = []
        self.X = {}
        self.mu_hat = np.ones(len(list_of_arms))
        self.Ti = Ti
        self.it_is_first_time = True
        self.t_total = 0
        self.ucbs = np.ones(len(list_of_arms))
        self.mu_bar = mu_bar
        self.s = np.zeros(len(list_of_arms))
        self.number_of_times_arm_pulled = np.zeros(len(list_of_arms))

    # def computeCucb(self):
    # for arm in self.list_of_arms:

    #  if abs(self.mu_bar[arm.i] - self.list_of_arms[arm.i].constant) < 0.05:

    #    else:
    #         self.ucbs[arm.i]=se;

    def pull_super_arm(self):
        #  self.ucbs=self.mu_bar
        # here we have u-bars we just need to write that and it will compute for us what to play
        selected_list = []
        # selected_list_indicies2 = sorted(range(len(self.ucbs)), key=lambda i: self.ucbs[i], reverse=True)[:500]
        selected_list_indicies_sorted = np.argsort(self.mu_bar)[::-1]

        for index in range(600):
            ### is this for the first time overal, or each first time a learern is created?
            arm = self.list_of_arms[selected_list_indicies_sorted[index]]
            if self.it_is_first_time:
                random.shuffle(self.list_of_arms)
                self.it_is_first_time = False

            # selected_list.append(self.list_of_arms[selected_list_indicies_sorted[index]])
            selected_list.append(arm)
        return selected_list  # note that here there are two options

    # first option is that ucbs will send the number associated to the edges (pairs of nodes) (more compatible with
    # definition of arm) second option is that ucbs will send the nodes that have been involved in matchings that
    # were involved (easier to implement) we can also make a matrix to of the left-right nodes and their weights (
    # weight means no conection) and use that easily

    def update_estimation(self, results):
        result_of_played_arm = results
        for arms in result_of_played_arm.matched_list:
            self.s[arms.i] += 1
            ### try this s ignoring when the an arm is not in matching (just consider those in the matching)
            self.mu_hat[arms.i] = (arms.constant + ((self.s[arms.i] - 1) * self.mu_hat[arms.i])) / self.s[
                arms.i]  ### he
            #  self.mu_hat[arm.i]=(arm.constant + self.mu_hat[arm.i])/2
            adjustment_term = np.sqrt(3 * np.log(self.t_total) / (2 * (self.Ti[arms.i]) + 0.0001))

            self.mu_bar[arms.i] = self.mu_hat[arms.i] + adjustment_term
            self.mu_bar[arms.i] = min(self.mu_bar[arms.i], 1)  # clipping the mean value to 1

    def update(self, results):
        self.t += 1
        result_of_played_arms = results
        reward_of_matching_list = result_of_played_arms.weight_of_matched_list()
        self.collected_rewards.append(reward_of_matching_list)
        self.update_estimation(results)
        # t is the number of rounds so far,
        # Ti is the number of times an arm (an edge, or a pair of nodes) is activated so far (from the beginning)
        # s is the number of times an arm is activated within one episode of experiment (which depends
        # the number of trials until the end of horizon) after that it should be set to zero in application
        # Mu_hat is the empirical expected value of the arm
        # In each round one super arm is played and the out- comes of all arms in the super arm  are revealed
        # that is the reason we go through each arm (but there is a problem with Ti[arm] an arm might be played several times but it might not be
        # in the optimal graph so I prefer to put it in the CMAB_Env to have all the out comes of an arm
        # The reward Rt (S ) might be as simple as a summation of the outcomes of the arms in
        # So I defined the reward for it as the summation of all weights
