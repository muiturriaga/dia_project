# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:37:01 2020

@author: Danial
"""

import numpy as np
from CMAB_Environment import *
from Learner import Learner
from Oracle import *
from Make_Bipartite import *
import matplotlib.pyplot as plt
import matplotlib.scale as sb

n_of_nodes = 100

T = 30

n_experiments = 300

reward_per_experiment = []

list_nodes = []
for i in range(n_of_nodes):
    # the node types are taken from pool of nodes and the activated nodes. Here I assumed some fixed nodes.
    if i <= 35:
        node_type = 'A'
    if 35 < i <= 65:
        node_type = 'B'
    if 65 < i <= 85:
        node_type = 'C'
    if i > 85:
        node_type = 'D'

    list_nodes.append(Nodes(num=i, node_type=node_type))
# then the weight of each edge is P[nodel.type,node2.type]*Bernoulli(p(#L,#R)) where p(#L,#R)s are the elements of P
# one time this information is given. obviously we hav ematrix P stroed. then the algorithm's aim is to learn these probs.
flag = 0
mu_bar_stored = np.ones(34 * 66)
# opt_mean=((0.35*35*25)+(0.32*35*15)+(0.28*25*25)+(0.25*25*15))/(61*39)

prob = Make_Bipartite(list_nodes)
# first_try_prob_matrix=np.zeros(61*39)


prob.calculate_probability()
p_temp = np.array(prob.prob_matrix)
first_prob_matrix = p_temp.reshape(34 * 66, 1)

prob.set_p(mu_bar_stored)
list_of_edges = prob.make_bipartite()
rewards_per_experiment = []
cucb_rewards_per_experiment = []

opt = []
MSE = []
env = CMAB_Environment(list_of_edges)
for e in range(0, n_experiments):

    cmab_learner = Learner(list_of_edges, env.Ti, env.mu_bar)
    t_total = 0
    opt_per_experiment = []

    for t in range(0, T):
        t_total += 1
        cmab_learner.t_total = t_total
        super_arm_to_pull = cmab_learner.pull_super_arm()
        results = env.round(super_arm_to_pull)

        cmab_learner.update(results)  # when do we use cmab_learner.pull_super_arm
        # env.mu_bar = cmab_learner.mu_bar
        # env.mu_hat=cmab_learner.mu_hat

        opt_per_experiment.append(env.opt())
        prob.set_p(env.mu_bar)
        list_of_edges = prob.make_bipartite()
        env.list_of_all_arms = list_of_edges

        results.weight_of_matched_list()

    print(env.mu_bar, "\n", e)
    # print(env.Si,'\n')# Si is the number of times and arm is failed to be pulled for it's weight is =0
    # print(env.Ti,'\n')# Ti is the number of times an arm is about to be pulled

    opt.append(opt_per_experiment)
    cucb_rewards_per_experiment.append(cmab_learner.collected_rewards)
    MSE.append(np.mean((((env.Si + 1) / (env.Ti + 1)) - first_prob_matrix) ** 2))

# print((cmab_learner.s/300)-first_prob_matrix)
# print(abs(cmab_learner.mu_bar - prob.return_edge_constant()), "/n")


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
# opt=env.opt()
# opt_reward=env.opt()
ooo = []
for i in range(n_experiments):
    ooo.append(np.mean([b - a for a, b in zip(opt[i], cucb_rewards_per_experiment[i])]))

plt.plot(np.cumsum(ooo, axis=0), 'g')
plt.show()

plt.figure(1)
plt.xlabel("Experiment")
plt.ylabel("MSE")
plt.yscale("log")
plt.plot(MSE, 'r')
plt.show()
# Instead, we assume the availability of an offline computation oracle that takes such knowledge as well as the
# expectations of outcomes of all arms as input and computes the optimal super arm with respect to the input. do you
# know how to write oracle?????
print("the estimated probabilities are:\n ")
print(env.Si / env.Ti, '\n')

print("the true probabilites are\n")
print(first_prob_matrix)