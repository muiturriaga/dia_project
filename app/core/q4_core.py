# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:37:01 2020
@author: Danial
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as sb
import sys, os

# to import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import bipartite.cmab_env as cmab
import bipartite.learner as learner
import bipartite.oracle as oracle
import bipartite.make_bipartite as m_b

n_of_nodes = 100

T = 20

n_experiments = 30

reward_per_experiment = []

list_nodes = []
for i in range(n_of_nodes):
    # node_type=np.random.choice(list_types,1,p=[0.25,0.25,0.25,0.25])
    # the node types are taken from pool of nodes and the activated nodes. Here I assumed some fixed nodes.
    if i <= 35:
        node_type = 'A'
    if 35 < i <= 65:
        node_type = 'B'
    if 65 < i <= 85:
        node_type = 'C'
    if i > 85:
        node_type = 'D'

    list_nodes.append(oracle.Nodes(num=i, node_type=node_type))
opt_mean=((0.35*35*25)+(0.32*35*15)+(0.28*25*25)+(0.25*25*15))/(61*39)
# at first we have to give a matrix of P=[p(0,71), p(1,71),p(2,71),...] this probability matrix P is self determined.
# then the weight of each edge is P[nodel.type,node2.type]*Bernoulli(p(#L,#R)) where p(#L,#R)s are the elements of P
# one time this information is given. obviously we hav ematrix P stroed. then the algorithm's aim is to learn these probs.
# I just wrote this to say that we have to add a P variable to our matching algorithm. In
# other cases we set P as zero. but for the first time we have to take this P into considerations.
flag = 0
mu_bar_stored = np.ones(34*66)

prob = m_b.Make_Bipartite(list_nodes)
#first_try_prob_matrix=np.zeros(61*39)


prob.calculate_probability()

p_temp = np.array(prob.prob_matrix)
first_prob_matrix=p_temp.reshape(34*66 , 1)

prob.set_p(mu_bar_stored)
list_of_edges = prob.make_bipartite()
rewards_per_experiment = []
cucb_rewards_per_experiment = []

env = cmab.CMAB_Environment(list_of_edges)
opt = []
MSE = []
t_total = 0
for e in range(0, n_experiments):
    list_of_edges = prob.make_bipartite()

    cmab_learner = learner.Learner(list_of_edges, env.Ti, env.mu_hat)
    cmab_learner.mu_bar = env.mu_bar

    opt_per_experiment = []
    for t in range(0, T):
        t_total += 1
        cmab_learner.t_total = t_total
        pulled_super_arm = cmab_learner.pull_super_arm()
        results = env.round(pulled_super_arm)
        cmab_learner.mu_hat = env.mu_hat
        cmab_learner.update(results)# when do we use cmab_learner.pull_super_arm
        opt_per_experiment.append(env.opt(opt_mean))
        env.mu_bar = cmab_learner.mu_bar


    print(cmab_learner.mu_bar,"\n",e)

    opt.append(opt_per_experiment)
    cucb_rewards_per_experiment.append(cmab_learner.collected_rewards)
    prob.set_p(cmab_learner.mu_bar)
    MSE.append(np.mean((((env.s + 0.0001)/ (env.Ti + 0.0001)) - first_prob_matrix)**2))

#print((cmab_learner.s/300)-first_prob_matrix)
# print(abs(cmab_learner.mu_bar - prob.return_edge_constant()), "/n")

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
# opt=env.opt()
# opt_reward=env.opt()
ooo = []
for i in range(n_experiments):
    ooo.append(np.mean([abs(a - b) for a, b in zip(opt[i], cucb_rewards_per_experiment[i])]))

plt.plot(np.cumsum(ooo, axis=0), 'g')
plt.show()

plt.figure(1)
plt.xlabel("Experiment")
plt.ylabel("MSE")

plt.yscale("log")
plt.plot(MSE, 'r')
plt.show()
