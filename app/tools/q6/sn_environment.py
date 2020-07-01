import sys
import os
import random
import numpy as np

# import tools
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'tools')))
sys.path.append(path2add)
import q5.q5_tools as q5_tools
import bipartite.cmab_env as cmab
import bipartite.learner as learner
import bipartite.oracle as oracle
import bipartite.make_bipartite as m_b

class SnEnvironment:
    def __init__(self, seeds_a, seeds_b, seeds_c, nodes_d, nodes_info, edges_info):
        self.seeds_A = seeds_a
        self.seeds_B = seeds_b
        self.seeds_C = seeds_c
        self.nodes_d = nodes_d
        self.nodes_info = nodes_info
        self.edges_info = edges_info

    # HAS TO RETURN VALUE OF MATCHING.
    # The problem I see -> at the moment, you create probabilities to estimate.
    # Substitute the part of creation of probs with features similarities
    def round(self, budget_alloc):
        starting_nodes_a = random.sample(self.seeds_A, budget_alloc[0])
        starting_nodes_b = random.sample(self.seeds_B, budget_alloc[1])
        starting_nodes_c = random.sample(self.seeds_C, budget_alloc[2])
        # print("Starting nodes ", starting_nodes_A, starting_nodes_B, starting_nodes_C)

        activated_nodes = []  # will contain all activated nodes

        # observe, given seeds for each type, the activation of nodes in the SN by spreading the message
        ret = q5_tools.list_activated_nodes(starting_nodes_a, budget_alloc[0], 'A', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)
        ret = q5_tools.list_activated_nodes(starting_nodes_b, budget_alloc[1], 'B', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)
        ret = q5_tools.list_activated_nodes(starting_nodes_c, budget_alloc[2], 'C', self.edges_info[1], self.nodes_info[1])
        if ret: activated_nodes.extend(ret)

        activated_nodes = q5_tools.convert_nodes(activated_nodes)  # convert from SN node class to Oracle node class
        activated_nodes.extend(self.nodes_d)  # add nodes D

        T = 20
        n_experiments = 30
        reward_per_experiment = []

        opt_mean = ((0.35*35*25)+(0.32*35*15)+(0.28*25*25)+(0.25*25*15))/(61*39)
        # at first we have to give a matrix of P=[p(0,71), p(1,71),p(2,71),...] this probability matrix P is self determined.
        # then the weight of each edge is P[nodel.type,node2.type]*Bernoulli(p(#L,#R)) where p(#L,#R)s are the elements of P
        # one time this information is given. obviously we hav ematrix P stroed. then the algorithm's aim is to learn these probs.
        # I just wrote this to say that we have to add a P variable to our matching algorithm. In
        # other cases we set P as zero. but for the first time we have to take this P into considerations.
        flag = 0
        mu_bar_stored = np.ones(34*66)

        prob = m_b.Make_Bipartite(activated_nodes)
        prob.calculate_probability()

        p_temp = np.array(prob.prob_matrix)
        first_prob_matrix = p_temp.reshape(34*66, 1)

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
                cmab_learner.update(results)  # when do we use cmab_learner.pull_super_arm
                opt_per_experiment.append(env.opt(opt_mean))
                env.mu_bar = cmab_learner.mu_bar

            print(cmab_learner.mu_bar,"\n",e)

            opt.append(opt_per_experiment)
            cucb_rewards_per_experiment.append(cmab_learner.collected_rewards)
            prob.set_p(cmab_learner.mu_bar)
            MSE.append(np.mean((((env.s + 0.0001)/ (env.Ti + 0.0001)) - first_prob_matrix)**2))

