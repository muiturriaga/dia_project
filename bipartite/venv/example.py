import numpy as np
from Learner import *
from matching import *
n_of_nodes = 160
mu, sigma = 0, 0.1  # mean and standard deviation

T = 300

n_experiments = 1

reward_per_experiment = []

list_nodes = []
for i in range(n_of_nodes):
    # node_type=np.random.choice(list_types,1,p=[0.25,0.25,0.25,0.25])
    # the node types are taken from pool of nodes and the activated nodes. Here I assumed some fixed nodes.
    if i <= 30:
        node_type = 'A'
    if 30 < i <= 70:
        node_type = 'B'
    if 70 < i <= 110:
        node_type = 'C'
    if i > 110:
        node_type = 'D'

    list_nodes.append(Nodes(num=i, node_type=node_type))
#     list of weights = matching
#      opt weight prob
rewards_per_experiment = []
for e in range(n_experiments):
    env = CMAB_Environment(list_nodes)
    learner = Learner(n_of_nodes)
    for t in range(T):
        # TS Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        learner.update(pulled_arm, reward)

    rewards_per_experiment.append(ts_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

regrets = opt - rewards_per_experiment
