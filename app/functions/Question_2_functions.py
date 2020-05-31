from .Tools import *


## Question 2.

# Find episode with initial nodes in order to verify the results
def find_episode(list_num_nodes, dataset):
    initial_nodes = [0]*len(dataset[0][0])
    for i in list_num_nodes:
        initial_nodes[i] = 1
    list_episode = []
    for i in range(len(dataset)):
        if list(dataset[i][0]) == initial_nodes:
            list_episode.append(i)

    sub_dataset = [Dataset[i] for i in list_episode]
    return list_episode,sub_dataset

# Given a dataset and a message, return the reward of initial_nodes. Here the reward is the same for all the nodes. We can change it by tracing the father of A nodes as in Question.3.

def calculate_minimum_rep(eps, delta, card_seeds = int):
    if card_seeds == 1:
        return (1/(eps)**2*np.log(2)*np.log(1/delta))
    else :
        return (1/(eps)**2*np.log(card_seeds)*np.log(1/delta))