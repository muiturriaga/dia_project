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


def estimate_node_message(dataset, message, list_nodes):
    n_nodes = len(list_nodes)
    list_credit = np.zeros(n_nodes)
    time_credit = [0]*(len(dataset)+1)
    i = 1

    for episode in dataset:
        credit = 0
        # What are the initial nodes ?
        idx_initial = np.argwhere(episode[0] == 1).reshape(-1)
        for index_steps in range(len(episode)):
            # What are the nodes activated at each step ?
            idx_w_active = np.argwhere(episode[index_steps] == 1).reshape(-1)
            for num_node in idx_w_active:
                # If they are of type 'message', we give them credits. More a node have credits, more it is able to activate message_nodes.
                if list_nodes[num_node].special_feature == message:
                    credit += 1
        # Upgrade credits of initial nodes.
        for node_index in idx_initial:
            list_credit[node_index] += credit/len(dataset)

        time_credit[i] = credit/len(dataset) + time_credit[i-1]
        i +=1
    return [list_credit, time_credit] # Return the number of nodes approximately activated at each episode.



