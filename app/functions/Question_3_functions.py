from app.functions.Tools import *


def get_edges_cascade(pull_node, list_edges_activated):
    list_new_nodes = [pull_node]
    list_cascade_edges = []
    while(len(list_new_nodes) != 0):
        tamp = []
        for node_head in list_new_nodes:
            for edge in list_edges_activated:
                if node_head == edge[0]:
                    list_cascade_edges.append(edge)
                    tamp.append(edge[1])
        list_new_nodes = tamp
    return list_cascade_edges


    # Find nodes who activate an A node.
def credit_nodes(list_A_node_activated, list_edges_activated, pulled_super_arm):
    list_credit = [0]*len(pulled_super_arm)
    for i in range(len(pulled_super_arm)):
        pull_node = pulled_super_arm[i]

        if pull_node in list_A_node_activated:
# If it is an initial node, we add 1 to his credit because it is possible that it has no edges.
            list_credit[i] += 1

        tamp = get_edges_cascade(pull_node, list_edges_activated)
        if len(tamp) != 0:
            list_cascade_node = [num[1] for num in tamp] # get the inherited nodes from the cascade
            for node in list_cascade_node:
                if node in list_A_node_activated: # add a credit if it activates an A node.
                    list_credit[i] +=1

    return list_credit
