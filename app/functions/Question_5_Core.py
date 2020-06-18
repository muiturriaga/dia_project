import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as sb
from Question_5_functions import *
from Tools import *

cum_budget = 100
discretized_vector = []
activated_nodes_A = []
activated_nodes_B = []
activated_nodes_C = []
activated_nodes = []

num_of_nodes = 100

Edges_info, Nodes_info, Color_map = create_graph(['A'], ['B','C'], num_of_nodes, proba_edges) # A nodes are in red.

### discretization
discretization_step = 10;
for i in range(int(cum_budget/discretization_step)+1):
    discretized_vector.append(i*discretization_step)

# main cycle, enumeration
for i in range(int(cum_budget^2)):
    # [A B C], [] if skipped
    budget_vector = budget_allocation(discretized_vector, i, cum_budget)
    #print(budget_vector)

    if (len(budget_vector) > 0):
        if (budget_vector[0] > 0):
            activated_nodes_A = list_activated_nodes(budget_vector, 'A', Edges_info[1], Nodes_info[2])
        if (budget_vector[1] > 0):
            activated_nodes_B = list_activated_nodes(budget_vector, 'B', Edges_info[1], Nodes_info[2])
        if (budget_vector[2] > 0):
            activated_nodes_C = list_activated_nodes(budget_vector, 'C', Edges_info[1], Nodes_info[2])

        activated_nodes = activated_nodes_A + activated_nodes_B + activated_nodes_C
        #print(activated_nodes)

        prob = Make_Bipartite(Nodes_info[2])

budget_vector = [100, 0, 0] #last case not covered in the cycle

activated_nodes_A = list_activated_nodes(budget_vector, 'A', Edges_info[1], Nodes_info[2])
#print(activated_nodes)

"""
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

    list_nodes.append(Nodes(num=i, node_type=node_type))



A_aloc=40
B_aloc=30
C_aloc=30

# we wanto introduce the best seeds of type A along with their edges infomation

# prob (weights)= similatiry measure (as weight)


list_of_best_nodes_A= func(budget,message_type_A,social_network)
list_of_best_nodes_B= func(budget,message_type_B,social_network)
list_of_best_nodes= func(budget,message_type_c,social_network)
# create some nodes of type D with features.

#includig the features, and # identier.







prob = Make_Bipartite(list_nodes)
#first_try_prob_matrix=np.zeros(61*39)

prob.calculate_probability()

prob.set_p(mu_bar_stored)
list_of_edges = prob.make_bipartite()

matching_result=Matching(liset_of_edges)





end """
