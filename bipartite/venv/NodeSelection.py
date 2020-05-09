import random
import numpy as np

class NodeSelector:
    def __init__(self, pool_of_nodes):
        self.pool_of_nodes = pool_of_nodes

    def selectNodes(self, num_of_subset):
        copy_pool = self.pool_of_nodes
        nodes_subset = []
        while len(copy_pool) > 0 :
            random.shuffle(copy_pool)
            returned_nodes = random.sample(copy_pool, num_of_subset)
            nodes_subset.append(returned_nodes)
            for nodes in returned_nodes:
                copy_pool.remove(nodes)
        return nodes_subset