'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import random
import csv
from sklearn.preprocessing import normalize


class Graph:
    def __init__(self, nx_G, is_directed, p, q, adj=None):
        """

        :param nx_G:
        :param is_directed:
        :param p:
        :param q:
        :param adj: A scipy sparse adjacency matrix
        :return:
        """
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.adj = adj

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def uniform_walk(self, walk_length):
        """
        simulate random walk starting at each node of length walk_length. This method assumes unweighted edges.
        For weighted edges use node2vecwalk
        :param walk_length:
        :return:
        """
        pass

    def output_walks(self, num_walks, walk_length, path):
        """
        write the random walks to file. This is necessary for large files where memory overflows
        :param num_walks: the number of random walks commencing at each vertex
        :param walk_length: the distance to walk
        :param path: path of output file
        :return: None
        """
        with open(path, 'wb') as f:
            writer = csv.writer(f)
            nodes = list(self.G.nodes())
            print 'Walk iteration:'
            for walk_iter in range(num_walks):
                print str(walk_iter + 1), '/', str(num_walks)
                random.shuffle(nodes)  # why is this necessary? It improves the speed of downstream SGD convergence
                for count, node in enumerate(nodes):
                    walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
                    writer.writerow(walk)
                    if count % 1000 == 0:
                        f.flush()

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print 'Walk iteration:'
        for walk_iter in range(num_walks):
            print str(walk_iter + 1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
                walks.append(walk)
        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge. This is the only use of the parameters p and q.
        It is called when initialising the alias table once for every edge
        :param src the source vertex of this edge
        :param dst the destination vertex of this edge
        :Return
        """
        G = self.G
        p = self.p
        q = self.q

        # we divide by these things.
        assert isinstance(p, float)
        assert isinstance(q, float)

        unnormalized_probs = []
        #  This loop uses two parameters p and q to bias the random walk
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:  # this node is the source
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):  # this node is distance 1 from the source
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:  # this node is distance 2 from the source
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        norm_consts = []
        for idx, node in enumerate(G.nodes()):
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            # norm_consts.append(norm_const)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        # deltas = []
        # edge_counts = np.array(self.adj.sum(axis=1)).flatten()
        # for i in range(len(edge_counts)):
        #     if edge_counts[i] != norm_consts[i]:
        #         deltas.append(edge_counts[i] - norm_consts[i])
        # print len(deltas)
        # print deltas
        #
        # adj_norm = normalize(self.adj, norm='l1', axis=1)


        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                # add edge in the other direction
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        # get the index of a probability less than 1/K
        small = smaller.pop()
        # get the index of a probability greater than 1/K
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
