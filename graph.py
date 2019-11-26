import json
import heapq
import numpy as np
import math
import sys
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
from sklearn.cluster import MiniBatchKMeans


FOLLOW = .85
TRANSPORTATION = 1 - FOLLOW

class Graph:
    """A simple representation of a directed graph for a web network

    methods:
        add_edge: adds an edge from a node to a new node
        page_rank: runs the Page Rank algorithm on the graph's matrix
        fiedler_clustering: cluster the network using spectral clustering

    attributes:
        edges: a map denoting edges from the key to values
        mapping: sorted order of nodes
    """
    def __init__(self, num_nodes):
        """Initializes an empty web network"""
        self.graph = sps.lil_matrix((num_nodes, num_nodes))
        self.size = num_nodes

    def __str__(self):
        """Overwritten string representation method"""
        return np.array2string(self.graph.toarray())

    def __repr__(self):
        """Overwritten string representation method"""
        return self.__str__()

    def add_edge(self, from_website, to_link):
        """Add an edge from_website to to_link
        If either of the nodes are already not in the network, add them

        params:
            from_website: str from the node
            to_link: str representing the edge to the other node 

        returns:
            none

        raises:
            ValueError: if the edge has already been created
        """
        self.graph[to_link - 1, from_website - 1] = 1

    def _print_matrix(self):
        """
        Prints the Page Rank matrix
        """
        print(self.matrix)

    def page_rank(self):
        """Runs the page rank algorithm

        returns:
            a vector corresponding to the steady state vecto
        """
        print("Generating the matrix...")
        G = self.graph
        p = FOLLOW
        column_sum = np.sum(G, axis=0, dtype=np.float64)
    
        n = self.graph.shape[0]

        D = sps.lil_matrix((n, n))
        D.setdiag(np.divide(1.,column_sum, where=column_sum != 0, out=np.zeros_like(column_sum)).reshape(-1, 1))
        self.diagonal = D
        print("created diagonal")
        e = np.ones((n, 1))
        I = sps.eye(n)
        x = sps.linalg.spsolve((I - p*G*D), e)
        x = x/np.sum(x)

        self.page_rank = x

        return x

    def fiedler_clustering(self, num_clusters):
        """Runs Fiedler clustering on the network
        
        returns:
            cluster: dict mapping of cluster to nodes in the cluster
            node_list: max_heap of nodes based on ranking
        """

        clusters = MiniBatchKMeans(n_clusters=num_clusters).fit_predict(self.graph)

        return clusters


if __name__ == '__main__':
    #family guy graph
    graph = Graph(4)

    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(3, 2)
    graph.add_edge(3, 1)

    page_rank = graph.page_rank()

    clusters = graph.fiedler_clustering(2)

    print("adjacency matrix")
    print(graph.graph.toarray())
    print(page_rank)
    print("\nPage rank ranking:\n{}".format(np.argsort(-page_rank.T)))
    print("\nFielder Clustering returned")
    print(clusters)

    for node in np.argsort(-page_rank.T):
        print("node: {} ranking: {} cluster: {}".format(node + 1, page_rank[node], clusters[node]))
    

    # # #testing fielder clustering
    graph = Graph(7)

    graph.add_edge(1, 4)
    graph.add_edge(1, 6)
    graph.add_edge(4, 1)
    graph.add_edge(4, 6)
    graph.add_edge(6, 1)
    graph.add_edge(6, 4)
    graph.add_edge(4, 2)
    graph.add_edge(2, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 2)
    graph.add_edge(2, 7)
    graph.add_edge(5, 7)
    graph.add_edge(7, 5)
    graph.add_edge(7, 2)
    graph.add_edge(5, 3)
    graph.add_edge(7, 3)
    graph.add_edge(3, 5)
    graph.add_edge(3, 7)

    page_rank = graph.page_rank()

    clusters = graph.fiedler_clustering(2)


    print("adjacency matrix")
    print(graph.graph.toarray())
    print(page_rank)
    print("\nPage rank ranking:\n{}".format(np.argsort(-page_rank.T)))
    print("\nFielder Clustering returned")
    print(clusters)

    for node in np.argsort(-page_rank.T):
        print("node: {} ranking: {} cluster: {}".format(node + 1, page_rank[node], clusters[node]))

    


