import json
import heapq
import numpy as np
import math
import sys
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


FOLLOW = .85
TRANSPORTATION = 1 - FOLLOW


class Node:
    """A simple Node class for representation in a web network"""
    def __init__(self, data):
        """Creates a node with a string of the data

        params:
            data: an integer or string corresponding identifying the node
            in the network (i.e. 357 corresponds to node
            number 357 in the network)
        """
        self.data = data
        self.hash = hash(str(self.data))
        self.ranking = 0
        self.cluster = -1

    def __hash__(self):
        """Overwritten hash method
        """
        return self.hash

    def __eq__(self, value):
        """Overwritten equality method
        params:
            value: a node to compare to 
        returns:
            boolean: if this node equals the other node
        """
        return self.data == value.data

    def __str__(self):
        """Overwritten string method

        returns:
            str: representation of the node's data
        """
        return str(self.data)

    def __repr__(self):
        """Overwritten representation method

        returns:
            str: representation of the class
        """
        return self.__str__()

    def __lt__(self, value):
        """Operator Overloading of less than

        params:
            value: a node to compare to 

        returns:
            boolean: if this node is less than value node
        """
        return self.ranking > value.ranking

    def __le__(self, value):
        """Operator Overloading of less than equal

        params:
            value: a node to compare to 

        returns:
            boolean: if this node is less than or equal to value node
        """
        return self.ranking >= value.ranking

    def __ge__(self, value):
        """Operator Overloading of greater than or equal

        params:
            value: a node to compare to 

        returns:
            boolean: if this node is greater than or equal to value node
        """
        return self.ranking <= value.ranking

    def __gt__(self, value):
        """Operator Overloading of greater than

        params:
            value: a node to compare to 

        returns:
            boolean: if this node is greater than value node
        """
        return self.ranking < value.ranking


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
        self.edges = {}
        self.mapping = []
        self.graph = sps.lil_matrix((num_nodes, num_nodes))
        self.size = num_nodes

    def __str__(self):
        """Overwritten string representation method"""
        return np.array2string(self.graph.toarray())

    def __repr__(self):
        """Overwritten string representation method"""
        return self.__str__()

    def _add_node(self, data):
        """Add a node to the graph

        params:
            data: str representing the node's website/data
        """
        node = Node(data)
        self.edges.update({node: []})

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

    def _create_pr_matrix(self):
        """
        Generate the page rank matrix
        """
        graph = self.graph.toarray()

        length = self.size

        self.matrix = (TRANSPORTATION/length)*np.ones((length, length))

        for i in range(length):
            col_sum = np.sum(graph[:, i])
            if col_sum != 0:
                self.matrix[:, i] = FOLLOW * (1/col_sum * graph[:, i]) + (TRANSPORTATION/length)*np.ones((length))
            else:
                self.matrix[:, i] = (1/length)*np.ones((length))

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
        self._create_pr_matrix()

        starting_vector = np.zeros((self.size, 1))
        starting_vector[0] = 1 

        px = starting_vector
        matrix = self.matrix
        for i in range(100):
            px = matrix.dot(px)

        self.ranking = (matrix).dot(px)

        ranking_list = self.ranking[np.argsort(-self.ranking.T)]

        return self.ranking, ranking_list

    def _create_diagonal_matrix(self):
        """
        Creates the diagonal matrix for fiedler clustering
        """
        print("Creating diagonal matrix...")
        self.diagonal = sps.lil_matrix((self.size, self.size))

        adjacency = self.graph
        size = self.size

        for i in range(size):
            row_sum = np.sum(adjacency[i, :])
            self.diagonal[i, i] = row_sum 

    def _cluster(self, fiedler_vector_list):
        print("Clustering...")
        clusters = {}

        size = self.size
        matrix = self.matrix
        ranking = self.ranking

        node_list = []

        for i in range(size):
            node = Node(i+1)
            node.ranking = ranking[i]

            binary = ['1' if fiedler_vector_list[x, i] > 0 else '0' for x in range(len(fiedler_vector_list))]
            cluster = int(''.join(binary), 2)
            node.cluster = cluster

            heapq.heappush(node_list, node)
            if cluster not in clusters:
                clusters.update({cluster: [node]})

            else:
                heap_list = clusters.get(cluster)
                heapq.heappush(heap_list, node)
                clusters.update({cluster: heap_list})

        return clusters, node_list

    def fiedler_clustering(self, num_clusters):
        """Runs Fiedler clustering on the network
        
        returns:
            cluster: dict mapping of cluster to nodes in the cluster
            node_list: max_heap of nodes based on ranking
        """
        self._create_diagonal_matrix()

        self.laplacian = self.diagonal - self.graph
        print("finding eigenvalues...")
        self.eigenvalues, self.eigenvectors = eigs(
            self.laplacian, k=num_clusters, which='SM')

        print("\neigenvalues found:")
        print(self.eigenvalues)
        print("\neigenvectors found:")
        print(self.eigenvectors)
        print()

        print("second smallest eigenvalue is {}\n".format(
            self.eigenvalues[1]))

        fiedler_vector_list = []
        for i in range(int(math.log2(num_clusters))):
            fiedler_vector = self.eigenvectors[:, i + 1]
            print("fiedler:\n")
            print(fiedler_vector)
            fiedler_vector_list.append(fiedler_vector)

        return self._cluster(np.array(fiedler_vector_list))


if __name__ == '__main__':
    #family guy graph
    graph = Graph(4)

    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(3, 2)
    graph.add_edge(3, 1)

    print("The graph is \n{}".format(graph))

    page_rank, ranking_list = graph.page_rank()

    clusters, node_list = graph.fiedler_clustering(2)

    print("diagonal matrix")
    print(graph.diagonal.toarray())
    print("adjacency matrix")
    print(graph.graph.toarray())
    print("laplacian matrix")
    print(graph.laplacian.toarray())
    print("\nPage rank ranking:\n{}".format(ranking_list))
    print("\nFielder Clustering returned")
    print(clusters)

    while node_list:
        node = heapq.heappop(node_list)
        print("node: {} ranking: {} cluster: {}".format(node, node.ranking, node.cluster))
    

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

    print("The graph is \n{}".format(graph))

    page_rank, ranking_list = graph.page_rank()

    clusters, node_list = graph.fiedler_clustering(2)


    print("diagonal matrix")
    print(graph.diagonal.toarray())
    print("adjacency matrix")
    print(graph.graph.toarray())
    print("laplacian matrix")
    print(graph.laplacian.toarray())
    print("\nPage rank ranking:\n{}".format(ranking_list))
    print("\nFielder Clustering returned")
    print(clusters)

    while node_list:
        node = heapq.heappop(node_list)
        print("node: {} ranking: {} cluster: {}".format(node, node.ranking, node.cluster))
    


