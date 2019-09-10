import json
import heapq
import numpy as np
import math

TOTAL_TELEPORTATION_RATE = .15
TOTAL_FOLLOW_LINK_RATE = .85


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
    def __init__(self):
        """Initializes an empty web network"""
        self.edges = {}
        self.mapping = []

    def __str__(self):
        """Overwritten string representation method"""
        str = ""
        for k in self.edges.keys():
            str += "({}, {})\n".format(k, self.edges.get(k))
        return str

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
        node = Node(to_link)
        from_node = Node(from_website)

        if from_node not in self.edges:
            self._add_node(from_website)

        if node not in self.edges:
            self._add_node(to_link)

        edges = self.edges.get(from_node)
        if node not in edges:
            edges.append(node)
        else:
            raise ValueError("Edge already created from {} to {}".format(
                from_website, to_link))

        self.edges.update({from_node: edges})

    def _create_pr_matrix(self):
        """Generate the page rank matrix"""

        self.matrix = np.zeros((self.size, self.size))

        for key in self.edges.keys():
            edges = self.edges.get(key)

            column = key.data - 1

            if len(edges) == 0:
                self.matrix[:, column] = 1/self.size

            else:
                teleportation_rate = TOTAL_TELEPORTATION_RATE/self.size
                link_rate = TOTAL_FOLLOW_LINK_RATE/len(edges)
                
                for i, node in enumerate(self.edges.keys()):
                    if node in edges:
                        self.matrix[node.data - 1, column] = teleportation_rate + link_rate
                    else:
                        self.matrix[node.data - 1, column] = teleportation_rate

    def _print_matrix(self):
        """Prints the Page Rank matrix"""
        matrix_str = ""
        for node in sorted(self.edges.keys()):
            matrix_str += str(node) + "\t"
        matrix_str += "\n"

        matrix_str += str(self.matrix)

        print(matrix_str)

    def _set_mapping(self):
        """Creates a list of nodes in sorted order for easier retrieval later"""
        num_keys = len(self.edges.keys())
        self.mapping = [None] * num_keys
        for node in self.edges.keys():
            self.mapping[node.data - 1] = node

    def page_rank(self):
        """Runs the page rank algorithm

        returns:
            a vector corresponding to the steady state vecto
        """
        self._create_pr_matrix()
        print("Generated the matrix: ")
        self._print_matrix()

        starting_vector = np.zeros((self.size, 1))
        starting_vector[0] = 1 

        ranking = np.linalg.matrix_power(self.matrix, 100).dot(starting_vector)

        ranking_list = np.array(self.mapping)[np.argsort(-ranking.T)]

        #iterate over ranking, add to node
        node_list = []
        for i in range(ranking.shape[0]):
            node = self.mapping[i]
            node.ranking = ranking[i]
            heapq.heappush(node_list, node)

        return node_list, ranking, ranking_list

    def _create_adjacency_matrix(self):
        """Creates the adjacency matrix for fiedler clustering"""
        self.adjacency = np.zeros((self.size, self.size))

        for key, edges in self.edges.items():
            row = key.data - 1

            for edge in edges:
                column = edge.data - 1
                self.adjacency[row, column] = 1

    def _create_diagonal_matrix(self):
        """Creates the diagonal matrix for fiedler clustering"""
        self.diagonal = np.zeros((self.size, self.size))
        for i in range(self.size):
            sum = np.sum(self.adjacency[i])
            self.diagonal[i, i] = sum

    def _cluster(self, fiedler_vector_list):
        clusters = {}

        for i in range(len(self.edges.keys())):
            node = self.mapping[i]
            binary = ['1' if fiedler_vector_list[x, i] > 0 else '0' for x in range(len(fiedler_vector_list))]
            cluster = int(''.join(binary), 2)
            node.cluster = cluster
            if cluster not in clusters:
                clusters.update({cluster: [node]})

            else:
                heap_list = clusters.get(cluster)
                heapq.heappush(heap_list, node)
                clusters.update({cluster: heap_list})

        return clusters


    def fiedler_clustering(self, num_clusters):
        """Runs Fiedler clustering on the network
        
        returns:
            cluster: dict mapping of cluster to nodes in the cluster
        """
        self._set_mapping()
        self.size = len(self.edges.keys())

        self._create_adjacency_matrix()
        self._create_diagonal_matrix()

        self.laplacian = self.diagonal - self.adjacency

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.laplacian)

        print("\neigenvalues:")
        print(self.eigenvalues)
        print("\neigenvectors:")
        print(self.eigenvectors)
        print()

        self.sorted_eigenvalues = np.argsort(self.eigenvalues)
        print("second smallest eigenvalue is {}\n".format(
            self.sorted_eigenvalues[1]))

        fiedler_vector_list = []
        for i in range(int(math.log2(num_clusters))):
            fiedler_vector = self.eigenvectors[:, self.sorted_eigenvalues[i + 1]]
            print("fiedler:\n")
            print(fiedler_vector)
            fiedler_vector_list.append(fiedler_vector)

        clusters = self._cluster(np.array(fiedler_vector_list))

        return clusters


if __name__ == '__main__':
    #family guy graph
    graph = Graph()

    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(3, 2)
    graph.add_edge(3, 1)

    print("The graph is \n{}".format(graph))

    clusters = graph.fiedler_clustering(4)

    node_list, page_rank, ranking_list = graph.page_rank()

    print("diagonal matrix")
    print(graph.diagonal)
    print("adjacency matrix")
    print(graph.adjacency)
    print("laplacian matrix")
    print(graph.laplacian)
    print("\nPage rank ranking:\n{}".format(ranking_list))
    print(page_rank)
    print("\nFielder Clustering returned")
    print(clusters)

    while node_list:
        node = heapq.heappop(node_list)
        print("node: {} ranking: {} cluster: {}".format(node, node.ranking, node.cluster))
    

    # #testing fielder clustering
    graph = Graph()

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

    clusters = graph.fiedler_clustering(4)

    node_list, page_rank, ranking_list = graph.page_rank()

    print("diagonal matrix")
    print(graph.diagonal)
    print("adjacency matrix")
    print(graph.adjacency)
    print("laplacian matrix")
    print(graph.laplacian)
    print("\nPage rank ranking:\n{}".format(ranking_list))
    print("\nFielder Clustering returned")
    print(clusters)

    while node_list:
        node = heapq.heappop(node_list)
        print("node: {} ranking: {} cluster: {}".format(node, node.ranking, node.cluster))
    


