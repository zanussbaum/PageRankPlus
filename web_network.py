import pickle
import argparse
import math
import heapq
from graph import Graph, Node


def save_graph(filename):
    graph = Graph()

    count = 0
    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith("#"):
                split = line.replace(" ", "").split()
                from_node = int(split[0])
                to_node = int(split[1])

                graph.add_edge(from_node, to_node)

                count += 1

                if count % 1000 == 0:
                    print("added {} to {}. Have added {} edges".format(
                        from_node, to_node, count))

    pickle_name = filename.replace(".txt", "")

    pickle_file = open(pickle_name, 'ab')

    pickle.dump(graph, pickle_file)

    pickle_file.close()

    print("saved the graph")

    return graph


def load_graph(filename):
    print("loading the graph...")
    pickle_name = filename.replace(".txt", "")

    pickle_file = open(pickle_name, 'rb')

    graph = pickle.load(pickle_file)

    pickle_file.close()

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load and save the graph")

    parser.add_argument('--file', help='File to use', required=True)
    parser.add_argument('--save', action='store_true', help='Save the graph')

    args = parser.parse_args()

    filename = args.file
    save = args.save

    if save:
        graph = save_graph(filename)
    else:
        graph = load_graph(filename)

    num_nodes = len(graph.edges.keys())
    
    
    # num_clusters = math.pow(2, math.ceil(math.log2(num_nodes//2)))
    # print("there are {} clusters".format(num_clusters))

    clusters = graph.fiedler_clustering(2)

    node_list, page_rank, ranking_list = graph.page_rank()

    for i in range(100):
        node = heapq.heappop(node_list)
        print("node: {} ranking: {} cluster: {}".format(
            node, node.ranking, node.cluster))

    