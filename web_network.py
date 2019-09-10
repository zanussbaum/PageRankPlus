import pickle
import argparse
from graph import Graph, Node


def save_graph(filename):
    graph = Graph()

    count = 0
    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith("#"):
                split = line.replace(" ", "").split()
                from_node = split[0]
                to_node = split[1]

                graph.add_edge(from_node, to_node)

                count += 1

                if count % 1000 == 0:
                    print("added {} to {}. Have added {} edges".format(
                        from_node, to_node, count))

    pickle_name = filename.replace(".txt", "")

    pickle_file = open(pickle_name, 'ab')

    pickle.dump(graph, pickle_file)

    pickle_file.close()

    return graph


def load_graph(filename):
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

    