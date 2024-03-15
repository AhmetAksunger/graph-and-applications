import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class AdjMatrixUndiGraph:
    """
    Represents an undirected graph using an adjacency matrix.
    """

    def __init__(self, v: int = None, filename: str = None):
        """
        Initializes an AdjMatrixUndiGraph object.

        Parameters:
        - v: int, optional, number of vertices.
        - filename: str, optional, name of the file to read the graph from.
        """

        if filename is None and v is None:
            raise ValueError("Both 'filename' and 'v' cannot be None!")

        if filename is None:
            self.V = v
            self.matrix = np.zeros((self.V, self.V), dtype=bool)
        else:
            with open(filename) as file:
                lines = file.readlines()
                self.V = int(lines[0])
                self.matrix = np.zeros((self.V, self.V), dtype=bool)
                for line in lines[2:]:
                    pairs = line.strip().split(' ')
                    self.add_edge(int(pairs[0]), int(pairs[1]))

    def add_edge(self, v: int, w: int) -> None:
        """
        Adds an edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        self.matrix[v][w] = True
        self.matrix[w][v] = True

    def remove_edge(self, v: int, w: int) -> None:
        """
        Removes the edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        self.matrix[v][w] = False
        self.matrix[w][v] = False

    def clear_edges_of_vertex(self, v: int) -> None:
        """
        Clears all edges connected to vertex v.

        Parameters:
        - v: int, vertex index.
        """
        self.matrix[v, :] = False
        self.matrix[:, v] = False

    def adj(self, v: int) -> np.ndarray:
        """
        Returns an array of adjacent vertices to vertex v.

        Parameters:
        - v: int, vertex index.

        Returns:
        - np.ndarray: Array of adjacent vertices.
        """
        return np.argwhere(self.matrix[v]).flatten()


class AdjMatrixDiGraph:
    """
    Represents a directed graph using an adjacency matrix.
    """

    def __init__(self, v: int):
        """
        Initializes an AdjMatrixDiGraph object.

        Parameters:
        - v: int, number of vertices.
        """
        self.V = v
        self.matrix = np.zeros((self.V, self.V), dtype=bool)

    def add_edge(self, v: int, w: int) -> None:
        """
        Adds a directed edge from vertex v to vertex w.

        Parameters:
        - v: int, source vertex index.
        - w: int, destination vertex index.
        """
        self.matrix[v][w] = True

    def remove_edge(self, v: int, w: int) -> None:
        """
        Removes the directed edge from vertex v to vertex w.

        Parameters:
        - v: int, source vertex index.
        - w: int, destination vertex index.
        """
        self.matrix[v][w] = False

    def remove_both_edges(self, v: int, w: int) -> None:
        """
        Removes both directed edges between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        self.remove_edge(v, w)
        self.remove_edge(w, v)

    def clear_edges_of_vertex(self, v: int) -> None:
        """
        Clears all edges originating from vertex v.

        Parameters:
        - v: int, vertex index.
        """
        self.matrix[v, :] = False

    def adj(self, v: int) -> np.ndarray:
        """
        Returns an array of vertices adjacent to vertex v.

        Parameters:
        - v: int, vertex index.

        Returns:
        - np.ndarray: Array of adjacent vertices.
        """
        return np.argwhere(self.matrix[v]).flatten()


class GraphVisualizer:
    """
    Visualizes graphs using NetworkX library.
    """

    def __init__(self, g: AdjMatrixUndiGraph | AdjMatrixDiGraph):
        """
        Initializes a GraphVisualizer object.

        Parameters:
        - graph: AdjMatrixUndiGraph or AdjMatrixDiGraph, the graph to visualize.
        """
        self.g = g

    def visualize(self, node_size=500, with_labels=True) -> None:
        """
        Visualizes the graph.

        Parameters:
        - node_size: int, optional, size of the nodes in the visualization.
        - with_labels: bool, optional, whether to display node labels.
        """
        if isinstance(self.g, AdjMatrixUndiGraph):
            self._visualize_undirected(node_size, with_labels)
        elif isinstance(self.g, AdjMatrixDiGraph):
            self._visualize_directed(node_size, with_labels)

    def _visualize_undirected(self, node_size=500, with_labels=True) -> None:
        """
        Visualizes an undirected graph.

        Parameters:
        - node_size: int, optional, size of the nodes in the visualization.
        - with_labels: bool, optional, whether to display node labels.
        """
        gr = nx.Graph()
        gr.add_edges_from(np.argwhere(self.g.matrix))
        nx.draw(gr, node_size=node_size, with_labels=with_labels)
        plt.show()

    def _visualize_directed(self, node_size=500, with_labels=True) -> None:
        """
        Visualizes a directed graph.

        Parameters:
        - node_size: int, optional, size of the nodes in the visualization.
        - with_labels: bool, optional, whether to display node labels.
        """
        gr = nx.DiGraph()
        gr.add_edges_from(np.argwhere(self.g.matrix))
        nx.draw(gr, node_size=node_size, with_labels=with_labels)
        plt.show()

