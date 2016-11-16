__author__ = 'benchamberlain'

from ..graph import Graph
from scipy.sparse import csr_matrix
import numpy as np

data = csr_matrix(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]))
data1 = csr_matrix(np.array([[0, 1], [1, 0]]))

edges = np.array([[1, 3, 0], [0, 2, 3], [1, 3, 0], [0, 1, 2]])

degs = np.array([2, 3, 2, 3])

walks = np.array([[0, 2, 3], [1, 3, 1]])


def test_number_of_vertices():
    g = Graph(data)
    assert g.n_vertices == 4


def test_input_degree():
    g = Graph(data)
    assert np.array_equal(degs, g.deg)


def test_input_edge_shape():
    g = Graph(data)
    truth = (4, 3)
    assert truth == g.edges.shape


def test_input_edges():
    g = Graph(data)
    g.build_edge_array()
    assert np.array_equal(edges, g.edges)


def test_initialise_walk_array():
    g = Graph(data)
    num_walks = 10
    walk_length = 20
    walks = g.initialise_walk_array(num_walks=num_walks, walk_length=walk_length)
    assert walks.shape == (40, 20)
    assert np.array_equal(walks[:, 0], np.array([0, 1, 2, 3] * 10))


def test_sample_next_vertices():
    """
    In the test graph the vertex with index 2 is only connected to vertices 1 and 3
    :return:
    """
    g = Graph(data)
    current_vertices = np.array([2, 2, 2, 2])
    for idx in range(10):
        next_vertex_indices = g.sample_next_vertices(current_vertices, degs)
        for elem in next_vertex_indices:
            assert (elem == 0) | (elem == 1)
        assert next_vertex_indices.shape == current_vertices.shape


def test_walks_to_list_of_strings():
    walks_str = walks.astype(str)
    walk_list = walks_str.tolist()
    for walk in walk_list:
        assert len(walk) == 3
        for elem in walk:
            assert type(elem) == str


def test_oscillating_random_walk_1walk():
    g = Graph(data1)
    g.build_edge_array()
    walks = g.generate_walks(1, 10)
    walk1 = [0, 1] * 5
    walk2 = [1, 0] * 5
    truth = np.array([walk1, walk2])
    print walks
    assert np.array_equal(walks, truth)


def test_oscillating_random_walk_2walks():
    g = Graph(data1)
    g.build_edge_array()
    walks = g.generate_walks(2, 10)
    walk1 = [0, 1] * 5
    walk2 = [1, 0] * 5
    truth = np.array([walk1, walk2, walk1, walk2])
    print walks
    assert np.array_equal(walks, truth)