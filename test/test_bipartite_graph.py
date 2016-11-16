__author__ = 'benchamberlain'

from ..bipartite_graph import BipartiteGraph
from scipy.sparse import csr_matrix
import numpy as np

data = csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0]]))
data1 = csr_matrix(np.array([[1, 0], [0, 1]]))

row_edges = np.array([[1, 2, 0], [0, 2, 3], [0, 1, 2]])
col_edges = np.array([[1, 2, 0], [0, 2, 0], [0, 1, 2], [1, 0, 0]])

row_degs = np.array([2, 3, 3])
col_degs = np.array([2, 2, 3, 1])

walks = np.array([[0, 2, 3], [1, 3, 1]])


def test_number_of_vertices():
    g = BipartiteGraph(data)
    assert g.n_rows == 3
    assert g.n_cols == 4


def test_input_degree():
    g = BipartiteGraph(data)
    assert np.array_equal(row_degs, g.row_deg)
    assert np.array_equal(col_degs, g.col_deg)


def test_input_edge_shape():
    g = BipartiteGraph(data)
    row_edges = (3, 3)
    col_edges = (4, 3)
    assert row_edges == g.row_edges.shape
    assert col_edges == g.col_edges.shape


def test_input_edges():
    g = BipartiteGraph(data)
    g.build_edge_array()
    assert np.array_equal(row_edges, g.row_edges)
    assert np.array_equal(col_edges, g.col_edges)


def test_initialise_walk_array():
    g = BipartiteGraph(data)
    num_walks = 10
    walk_length = 20
    walks = g.initialise_walk_array(num_walks=num_walks, walk_length=walk_length)
    assert walks.shape == (30, 20)
    assert np.array_equal(walks[:, 0], np.array([0, 1, 2] * 10))


def test_sample_next_vertices():
    """
    In the test graph the vertex with index 2 is only connected to vertices 1 and 3
    :return:
    """
    g = BipartiteGraph(data)
    current_vertices = np.array([0, 0, 0, 0])
    for idx in range(10):
        next_vertex_indices = g.sample_next_vertices(current_vertices, row_degs)
        for elem in next_vertex_indices:
            assert (elem == 0) | (elem == 1)
        assert next_vertex_indices.shape == current_vertices.shape


def test_oscillating_random_walk_1walk():
    g = BipartiteGraph(data1)
    g.build_edge_array()
    walks = g.generate_walks(1, 10)
    walk1 = [0, 2] * 5
    walk2 = [1, 3] * 5
    truth = np.array([walk1, walk2])
    print walks
    assert np.array_equal(walks, truth)


# def test_oscillating_random_walk_2walks():
#     g = BipartiteGraph(data1)
#     g.build_edge_array()
#     walks = g.generate_walks(2, 10)
#     walk1 = [0, 1] * 5
#     walk2 = [1, 0] * 5
#     truth = np.array([walk1, walk2, walk1, walk2])
#     print walks
#     assert np.array_equal(walks, truth)
