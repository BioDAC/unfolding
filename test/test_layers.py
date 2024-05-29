import numpy as np
from unfolding import unfold_layers, extract_layers
from unfolding._layers import _map_point_in_triangle
from unfolding._utils import sphere, flat
from numpy.testing import assert_array_almost_equal


def test_unfolded_layers():
    image, _ = sphere()
    verts = np.load("test/verts.npy")
    faces = np.load("test/faces.npy")
    verts_2d = np.load("test/verts_2d.npy")
    faces_2d = np.load("test/faces_2d.npy")
    dict_2d_3d = np.load("test/dict_2d_3d.npy")
    layers = unfold_layers(
        verts, faces, verts_2d, faces_2d, dict_2d_3d, image, n_layers=20
    )
    layers_true = np.load("test/layers.npy")
    assert np.linalg.norm(layers - layers_true) / layers_true.size < 0.01


def test_map_points_in_triangle():
    v1 = np.random.uniform(size=(3, 3))
    v2 = np.random.uniform(size=(3, 3))
    e = np.array([0, 1, 2])
    triangle1 = v1[e]
    triangle2 = v2[e]
    x = np.array([0.25, 0.25, 0]).reshape(3, 1)
    x = np.random.uniform(size=(1, 3))
    y = _map_point_in_triangle(x, triangle1, triangle2)
    z = _map_point_in_triangle(y, triangle2, triangle1)
    assert np.linalg.norm(x - z) < 1e-3


def test_flat_layer():
    """A flat plane layer should be identical after projection"""
    image, _, verts, faces = flat()
    layers = extract_layers(
        verts, verts[:, :2], faces, image, layers=np.linspace(-1, 1, 3)
    )
    layers_true = image[0 : layers.shape[0] + 1, 0 : layers.shape[1], 60]
    assert_array_almost_equal(layers[10:50, 10:50, 1], layers_true[10:50, 10:50], 3)


def test_flat_layer_unfold():
    """A flat plane layer should be identical after projection"""
    image, _, verts, faces = flat()
    layers = unfold_layers(
        verts, faces, verts[:, :2], faces, np.arange(verts.shape[0]), image, n_layers=1
    )
    layers_true = image[0 : layers.shape[0] + 1, 0 : layers.shape[1], 60]
    assert_array_almost_equal(layers[10:50, 10:50, 1], layers_true[10:50, 10:50], 0)


def test_extract_layer():
    image, _ = sphere()
