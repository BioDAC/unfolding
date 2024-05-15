import numpy as np
from unfolding import unfolded_layers
from unfolding._utils import dummy


def test_unfolded_layers():
    image, _ = dummy()
    verts = np.load("test/verts.npy")
    faces = np.load("test/faces.npy")
    verts_2d = np.load("test/verts_2d.npy")
    faces_2d = np.load("test/faces_2d.npy")
    dict_2d_3d = np.load("test/dict_2d_3d.npy")
    layers = unfolded_layers(verts, faces, verts_2d, faces_2d, dict_2d_3d, image, 20)
    layers_true = np.load("test/layers.npy")
    assert np.linalg.norm(layers - layers_true) / layers_true.size < 0.01
