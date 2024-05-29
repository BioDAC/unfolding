"""Test tesselation module"""

import numpy as np
from scipy.spatial import distance_matrix
from unfolding import unfold, mesh_from_label
from unfolding._utils import sphere, conic, plane


def test_mesh_from_label():
    _, label = sphere()
    verts, faces = mesh_from_label(label, num_vertices=30)
    verts_true = np.load("test/verts.npy")
    faces_true = np.load("test/faces.npy")
    assert np.linalg.norm(verts - verts_true) < 1e-3
    assert np.linalg.norm(faces - faces_true) < 1e-3


def test_create_simplified_tesselation_conic():
    """Compare the tesseslation to a delaunay tesselation of the conic function"""
    _, label, verts_true, faces_true = conic()
    verts, _ = mesh_from_label(label, num_vertices=faces_true.shape[0])
    d = np.amin(distance_matrix(verts, verts_true), 0)
    assert d.max() < 20
    assert d.mean() < 5


def test_create_simplified_tesselation_plane():
    """Compare the tesseslation to a delaunay tesselation of the conic function"""
    _, label, verts_true, faces_true = plane()
    verts, _ = mesh_from_label(label, num_vertices=faces_true.shape[0])
    d = np.amin(distance_matrix(verts, verts_true), 0)
    assert d.max() < 10
    assert d.mean() < 5


def test_unfold_tessellation():
    verts_true = np.load("test/verts.npy")
    faces_true = np.load("test/faces.npy")
    verts_2d_true = np.load("test/verts_2d.npy")
    faces_2d_true = np.load("test/faces_2d.npy")
    verts_2d, faces_2d, dict_2d_3d = unfold(verts_true, faces_true)
    np.save("dict_2d_3d.npy", dict_2d_3d)
    assert np.linalg.norm(verts_2d - verts_2d_true) < 1e-3
    assert np.linalg.norm(faces_2d - faces_2d_true) < 1e-3
