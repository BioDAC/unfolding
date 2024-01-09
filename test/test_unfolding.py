"""
Created on Tue Jan  2 12:56:15 2024

@author: Anita Karsa, University of Cambridge, UK
"""

import numpy as np

# import sys

# sys.path.append("../")
from src.utils_unfolding import (  # noqa: E402
    create_simplified_tessellation,
    unfold_tessellation,
    unfolded_layers,
)


def test_unfolding():
    # load ground truth values
    with open("test/verts.npy", "rb") as f:
        verts_true = np.load(f)
    with open(
        "test/faces.npy",
        "rb",
    ) as f:
        faces_true = np.load(f)
    with open(
        "test/verts_2d.npy",
        "rb",
    ) as f:
        verts_2d_true = np.load(f)
    with open(
        "test/faces_2d.npy",
        "rb",
    ) as f:
        faces_2d_true = np.load(f)
    with open(
        "test/layers.npy",
        "rb",
    ) as f:
        layers_true = np.load(f)

    # create dummy
    image, label = create_dummy()

    # create simplified tessellation
    verts, faces = create_simplified_tessellation(label, num_vertices=30)
    # check results
    if np.linalg.norm(verts - verts_true) > 1e-3:
        print("Test failed for create_simplified_tessellation: verts not correct")
        return 1
    if np.linalg.norm(faces - faces_true) > 1e-3:
        print("Test failed for create_simplified_tessellation: faces not correct")
        return 2

    # unfold tessellation
    verts_2d, faces_2d, dict_2d_3d = unfold_tessellation(verts, faces, 0, 0)
    # check results
    if np.linalg.norm(verts_2d - verts_2d_true) > 1e-3:
        print("Test failed for unfold_tessellation: verts_2d not correct")
        return 3
    if np.linalg.norm(faces_2d - faces_2d_true) > 1e-3:
        print("Test failed for unfold_tessellation: faces_2d not correct")
        return 4

    # unfold and extract layers
    layers = unfolded_layers(verts, faces, verts_2d, faces_2d, dict_2d_3d, image, 20)
    if np.linalg.norm(layers - layers_true) / layers_true.size > 0.01:
        print("Test failed for unfolded_layers: layers not correct")
        return 5

    print("Everything ok.")

    return None


def create_dummy():
    x, y, z = np.meshgrid(
        np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    )

    image = x + y + z

    label = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) < 0.2**2

    return image, label


# test_unfolding()
