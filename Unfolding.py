"""
Created on Fri Jan  5 10:25:57 2024

@author: Anita Karsa, University of Cambridge, UK
"""

import numpy as np

# import sys

# sys.path.append("../")
from unfolding._mesh import (  # noqa: E402
    create_simplified_tessellation,
    unfold,
    unfolded_layers,
    draw_triangles_in_3d_and_2d,
    show_3d_and_contours,
)


def create_dummy():
    x, y, z = np.meshgrid(
        np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    )

    image = x + y + z

    label = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) < 0.2**2

    return image, label


# create dummy data
image, label = create_dummy()

# create simplified tessellation
verts, faces = create_simplified_tessellation(
    label, num_vertices=30, initial_step_size=2
)
# num_vertices: target number of vertices in the simplified tessellation

# unfold tessellation
verts_2d, faces_2d, dict_2d_3d = unfold(verts, faces, base_triangle=0, draw=0)
# base_triangle: the index of the row in faces that contains the first triangle to consider
# (this will be the middle of the unfolded surface)
# draw: 0 or 1 indicating whether the function should plot the unfolded tessellation or not

# unfold and extract layers
layers = unfolded_layers(
    verts, faces, verts_2d, faces_2d, dict_2d_3d, image, n_layers=20
)
# n_layers: number of layers to be exported on both sides of the surface
# (i.e. layers will have 2*n_layers+1 slices)

# IN DEVELOPMENT: create coloured plots for establishing correspondence between 2d and 3d
triangles_3d, triangles_2d = draw_triangles_in_3d_and_2d(
    verts, faces, verts_2d, faces_2d, dict_2d_3d, image.shape
)
# show_3d_and_contours(image,triangles_3d,[0,5])
show_3d_and_contours(np.max(layers, axis=2, keepdims=True), triangles_2d, [0, 5])
