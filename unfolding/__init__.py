"""Unfolding of surfaces extracted from 3D volume"""

__all__ = [
    "mesh_from_label",
    "find_center_triangle",
    "unfold",
    "unfold_layers",
    "show_3d_and_contours",
    "draw_triangles_in_3d_and_2d",
    "extract_layers",
    "smacof_mesh",
]

from unfolding._mesh import (
    mesh_from_label,
    unfold,
    find_center_triangle,
    smacof_mesh,
)
from unfolding._layers import unfold_layers, extract_layers
from unfolding._display import show_3d_and_contours, draw_triangles_in_3d_and_2d
