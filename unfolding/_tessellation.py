"""Manipulate and unfold tesselations"""

import numpy as np

import skimage

from matplotlib import pyplot as plt

import pymeshlab as ml


def create_simplified_tessellation(
    label: np.ndarray, num_vertices: int = 30, initial_step_size: float = 2
):
    """
    Extract a surface mesh from the label volume and simplifies

    Parameters
    ----------
    label: ndarray
        3D numpy array label map (binary values)
    num_vertices: int
        number of vertices (default = 30)
    initial_step_size: float
        step size for the marching cubes algorithm

    Returns
    -------
    verts: ndarray
        Vertices of the mesh
    faces: ndarray
        Faces of the mesh

    """

    verts, faces, _, _ = skimage.measure.marching_cubes(
        label, 0.0, step_size=initial_step_size, allow_degenerate=False
    )

    # simplify mesh
    m = ml.Mesh(verts, faces)

    # Generate meshSet and add mesh
    ms = ml.MeshSet()
    ms.add_mesh(m)
    numFaces = 100 + 2 * num_vertices
    while ms.current_mesh().vertex_number() > num_vertices:
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=numFaces,
            preservenormal=True,
        )
        numFaces = numFaces - (ms.current_mesh().vertex_number() - num_vertices)
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return verts, faces


def unfold_tessellation(
    verts: np.ndarray, faces: np.ndarray, base_triangle: int = 0, draw: bool = False
):
    """
    Unfold the tesselation

    Parameters
    ----------
    verts: ndarray
        Vertices of the mesh
    faces: ndarray
        Faces of the mesh
    base_triangle: int
        Index of the first triangle to draw (this will be the middle of the unfolded image)
    draw: bool
        activate drawing of triangles

    Returns
    -------
    verts_2d: ndarray
        Vertices of the 2D mesh
    faces_2d: ndarray
        Faces of the 2D mash
    dict_2d_3d: dict
        Mapping indices of 2D to 3D vertices

    """

    # Draw base triangle
    faces_copy = faces.copy()
    triang_3d = [verts[vert] for vert in faces_copy[base_triangle]]
    triang_2d = [
        np.array([0, 0]),
        np.array([np.linalg.norm(triang_3d[1] - triang_3d[0]), 0]),
    ]
    triang_2d.append(find_2d_coordinates(triang_3d, triang_2d, 1))

    if draw == 1:
        draw_2d_triangle(triang_2d)
        for j in range(3):
            plt.text(triang_2d[j][0], triang_2d[j][1], str(j), fontsize=10)

    # Initialise and add elements to verts_2d, faces_2d, and dict_2d_3d
    verts_2d = [triang_2d[0], triang_2d[1], triang_2d[2]]
    faces_2d = [np.array([0, 1, 2])]
    dict_2d_3d = [
        faces_copy[base_triangle][0],
        faces_copy[base_triangle][1],
        faces_copy[base_triangle][2],
    ]

    # List outer edges and remove first triangle from faces_copy
    outer_edges = [faces_2d[0][[1, 0]], faces_2d[0][[2, 1]], faces_2d[0][[0, 2]]]
    faces_copy = np.delete(faces_copy, base_triangle, 0)

    # Loop through outder_edges
    # for i in range(4):
    while len(faces_copy) > 0:
        outer_edges_next = []
        for edge in outer_edges:
            # Convert edge to 3d coordinates
            edge_3d = np.array([dict_2d_3d[edge[0]], dict_2d_3d[edge[1]]])

            # Find 3d face with both vertices in it
            face_index = np.where(
                (np.prod(faces_copy - edge_3d[0], axis=1) == 0)
                * (np.prod(faces_copy - edge_3d[1], axis=1) == 0)
            )

            if len(face_index[0]) != 0:
                current_face = faces_copy[int(face_index[0][0])]
                new_vertice_3d = current_face[~np.isin(current_face, edge_3d)]
                current_face = np.concatenate([edge_3d, new_vertice_3d])

                # Calculate coordinates of new vertice and draw triangle
                triang_3d = [verts[vert] for vert in current_face]
                triang_2d = [verts_2d[vert] for vert in edge]
                new_coordinate = find_2d_coordinates(triang_3d, triang_2d, 1)

                if not np.isnan(new_coordinate[0]):
                    triang_2d.append(new_coordinate)

                    if draw is True:
                        draw_2d_triangle(triang_2d)
                        plt.text(
                            triang_2d[2][0],
                            triang_2d[2][1],
                            str(len(verts_2d)),
                            fontsize=10,
                        )

                    # Update verts_2d, faces_2d, and dict_2d_3d
                    current_face_2d = np.concatenate([edge, np.array([len(verts_2d)])])
                    verts_2d.append(triang_2d[2])
                    faces_2d.append(np.array(current_face_2d))
                    dict_2d_3d.append(int(new_vertice_3d[0]))

                    # Update new outer_edges and remove triangle from faces_copy
                    outer_edges_next.append(current_face_2d[[0, 2]])
                    outer_edges_next.append(current_face_2d[[2, 1]])
                    faces_copy = np.delete(faces_copy, int(face_index[0][0]), 0)
                else:
                    faces_copy = np.delete(faces_copy, int(face_index[0][0]), 0)

        outer_edges = outer_edges_next

    return verts_2d, faces_2d, dict_2d_3d


def draw_2d_triangle(vertices):
    x = [vert[0] for vert in vertices]
    x.append(x[0])
    y = [vert[1] for vert in vertices]
    y.append(y[0])
    plt.plot(x, y)


def find_2d_coordinates(vertices_3D, vertices_2D, orientation):
    a = np.linalg.norm(vertices_3D[1] - vertices_3D[0])
    b = np.linalg.norm(vertices_3D[2] - vertices_3D[0])
    c = np.linalg.norm(vertices_3D[2] - vertices_3D[1])
    gamma = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    unit_vector01 = (vertices_2D[1] - vertices_2D[0]) / np.linalg.norm(
        vertices_2D[1] - vertices_2D[0]
    )
    unit_vector01_perp = np.flip(unit_vector01) * np.array([-1, 1]) * orientation
    return (
        vertices_2D[0]
        + unit_vector01 * np.cos(gamma) * b
        + unit_vector01_perp * np.sin(gamma) * b
    )
