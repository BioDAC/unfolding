# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:34:26 2023

@author: Anita Karsa, University of Cambridge, UK
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
from tqdm import tqdm

# import pandas as pd


import pymeshlab as ml


# In[]: Creating a simplified mesh


def create_simplified_tessellation(label, num_vertices=30):
    # label: 3D numpy array label map (binary values)
    # num_vertices: number of vertices (default = 30)

    verts, faces, _, _ = skimage.measure.marching_cubes(
        label, 0.0, step_size=2, allow_degenerate=False
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


# In[]: Unfold tessellation into 2D


def unfold_tessellation(verts, faces, base_triangle, draw):
    # verts,faces: 3D vertex coordinates and triangle faces from create_simplified_tessellation
    # base_triangle: index of the first triangle to draw (this will be the middle of the unfolded image)
    # draw: if ==1, this function will also plot the triangles of the unfolded tessellation

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

                    if draw == 1:
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


# In[]: Get layers parallel with the tessellation from the 3D image


def unfolded_layers(verts, faces, verts_2d, faces_2d, dict_2d_3d, im, n_layers):
    # verts, faces: 3D vertice coordinates and triangle faces from create_simplified_tessellation
    # verts_2d, faces_2d, dict_2d_3d: 2D vertice coordinates and triangle faces,
    # and the correspondence dictionary between 3D and 2D vertices from unfold_tessellation
    # im: 3D numpy array containing the grayscale image
    # n_layers: number of layers to be exported on both sides of the tessellation

    # Create an image to save the unfolded layers
    x = [vert[0] for vert in verts_2d]
    pix_x0 = np.min(x)
    pix_xw = np.max(x) - pix_x0
    y = [vert[1] for vert in verts_2d]
    pix_y0 = np.min(y)
    pix_yw = np.max(y) - pix_y0

    layers = np.zeros([int(pix_xw) + 5, int(pix_yw) + 5, 2 * n_layers + 1])
    del x, y

    # Loop through all 2d triangles
    for face_2d in tqdm(faces_2d[:]):
        # Get coordinates of triangle in both the 2d and the 3d image
        coord_2d = [verts_2d[vert].copy() for vert in face_2d]
        coord_3d = [
            verts[vert].copy() for vert in [dict_2d_3d[vert_id] for vert_id in face_2d]
        ]

        if triangle_area(coord_3d) > 1e-10:
            # Get mip perpendicular to triangle
            layers_crop, coord_new = get_perp_layers(coord_3d, coord_2d, im, n_layers)

            # Check that both triangles have the same size using one side
            assert (
                np.abs(
                    np.linalg.norm(coord_2d[2] - coord_2d[1])
                    / np.linalg.norm(coord_new[2] - coord_new[1])
                    - 1
                )
                < 0.001
            ), "Something might be wrong. Triangles are not the same size"

            # Embed mip in final_mip
            embedding_window = (np.min(coord_2d, axis=0) - [pix_x0, pix_y0]).astype(int)
            layers_window = layers[
                embedding_window[0] : (embedding_window[0] + layers_crop.shape[0]),
                embedding_window[1] : (embedding_window[1] + layers_crop.shape[1]),
                :,
            ]
            layers[
                embedding_window[0] : (embedding_window[0] + layers_crop.shape[0]),
                embedding_window[1] : (embedding_window[1] + layers_crop.shape[1]),
                :,
            ] = np.max([layers_crop, layers_window], axis=0)

    return layers


def get_perp_layers(coord_3d, coord_2d, im, n_layers):
    # Create mask of triangle to follow evolution of the image
    mask = np.zeros(im.shape)
    targets = [
        coord_3d[1] + (coord_3d[2] - coord_3d[1]) * epsilon
        for epsilon in np.linspace(0, 1, 200)
    ]
    for target in targets:
        for epsilon in np.linspace(0, 1, 200):
            point = coord_3d[0] + (target - coord_3d[0]) * epsilon
            mask[int(point[0]), int(point[1]), int(point[2])] = 1

    # print('Triangle done!')
    # im2 = im + mask*200

    # Find normal vector of triangle
    n_triangle = np.cross(coord_3d[1] - coord_3d[0], coord_3d[2] - coord_3d[0])
    n_triangle = n_triangle / np.linalg.norm(n_triangle)

    # Crop im to only the area that could end up in the output
    coord_3d_plus = [
        coord_3d_corner + n_layers * n_triangle for coord_3d_corner in coord_3d
    ]
    coord_3d_minus = [
        coord_3d_corner - n_layers * n_triangle for coord_3d_corner in coord_3d
    ]

    crop_min = np.min(
        [np.min(coord_3d_plus, axis=0), np.min(coord_3d_minus, axis=0)], axis=0
    )
    crop_max = np.max(
        [np.max(coord_3d_plus, axis=0), np.max(coord_3d_minus, axis=0)], axis=0
    )
    crop_min = np.max([crop_min - 5, [0, 0, 0]], axis=0).astype(int)
    crop_max = np.min([crop_max + 5, im.shape], axis=0).astype(int)

    im_cropped = im[
        crop_min[0] : crop_max[0], crop_min[1] : crop_max[1], crop_min[2] : crop_max[2]
    ].copy()
    mask_cropped = mask[
        crop_min[0] : crop_max[0], crop_min[1] : crop_max[1], crop_min[2] : crop_max[2]
    ].copy()

    # print(crop_min)

    coord_3d = [coord - crop_min for coord in coord_3d]

    # im_cropped = im.copy()
    # mask_cropped = mask.copy()

    # print(im_cropped.shape)

    # print(coord_3d)

    # Rotate about the x axis to align with z
    if (n_triangle[2] == 0) and (n_triangle[1] == 0):
        angle = 0
    else:
        angle = np.arccos(
            n_triangle[2] / np.sqrt(n_triangle[2] ** 2 + n_triangle[1] ** 2)
        ) * np.sign(n_triangle[1])

    im_rot, mask_rot = rotate_im_and_mask(im_cropped, mask_cropped, angle, (1, 2))

    # Update vertice positions
    axes = [1, 2]
    middle_voxel_before = np.asarray(mask_cropped.shape) / 2
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            rotate_point,
            coord_3d,
            [axes] * len(coord_3d),
            [middle_voxel_before] * len(coord_3d),
            [middle_voxel_after] * len(coord_3d),
            [angle] * len(coord_3d),
        )
    )
    # print(coord_3d)

    # print('x rotation done!')

    # Rotate about the y axis to align with z
    angle = np.arccos(
        n_triangle[2] / np.sqrt(n_triangle[2] ** 2 + n_triangle[0] ** 2)
    ) * np.sign(n_triangle[0])

    im_rot, mask_rot = rotate_im_and_mask(im_rot, mask_rot, angle, (0, 2))

    # Update vertice positions
    axes = [0, 2]
    middle_voxel_before = middle_voxel_after.copy()
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            rotate_point,
            coord_3d,
            [axes] * len(coord_3d),
            [middle_voxel_before] * len(coord_3d),
            [middle_voxel_after] * len(coord_3d),
            [angle] * len(coord_3d),
        )
    )
    # print(coord_3d)

    # print('y rotation done!')

    # Check that all three are at the same z coordinate
    assert (
        np.abs(coord_3d[0][2] - coord_3d[1][2]) < 1
        and np.abs(coord_3d[0][2] - coord_3d[2][2]) < 1
    ), "Triangle is not in a single z plane!"

    # Rotate about the z axis to align with 2d orientation
    current_01 = coord_3d[1][0:2] - coord_3d[0][0:2]
    target_01 = coord_2d[1] - coord_2d[0]

    current_angle = np.arctan2(current_01[1], current_01[0])
    target_angle = np.arctan2(target_01[1], target_01[0])

    angle = target_angle - current_angle

    im_rot, mask_rot = rotate_im_and_mask(im_rot, mask_rot, angle, (0, 1))

    # Update vertice positions
    axes = [0, 1]
    middle_voxel_before = middle_voxel_after.copy()
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            rotate_point,
            coord_3d,
            [axes] * len(coord_3d),
            [middle_voxel_before] * len(coord_3d),
            [middle_voxel_after] * len(coord_3d),
            [angle] * len(coord_3d),
        )
    )
    # print(coord_3d)

    # print('z rotation done!')

    # Shift image to an integer middle
    triangle_slice = int(np.round(coord_3d[0][2]))
    # shift = coord_3d[0][2] - triangle_slice
    # print(shift)

    # mask_rot = scipy.ndimage.shift(
    #     mask_rot,
    #     [0,0,shift],
    #     order=3,
    #     mode="constant",
    #     cval=0.0,
    #     prefilter=False,
    # )
    # im_rot = scipy.ndimage.shift(
    #     im_rot,
    #     [0,0,shift],
    #     order=3,
    #     mode="constant",
    #     cval=0.0,
    #     prefilter=False,
    # )

    # Extract layers around triangle
    layers = np.zeros([im_rot.shape[0], im_rot.shape[1], 2 * n_layers + 1])
    first_slice = np.max([triangle_slice - n_layers, 0])
    last_slice = np.min([triangle_slice + n_layers + 1, im_rot.shape[2]])
    layers[
        :,
        :,
        n_layers
        - (triangle_slice - first_slice) : n_layers
        + (last_slice - triangle_slice),
    ] = im_rot[:, :, first_slice:last_slice]

    # Apply triangular mask to each layer
    mask_rot = np.max(mask_rot, axis=2) > 0.1
    layers *= np.tile(np.expand_dims(mask_rot, 2), [1, 1, 2 * n_layers + 1])

    # Crop layers
    x = np.where(np.sum(mask_rot, axis=0) > 0)[0]
    y = np.where(np.sum(mask_rot, axis=1) > 0)[0]
    layers = layers[y[0] : (y[-1] + 1), x[0] : (x[-1] + 1)]
    coord_new = [c[0:2] - [y[0], x[0]] for c in coord_3d]

    # print('layers extracted!')

    return layers, coord_new


def triangle_area(vertices):
    return (
        np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
        / 2
    )


def rotate_point(vector, axes, middle_point_before, middle_point_after, angle):
    rot_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    vector[[axes]] = (
        np.squeeze(
            np.matmul(
                rot_matrix,
                np.reshape(vector[[axes]] - middle_point_before[[axes]], [2, 1]),
            )
        )
        + middle_point_after[[axes]]
    )
    return vector


def rotate_im_and_mask(im, mask, angle, axes):
    mask_rot = scipy.ndimage.rotate(
        mask,
        angle / (2 * np.pi) * 360,
        axes=axes,
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    im_rot = scipy.ndimage.rotate(
        im,
        angle / (2 * np.pi) * 360,
        axes=axes,
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return im_rot, mask_rot


# In[]: Draw triangles on 3d image


def draw_triangles_on_image(im, verts, faces):
    im_out = im.copy()
    for face in faces:
        coord_3d = [np.reshape(verts[vert].copy(), [3, 1]) for vert in face]
        targets = coord_3d[1] + (coord_3d[2] - coord_3d[1]) * np.reshape(
            np.linspace(0, 1, 200), [1, -1]
        )
        for i in range(targets.shape[1]):
            target = targets[:, i : i + 1]
            points = (
                coord_3d[0]
                + (target - coord_3d[0]) * np.reshape(np.linspace(0, 1, 200), [1, -1])
            ).astype(int)
            # remove points that are not in the image
            points = np.squeeze(points[:, np.where(np.sum((points < 0), axis=0) == 0)])
            points = np.squeeze(
                points[
                    :,
                    np.where(
                        np.sum(
                            (
                                (np.reshape(np.asarray(im_out.shape), [3, 1]) - points)
                                <= 0
                            ),
                            axis=0,
                        )
                        == 0
                    ),
                ]
            )
            im_out[points[0, :], points[1, :], points[2, :]] = 1
    return im_out
