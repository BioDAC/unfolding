"""Layer extraction from unfolded surface"""

from tqdm import tqdm

import numpy as np
from scipy import ndimage as ndi
from scipy import interpolate
from math import floor, ceil

from ._utils import triangle_area


def unfolded_layers(verts, faces, verts_2d, faces_2d, dict_2d_3d, im, spacing=[1,1,1], n_layers=1):
    """
    Extract layers from the volume

    Parameters
    ----------
    verts, faces: 3D vertice coordinates and triangle faces from create_simplified_tessellation
    verts_2d, faces_2d, dict_2d_3d: 2D vertice coordinates and triangle faces,
    and the correspondence dictionary between 3D and 2D vertices from unfold_tessellation
    im: 3D numpy array containing the grayscale image
    spacing: ndarray 
    pixel spacing
    n_layers: number of layers to be exported on both sides of the tessellation

    Returns
    -------
    layers: ndarray


    """
    verts = [v/spacing for v in verts]
    verts_2d = [v/spacing[:2] for v in verts_2d]
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
            layers_crop, coord_new = _get_perp_layers(coord_3d, coord_2d, im, n_layers)

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


def _get_perp_layers(coord_3d, coord_2d, im, n_layers):
    # Create mask of triangle to follow evolution of the image
    mask = np.zeros(im.shape)
    targets = [
        coord_3d[1] + (coord_3d[2] - coord_3d[1]) * epsilon
        for epsilon in np.linspace(0, 1, 200)
    ]
    for target in targets:
        for epsilon in np.linspace(0, 1, 200):
            point = coord_3d[0] + (target - coord_3d[0]) * epsilon
            # check that the point is in the mask
            if (
                int(point[0]) < mask.shape[0]
                and int(point[1]) < mask.shape[1]
                and int(point[2]) < mask.shape[2]
            ):
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

    im_rot, mask_rot = _rotate_im_and_mask(im_cropped, mask_cropped, angle, (1, 2))

    # Update vertice positions
    axes = [1, 2]
    middle_voxel_before = np.asarray(mask_cropped.shape) / 2
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = _rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            _rotate_point,
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

    im_rot, mask_rot = _rotate_im_and_mask(im_rot, mask_rot, angle, (0, 2))

    # Update vertice positions
    axes = [0, 2]
    middle_voxel_before = middle_voxel_after.copy()
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = _rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            _rotate_point,
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

    im_rot, mask_rot = _rotate_im_and_mask(im_rot, mask_rot, angle, (0, 1))

    # Update vertice positions
    axes = [0, 1]
    middle_voxel_before = middle_voxel_after.copy()
    middle_voxel_after = np.asarray(mask_rot.shape) / 2
    n_triangle = _rotate_point(n_triangle, axes, np.zeros(3), np.zeros(3), angle)
    coord_3d = list(
        map(
            _rotate_point,
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
        n_layers - (triangle_slice - first_slice) : n_layers
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


def _rotate_im_and_mask(im, mask, angle, axes):
    mask_rot = ndi.rotate(
        mask,
        angle / (2 * np.pi) * 360,
        axes=axes,
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    im_rot = ndi.rotate(
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


def _rotate_point(vector, axes, middle_point_before, middle_point_after, angle):
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


def draw_triangles_in_3d_and_2d(verts, faces, verts_2d, faces_2d, dict_2d_3d, im_shape):
    # Create 3D image
    triangles_3d = np.zeros(im_shape)

    # Create 2D image
    x = [vert[0] for vert in verts_2d]
    pix_x0 = np.min(x)
    pix_xw = np.max(x) - pix_x0
    y = [vert[1] for vert in verts_2d]
    pix_y0 = np.min(y)
    pix_yw = np.max(y) - pix_y0

    triangles_2d = np.zeros([int(pix_xw) + 5, int(pix_yw) + 5, 1])
    del x, y

    colour = 1

    # Loop through all 2d triangles
    for face_2d in tqdm(faces_2d[:]):
        # Get coordinates of triangle in both the 2d and the 3d image
        coord_2d = [verts_2d[vert].copy() - [pix_x0, pix_y0] for vert in face_2d]
        coord_3d = [
            verts[vert].copy() for vert in [dict_2d_3d[vert_id] for vert_id in face_2d]
        ]

        if triangle_area(coord_3d) > 1e-10:
            # Draw on the 3D image
            coord_3d = [np.reshape(coord, [3, 1]) for coord in coord_3d]
            targets = coord_3d[1] + (coord_3d[2] - coord_3d[1]) * np.reshape(
                np.linspace(0, 1, 200), [1, -1]
            )
            for i in range(targets.shape[1]):
                target = targets[:, i : i + 1]
                points = (
                    coord_3d[0]
                    + (target - coord_3d[0])
                    * np.reshape(np.linspace(0, 1, 200), [1, -1])
                ).astype(int)
                # remove points that are not in the image
                points = np.squeeze(
                    points[:, np.where(np.sum((points < 0), axis=0) == 0)]
                )
                points = np.squeeze(
                    points[
                        :,
                        np.where(
                            np.sum(
                                (
                                    (np.reshape(np.asarray(im_shape), [3, 1]) - points)
                                    <= 0
                                ),
                                axis=0,
                            )
                            == 0
                        ),
                    ]
                )
                triangles_3d[points[0, :], points[1, :], points[2, :]] = colour

            # Draw on the 2D image
            coord_2d = [np.reshape(coord, [2, 1]) for coord in coord_2d]
            for ratio in [0.96, 0.97, 0.98]:
                point_a = coord_2d[0] + (coord_2d[1] - coord_2d[0]) * ratio
                point_b = coord_2d[0] + (coord_2d[2] - coord_2d[0]) * ratio
                points = point_a + (point_b - point_a) * np.reshape(
                    np.linspace(0, 1, 200), [1, -1]
                )
                triangles_2d[points[0, :].astype(int), points[1, :].astype(int)] = (
                    colour
                )

                point_a = coord_2d[1] + (coord_2d[0] - coord_2d[1]) * ratio
                point_b = coord_2d[1] + (coord_2d[2] - coord_2d[1]) * ratio
                points = point_a + (point_b - point_a) * np.reshape(
                    np.linspace(0, 1, 200), [1, -1]
                )
                triangles_2d[points[0, :].astype(int), points[1, :].astype(int)] = (
                    colour
                )

                point_a = coord_2d[2] + (coord_2d[1] - coord_2d[2]) * ratio
                point_b = coord_2d[2] + (coord_2d[0] - coord_2d[2]) * ratio
                points = point_a + (point_b - point_a) * np.reshape(
                    np.linspace(0, 1, 200), [1, -1]
                )
                triangles_2d[points[0, :].astype(int), points[1, :].astype(int)] = (
                    colour
                )

        colour += 1

    return triangles_3d, triangles_2d


def _points_in_triangle(grid, triangle):
    """
    List the points of the grid inside the triangle

    Parameters
    ----------
    grid: ndarray (N,3)
        3D grid
    triangle: ndarray (3,2)
        2D Triangle

    Returns
    -------
    Indices
    points: ndarray (N,3)
        Coordinates of the points in the triangle
    """

    # find points of the grid in the triangle bounding box
    lower = np.min(triangle, axis=0)
    upper = np.max(triangle, axis=0)
    idx = np.where(
        np.logical_and(
            np.all(grid[:, :2] >= lower, axis=1), np.all(grid[:, :2] <= upper, axis=1)
        )
    )[0]

    # find points in the triangle
    # https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    x, y = grid[idx, 0], grid[idx, 1]
    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]
    side_1 = np.sign((x - bx) * (ay - by) - (ax - bx) * (y - by))
    side_2 = np.sign((x - cx) * (by - cy) - (bx - cx) * (y - cy))
    side_3 = np.sign((x - ax) * (cy - ay) - (cx - ax) * (y - ay))
    # take into account points on the edge
    inside = np.ones(side_1.shape, dtype=bool)
    inside[np.logical_and(side_1 == 1, side_2 == -1)] = False
    inside[np.logical_and(side_1 == -1, side_2 == 1)] = False
    inside[np.logical_and(side_1 == 1, side_3 == -1)] = False
    inside[np.logical_and(side_1 == -1, side_3 == 1)] = False
    # inside = np.all(np.stack((side_1 == side_2, side_1 == side_3), axis=1), axis=1)
    return idx[inside]


def _map_point_in_triangle(points, triangle1, triangle2):
    """
    Map a point in triangle1 to a point in triangle2

    Parameters
    ----------
    points: ndarray (N,3)
    triangle1: ndarray (3,2)
    triangle2: ndarray (3,3)

    Returns
    -------
    point: ndarray (N,3)
    """
    # basis change from xyz to triangle 1 system
    P = np.stack(
        (
            triangle1[1] - triangle1[0],
            triangle1[2] - triangle1[0],
            np.cross(triangle1[1] - triangle1[0], triangle1[2] - triangle1[0]),
        ),
        axis=1,
    )
    # basis change from xyz to triangle 2 system
    Q = np.stack(
        (
            triangle2[1] - triangle2[0],
            triangle2[2] - triangle2[0],
            np.cross(triangle2[1] - triangle2[0], triangle2[2] - triangle2[0]),
        ),
        axis=1,
    )
    # return the combination of basis and origin change
    return (
        triangle2[0].reshape(3, 1)
        + Q @ np.linalg.inv(P) @ (points.T - triangle1[0].reshape(3, 1))
    ).T


def extract_layers(
    verts3d, verts2d, faces, image, spacing=[1, 1, 1], layers=np.linspace(-1, 1, 3)
):
    """
    Extract layers from an image around a 3D mesh projected in 2D

    Parameters
    ----------
    verts3d: ndarray (N3,3)
        Vertices of the 3D mesh  (x,y,z)
    verts2d: ndarray (N2,2)
        Vertices of the 2D mesh (x,y)
    faces: ndarray (M,3)
        Faces of the mesh (triangles)
    image: ndarray
        Input 3D array
    spacing: list
        Pixel spacing of the image
    layers: ndarray (K,)
        Position of layers to extract

    Results
    -------
    extracted_layers: ndarray

    """

    if image.ndim != 3:
        raise Exception(f"Image dimentionality ({image.ndim}) should be 3.")
    if verts3d.shape[1] != 3:
        raise Exception(f"verts3d shape ({verts3d.shape}) should be (N,3)")
    if verts2d.shape[1] != 2:
        raise Exception(f"verts3d shape ({verts3d.shape}) should be (N,2)")
    if faces.shape[1] != 3:
        raise Exception(f"faces shape ({verts3d.shape}) should be (M,3)")

    spacing = np.array(spacing)
    verts3d = verts3d / spacing
    verts2d = verts2d / spacing[:2]

    # compute the shape of the accumulators from verts2d and the number of layers
    lower = np.floor(np.min(verts2d, axis=0)).astype(int)
    upper = np.ceil(np.max(verts2d, axis=0)).astype(int)
    bbox = upper - lower
    shape = [bbox[0], bbox[1], len(layers)]  # yxz
    acc1 = np.zeros(shape)
    acc2 = np.zeros(shape)

    # compute the 2d xyz pixel grid as N x[x,y,z]
    grid = np.stack(
        [v.ravel() for v in np.meshgrid(*[np.arange(n) for n in shape], indexing="xy")],
        1,
    )

    for face in faces:
        # compute the coordinates of the 2d triangle
        triangle2d = np.zeros((3, 3))
        triangle2d[:, :2] = verts2d[face, :] - lower
        # list points [xyz] of the grid inside the 2d triangle
        idx = _points_in_triangle(grid, triangle2d[:, :2])
        if len(idx) > 0:
            # compute the coordinates of the 3d triangle
            triangle3d = verts3d[face, :]
            # compute the coordinate applying layer
            dst = grid[idx, :]
            dst[:, -1] = layers[dst[:, -1]]
            # get 3d points the corresponding matching 2d points
            src = _map_point_in_triangle(dst, triangle2d, triangle3d)
            # get the values in the initial image
            values = ndi.map_coordinates(image, src.T)
            # add the value in the accumulator
            acc1[grid[idx, 0], grid[idx, 1], grid[idx, 2]] += values
            acc2[grid[idx, 0], grid[idx, 1], grid[idx, 2]] += values > 0
    # normalize the accumulator
    acc1[acc2 > 1] /= acc2[acc2 > 1]
    extracted_layers = acc1
    return extracted_layers
