"""Display tools for unfolded tesselation"""

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from ._utils import triangle_area


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


def show_3d_and_contours(im, edges, scale):
    try:
        import ipywidgets as widgets
        from ipywidgets import interact
    except ModuleNotFoundError:
        Warning.warn("ipywidget not found")
        return

    n_slices = im.shape[2]
    max_label = np.max(edges)

    def update(slice_num):
        slice_num = np.round(slice_num)
        plt.figure()
        plt.imshow(
            im[:, :, int(slice_num)],
            cmap="gray",
            interpolation="none",
            vmin=scale[0],
            vmax=scale[1],
        )
        plt.imshow(
            edges[:, :, int(slice_num)],
            vmax=max_label,
            cmap="prism",
            alpha=0.5 * (edges[:, :, int(slice_num)] > 0),
            interpolation="none",
        )
        plt.show()

    slice_widget = widgets.FloatSlider(
        value=int(n_slices / 2),
        min=0,
        max=n_slices - 1,
        step=1,
        description="Slice: ",
        disabled=False,
        continuous_update=True,
        orientation="horizontal",
        readout=True,
        readout_format="",
        style={"description_width": "initial"},
    )

    interact(update, slice_num=slice_widget)
