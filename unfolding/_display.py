"""Display tools for unfolded tesselation"""

import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact


def show_3d_and_contours(im, edges, scale):
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
