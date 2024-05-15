*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*Unfolding is distributed under the terms of the GNU General Public License*

ABOUT
-------------------------------------------------------------------------------
Unfolding is a Python-based tool for unfolding 3D surfaces in a grayscale image
into a 2D plane. It requires the 3D grayscale image and a binary segmentation as
inputs. The surface of the segmentation is tessellated and layers parallel to
the surface are extracted from the grayscale image, and unfolded into 2D.


HOW TO USE
-------------------------------------------------------------------------------
First, install required packages (see Dependencies).

To perform unfolding (see src/Unfolding.py):

1. Load or define image and label:
image, label = create_dummy()

2. Create simplified tessellation
verts, faces = create_simplified_tessellation(label, num_vertices=30)
* num_vertices: target number of vertices in the simplified tessellation

3. Unfold tessellation
verts_2d, faces_2d, dict_2d_3d = unfold_tessellation(

    verts, faces, base_triangle=0, draw=0

)
* base_triangle: the index of the row in faces that contains the first triangle
to consider (this will be the middle of the unfolded surface)
* draw: 0 or 1 indicating whether the function should plot the unfolded
tessellation or not

4. Unfold and extract layers
layers = unfolded_layers(
    verts, faces, verts_2d, faces_2d, dict_2d_3d, image, n_layers=20
)
* n_layers: number of layers to be exported on both sides of the surface
(i.e. layers will have 2*n_layers+1 slices)

HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
@software{unfolding,

  author       = {Anita Karsa},

  title        = {{Unfolding}},

  month        = jan,

  year         = 2024,

  url 	       = {https://github.com/akarsa/unfolding}

}

INSTALLATION
-------------------------------------------------------------------------------

To install the package:
```bash
pip install git+https://github.com/akarsa/unfolding
```


DEPENDENCIES
-------------------------------------------------------------------------------
numpy (https://numpy.org)

scipy (https://scipy.org)

matplotlib (https://matplotlib.org)

scikit-image (https://scikit-image.org)

tqdm (https://tqdm.github.io)

pymeshlab (https://pymeshlab.readthedocs.io)


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.

Dept. of Physiology, Development, and Neuroscience

University of Cambridge,

Cambridge, UK

ak2557@cam.ac.uk
