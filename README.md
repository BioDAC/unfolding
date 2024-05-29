*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*Unfolding is distributed under the terms of the GNU General Public License*

ABOUT
-------------------------------------------------------------------------------
Unfolding is a Python-based tool for unfolding 3D surfaces in a grayscale image
into a 2D plane. It requires the 3D grayscale image and a binary segmentation as
inputs. The surface of the segmentation is tessellated and layers parallel to
the surface are extracted from the grayscale image, and unfolded into 2D.

INSTALLATION
-------------------------------------------------------------------------------

To install the package from github directly:
```bash
pip install git+https://github.com/akarsa/unfolding
```
or clone the repository and install the package:
```bash
git clone https://github.com/akarsa/unfolding
cd unfolding
pip install -e .
```


HOW TO USE
-------------------------------------------------------------------------------

To perform unfolding (see src/Unfolding.py):

1. Load or define image and label:
```python
import unfolding 
image, label = unfolding.sphere()
```

2. Create simplified tessellation
```python
verts, faces = unfolding.mesh_from_labels(label, num_vertices=30)
````

where 
* _**num_vertices**_ is target number of vertices in the simplified mesh.

3. Unfold the mesh

```python
verts_2d, faces_2d, dict_2d_3d = unfolding.unfold(
    verts, faces, base_triangle=0, draw=False)
```

where
* _**base_triangle**_ is the index of the row in faces that contains the first triangle
to consider (this will be the middle of the unfolded surface)
* _**draw**_ takingTrue or False indicating whether the function should plot the unfolded mesh or not

4. Finaly, extract the layers:
```python
layers = unfold_layers(
    verts, faces, verts_2d, faces_2d, dict_2d_3d, image, n_layers=20
)
```
where _**n_layers**_ number of layers to be exported on both sides of the surface (i.e. layers will have 2*n_layers+1 slices)

HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
```bibtex
@software{unfolding,
  author       = {Anita Karsa},
  title        = {{Unfolding}},
  month        = jan,
  year         = 2024,
  url 	       = {https://github.com/akarsa/unfolding}
}
```

DEPENDENCIES
-------------------------------------------------------------------------------
* numpy (https://numpy.org)
* scipy (https://scipy.org)
* matplotlib (https://matplotlib.org)
* scikit-image (https://scikit-image.org)
* tqdm (https://tqdm.github.io)
* pymeshlab (https://pymeshlab.readthedocs.io)
* fast_simplification


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.<br>
Dept. of Physiology, Development, and Neuroscience<br>
University of Cambridge,v
Cambridge, UK<br>
ak2557@cam.ac.uk
