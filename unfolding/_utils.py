import numpy as np
from scipy.spatial import Delaunay


def sphere():
    """Create a sphere image and label

    Returns
    -------
    image: ndarray
        intensity 3d image
    label : ndarray
        binary image with bounday corresponding to the surface to extract

    """
    v = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(v, v, v)

    image = x + y + z

    label = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) < 0.2**2

    return image, label


def half_sphere(n=10):
    """Create a dumy test example with a halph sphere

    Parameters
    ----------
    n : int
        Number of point along each axis

    Returns
    -------
    image: ndarray
        intensity 3d image
    label : ndarray
        binary image with bounday corresponding to the surface to extract
    verts: ndarray (N,3)
        vertices of the mesh
    faces: ndarray (M,3)
        triangles of the mesh
    """
    vi = np.linspace(0, 100, n)
    xi, yi = np.meshgrid(vi, vi, indexing="xy")
    zi = 100 - np.sqrt(np.maximum(0, 50 * 50 - (xi - 50) ** 2 - (yi - 50) ** 2))
    verts = np.stack((xi.ravel(), yi.ravel(), zi.ravel()), axis=1)
    tri = Delaunay(verts)
    faces = tri.simplices

    v = np.arange(100)
    x, y, z = np.meshgrid(v, v, v, indexing="ij")
    label = (x - 50) ** 2 + (y - 50) ** 2 < 51 * 51 - (z - 100) ** 2
    image = 1 + (
        np.cos(2 * np.pi * x / 100)
        * np.cos(6 * np.pi * y / 100)
        * np.cos(np.pi * z / 100)
    )
    return image, label, verts, faces


def conic(n=10):
    """Create a dumy test example with a conic

    Parameters
    ----------
    n : int
        Number of point along each axis

    Returns
    -------
    image: ndarray
        intensity 3d image
    label : ndarray
        binary image with bounday corresponding to the surface to extract
    verts: ndarray (N,3)
        vertices of the mesh
    faces: ndarray (M,3)
        triangles of the mesh
    """
    vi = np.linspace(0, 100, n)
    xi, yi = np.meshgrid(vi, vi, indexing="xy")
    zi = ((xi - 50) ** 2 + (yi - 50) ** 2) / 50
    verts = np.stack((xi.ravel(), yi.ravel(), zi.ravel()), axis=1)
    tri = Delaunay(verts)
    faces = tri.simplices

    v = np.arange(100)
    x, y, z = np.meshgrid(v, v, v, indexing="ij")
    label = ((x - 50) ** 2 + (y - 50) ** 2) / 50 < z
    image = 1 + (
        np.cos(2 * np.pi * x / 100)
        * np.cos(6 * np.pi * y / 100)
        * np.cos(np.pi * z / 100)
    )
    return image, label, verts, faces


def plane():
    """Create a dumy test example with a plane

    Parameters
    ----------
    n : int
        Number of point along each axis

    Returns
    -------
    image: ndarray
        intensity 3d image
    label : ndarray
        binary image with bounday corresponding to the surface to extract
    verts: ndarray (N,3)
        vertices of the mesh
    faces: ndarray (M,3)
        triangles of the mesh

    """
    vi = np.linspace(0, 100, 10)
    xi, yi = np.meshgrid(vi, vi, indexing="xy")
    zi = xi
    verts = np.stack((xi.ravel(), yi.ravel(), zi.ravel()), axis=1)
    tri = Delaunay(verts[:, :2])
    faces = tri.simplices

    v = np.arange(100)
    x, y, z = np.meshgrid(v, v, v, indexing="ij")
    label = x > z
    image = 1 + (
        np.cos(2 * np.pi * x / 100)
        * np.cos(4 * np.pi * y / 100)
        * np.cos(np.pi * z / 100)
    )
    return image, label, verts, faces


def flat():
    """Create a flat test example

    Parameters
    ----------
    n : int
        Number of point along each axis

    Returns
    -------
    image: ndarray
        intensity 3d image
    label : ndarray
        binary image with bounday corresponding to the surface to extract
    verts: ndarray (N,3)
        vertices of the mesh
    faces: ndarray (M,3)
        triangles of the mesh

    """

    xi, yi = np.meshgrid(np.linspace(0, 150, 5), np.linspace(0, 100, 5), indexing="xy")
    zi = 60 * np.ones(xi.shape)
    verts = np.stack((xi.ravel(), yi.ravel(), zi.ravel()), axis=1)
    tri = Delaunay(verts[:, :2])
    faces = tri.simplices

    x, y, z = np.meshgrid(np.arange(150), np.arange(100), np.arange(120), indexing="ij")
    label = z > 60
    image = 1 + (
        np.cos(2 * np.pi * x / 100)
        * np.cos(6 * np.pi * y / 100)
        * np.cos(2 * np.pi * z / 100)
    )
    return image, label, verts, faces


def triangle_area(vertices):
    return (
        np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
        / 2
    )
