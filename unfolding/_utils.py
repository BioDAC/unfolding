import numpy as np


def dummy():
    """Create a dummy image and label"""
    x, y, z = np.meshgrid(
        np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    )

    image = x + y + z

    label = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) < 0.2**2

    return image, label


def triangle_area(vertices):
    return (
        np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
        / 2
    )
