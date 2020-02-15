import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def get_topology(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('u'))
    triangles = vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size // 4  # number of cells
    tri = np.take(triangles, [n for n in range(
        triangles.size) if n % 4 != 0]).reshape(ntri, 3)
    return x, u, tri


def plot_mesh_general(path):
    x, _, tri = get_topology(path)
    fig = plt.figure(figsize=(8, 8))
    plt.triplot(x[:, 0], x[:, 1], tri, linewidth=1, color='r')
    name = 'mesh'
    plt.gca().set_aspect('equal')
    plt.axis('off')
