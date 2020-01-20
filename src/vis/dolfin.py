import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from .. import arguments

def read_data(args, name):
    path = args.root_path + '/' + args.solutions_path + '/dolfin/' + name + '000000.vtu'  
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    u = vtk_to_numpy(data.GetPointData().GetVectors('u'))
    colors = u
    triangles =  vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size//4  # number of cells
    tri = np.take(triangles, [n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)
    return x, u, tri

def plot_dolfin(args, x, u, tri, name, mesh_flag, vmin=None, vmax=None):
    fig = plt.figure(figsize=(8, 8))
    colors = u
    if mesh_flag:
        plt.triplot(x[:,0], x[:,1], tri, linewidth=1, color='r') 
        name = 'mesh'
    else:
        # tpc = plt.tripcolor(x[:,0], x[:,1], tri, colors, shading='gouraud')
        tpc = plt.tripcolor(x[:,0], x[:,1], tri, colors, shading='flat', vmin=vmin, vmax=vmax)
        # plt.colorbar(tpc)

    plt.gca().set_aspect('equal')
    plt.axis('off')
    # plt.axis('equal')
    # plt.xlim(-2, 3.5)
    # plt.ylim(-0.5, 11)
    # plt.legend(loc='right')
    fig.savefig(args.root_path + '/others/' + name + '.png', bbox_inches='tight')


def plot_error(args):
    x, u_fem, tri = read_data(args, 'opt_fem_f')
    _, u_ad, _ = read_data(args, 'opt_ad_f')   
    _, u_nn, _ = read_data(args, 'opt_nn_f')
    print(np.max(np.abs(u_ad-u_nn)))
    plot_dolfin(args, x, np.abs(u_fem - u_ad), tri, 'ad_error', False, 0, 80)
    plot_dolfin(args, x, np.abs(u_fem - u_nn), tri, 'nn_error', False, 0, 80)


def plot_from_file(args, name, mesh_flag):
    x, u, tri = read_data(args, name)
    plot_dolfin(args, x, u, tri, name, mesh_flag, 0, 80)


if __name__ == '__main__':
    args = arguments.args
    plot_from_file(args, 'opt_nn_f', True)
    plot_from_file(args, 'opt_nn_f', False)
    plot_from_file(args, 'opt_ad_f', False)
    plot_from_file(args, 'opt_fem_f', False)
    plot_error(args)
    plt.show()
