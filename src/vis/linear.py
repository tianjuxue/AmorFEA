import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from ..opt.optimizer_robot import heart_shape
from .. import arguments


def plot_sol(args, name):
    path = args.root_path + '/' + args.solutions_path + '/linear/' + name + '000000.vtu'  
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
    fig = plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal')
    plt.axis('off')
    tpc = plt.tripcolor(x[:,0], x[:,1], tri, colors, shading='flat', vmin=None, vmax=None)
    cb = plt.colorbar(tpc, aspect=10)
    cb.ax.tick_params(labelsize=20)


def plot_mesh(args):
    path = args.root_path + '/' + args.solutions_path + '/linear/u000000.vtu'  
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    triangles =  vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size//4  # number of cells
    tri = np.take(triangles, [n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)    
    fig = plt.figure(figsize=(8, 8))
    plt.triplot(x[:,0], x[:,1], tri, linewidth=1, color='r') 
    name = 'mesh'
    plt.gca().set_aspect('equal')
    plt.axis('off')
    # plt.axis('equal')
    # plt.xlim(-2, 3.5)
    # plt.ylim(-0.5, 11)
    # plt.legend(loc='right')
    # fig.savefig(args.root_path + '/images/' + name + '.png', bbox_inches='tight')

def plot_L(args):
    path_a = args.root_path + '/' + args.numpy_path + '/linear/L_inf_a.npy' 
    path_s = args.root_path + '/' + args.numpy_path + '/linear/L_inf_s.npy' 
    L_inf_a = np.load(path_a)[:-1]
    L_inf_s = np.load(path_s)[:-1]  
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L_inf_a), 1)
    ax.plot(epoch, L_inf_a, linestyle='--', marker='o', color='red', label='AFEM')
    ax.plot(epoch, L_inf_s, linestyle='--', marker='o', color='blue', label='Supervised Training')
    ax.set_yscale('log')
    ax.legend(loc='upper right', prop={'size': 12})
    ax.tick_params(labelsize=14)
    # plt.yticks(np.arange(min(L_inf_a), max(L_inf_a)+1, 1.0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


if __name__ == '__main__':
    args = arguments.args
    # plot_L(args)
    plot_sol(args, 'f')
    plot_sol(args, 'u')
    plt.show()