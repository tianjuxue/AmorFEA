import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from .. import arguments
from ..opt.trajectories import heart_shape

def plot_robot_and_trajectory(args):
    x_s, y_s = heart_shape()
    x_s, y_s = x_s + 0.25, y_s + 10 
    index = 61
    x_r = []
    y_r = []
    for i in range(len(x_s)):
        path = args.root_path + '/' + args.solutions_path + '/time_series_gt/u' + str(i) + '000000.vtu'  
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        data = reader.GetOutput()
        points = data.GetPoints()
        npts = points.GetNumberOfPoints()
        x = vtk_to_numpy(points.GetData())
        u = vtk_to_numpy(data.GetPointData().GetVectors('u'))
        x = x + u

        x_r.append(x[index][0])
        y_r.append(x[index][1])

        triangles =  vtk_to_numpy(data.GetCells().GetData())
        ntri = triangles.size//4  # number of cells
        tri = np.take(triangles, [n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)
        x_s, y_s = heart_shape()
        x_s, y_s = x_s + 0.25, y_s + 10 

        fig = plt.figure(figsize=(8, 8))
        # plt.triplot(x[:,0], x[:,1], tri, linewidth=0.2, color='r')    
        plt.triplot(x[:,0], x[:,1], tri, marker='o', markersize=1, linewidth=1, color='orange')
        plt.plot(x_s[:index + 1], y_s[:index + 1])
        plt.plot(x_s, y_s, color='blue', label='prescribed')
        plt.plot(x_r, y_r, color='red', label='optimized')
        plt.gca().set_aspect('equal')
        plt.axis('off')
        # plt.axis('equal')
        plt.xlim(-2, 3.5)
        plt.ylim(-0.5, 11)
        plt.legend(loc='right')
        fig.savefig(args.root_path + '/others/trajectory' + f'{i:04}' + '.png', bbox_inches='tight')


if __name__ == '__main__':
    args = arguments.args
    plot_robot_and_trajectory(args)
    plt.show()
