import numpy as np
import matplotlib.pyplot as plt
import torch
import mpl_toolkits.mplot3d as plt3d
from matplotlib.collections import LineCollection
import fenics as fa


def save_solution(args, solution, name):
    file = fa.File(args.root_path + '/' + args.solutions_path + '/' + name + '.pvd')
    solution.rename('u', 'u')
    file << solution

def model_prediction(source, graph, model):
    source /= graph.num_vertices
    boundary = np.zeros(graph.num_vertices)
    source = np.matmul(graph.reset_matrix_boundary, boundary) + np.matmul(graph.reset_matrix_interior, source)    
    source = torch.tensor(source, dtype=torch.float).view(1, -1)
    solution = model(source)
    return np.squeeze(solution.data.numpy())

def scalar_field_paraview(args, attribute, pde, name):
    solution = fa.Function(pde.V)
    solution.vector()[:] = attribute
    save_solution(args, solution, name)
 
def scalar_field_3D(attribute, graph):
    max_att = np.max(attribute)
    min_att = np.min(attribute)
    range_att = max_att - min_att
    # print(np.max(attribute))

    fig = plt.figure()
    # fig.set_size_inches(10, 10)
    ax = plt.axes(projection='3d')

    for i in range(graph.num_vertices):
        for j in graph.ordered_adjacency_list[i]:
            x1 = (graph.x1[i], graph.x1[j])
            x2 = (graph.x2[i], graph.x2[j])
            x3 = (attribute[i], attribute[j])
            line = plt3d.art3d.Line3D(x1, x2, x3, 
                                      color=[(x3[0] - min_att)/range_att, 0., 1 - (x3[0] - min_att)/range_att],
                                      linewidth=1)
            ax.add_line(line)

    ax.autoscale(enable=True, axis='both', tight=None)
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(-1, 2)
    ax.set_zlim3d(-0.5, 0.5)
 
def scalar_field_2D(attribute, graph):
    max_att = np.max(attribute)
    min_att = np.min(attribute)
    range_att = max_att - min_att

    fig = plt.figure(1)
    ax = fig.gca()
    lines = [[(graph.x1[i], graph.x2[i]), (graph.x1[j], graph.x2[j])] 
                for i in range(graph.num_vertices)
                    for j in graph.ordered_adjacency_list[i]]

    c = [ [(attribute[i] - min_att)/range_att, 0., 1 - (attribute[i] - min_att)/range_att, 1]
                for i in range(graph.num_vertices)
                    for j in graph.ordered_adjacency_list[i]]

    lc = LineCollection(lines, color=c)
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.autoscale()


def vector_field_2D(attribute_x1, attribute_x2, graph):
    fig = plt.figure(2)
    ax = fig.gca()
    x_pos = graph.x1
    y_pos = graph.x2
    x_direct = attribute_x1
    y_direct = attribute_x2
    ax.quiver(x_pos, y_pos, x_direct, y_direct, scale=None)
    ax.set_aspect('equal')


def plot_training(L):
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L), 1)
    ax.plot(epoch, L)
    ax.set_yscale('log')

