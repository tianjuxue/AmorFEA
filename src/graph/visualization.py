import numpy as np
import matplotlib.pyplot as plt
import torch
import mpl_toolkits.mplot3d as plt3d
from matplotlib.collections import LineCollection
from ..graph.domain import GraphManual, GraphMSHR
from .. import arguments

def f1(x1, x2):
    return np.ones_like(x1)

def f2(x1, x2):
    return np.sin(2*np.pi*x1)

def get_graph_attributes(func, graph):
    return func(graph.x1, graph.x2)

def model_prediction(source, graph, model):
    source /= graph.num_vertices
    boundary = np.zeros(graph.num_vertices)
    source = np.matmul(graph.reset_matrix_boundary, boundary) + np.matmul(graph.reset_matrix_interior, source)    
    source = torch.tensor(source, dtype=torch.float).view(1, -1)
    solution = model(source)
    return np.squeeze(solution.data.numpy())

def scalar_field_3D(attribute, graph):
    max_att = np.max(attribute)
    min_att = np.min(attribute)
    range_att = max_att - min_att
    # print(np.max(attribute))

    fig = plt.figure(0)
    # fig.set_size_inches(10, 10)
    ax = fig.gca(projection='3d')

    for i in range(graph.num_vertices):
        for j in graph.ordered_adjacency_list[i]:
            x1 = (graph.x1[i], graph.x1[j])
            x2 = (graph.x2[i], graph.x2[j])
            x3 = (attribute[i], attribute[j])
            line = plt3d.art3d.Line3D(x1, x2, x3, 
                                      color=[(x3[0] - min_att)/range_att, 0., 1 - (x3[0] - min_att)/range_att],
                                      linewidth=1)
            ax.add_line(line)

    ax.set_zlim3d(0, 0.1)

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


def plot_training(L_inf, L_fro):
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L_inf), 1)
    ax.plot(epoch, L_inf)
    # ax.set_yscale('log')
    plt.show()

if __name__ == "__main__":
    args = arguments.args
    graph = GraphManual(args)
    model_path = args.root_path + '/' + args.model_path + '/linear/model_6'

    model =  torch.load(model_path)
    source = get_graph_attributes(f1, graph)
    solution = model_prediction(source, graph, model)

    grad_u_1 = np.matmul(graph.gradient_x1, solution)
    grad_u_2 = np.matmul(graph.gradient_x2, solution)

    scalar_field_3D(solution, graph)
    scalar_field_2D(solution, graph)
    vector_field_2D(grad_u_1, grad_u_2, graph)
    
    # np_data = np.load(args.root_path + '/' + args.numpy_path + '/linear/error_11.npy')
    # L_inf = np.trim_zeros(np_data[0, :], 'b')
    # L_fro = np.trim_zeros(np_data[1, :], 'b')
    # plot_training(L_inf, L_fro)

    plt.show()

