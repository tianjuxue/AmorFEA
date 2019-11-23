import numpy as np
import matplotlib.pyplot as plt
import torch
import mpl_toolkits.mplot3d as plt3d
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

def plot_graph_attributes(attribute, graph):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.gca(projection='3d')

    print(np.max(attribute))

    max_att = np.max(attribute)
    min_att = np.min(attribute)
    range_att = max_att - min_att

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

    plt.show()


def plot_training(L_inf, L_fro):
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L_inf), 1)
    ax.plot(epoch, L_fro)
    ax.set_yscale('log')
    plt.show()

if __name__ == "__main__":
    args = arguments.args
    graph = GraphManual(args)
    model_path = args.root_path + '/' + args.model_path + '/model_11'

    model =  torch.load(model_path)
    source = get_graph_attributes(f1, graph)
    solution = model_prediction(source, graph, model)
    # plot_graph_attributes(solution, graph)


    np_data = np.load(args.root_path + '/' + args.numpy_path + '/error_11.npy')
    L_inf = np.trim_zeros(np_data[0, :], 'b')
    L_fro = np.trim_zeros(np_data[1, :], 'b')

    plot_training(L_inf, L_fro)


