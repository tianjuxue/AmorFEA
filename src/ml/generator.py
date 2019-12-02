import numpy as np
import math
from .. import arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from ..graph.domain import GraphManual, GraphMSHR, GraphMSHRTrapezoid
from ..graph.visualization import *


def generate_gaussian_samples(args, graph, num_samps, linear_flag):

    def RBF_kernel(x1, x2):
        sigma = 1
        l = 0.5
        k = sigma**2 * np.exp( -np.linalg.norm(x1 - x2)**2 / (2*l**2) )
        return k
     
    def mean(x):
        return 0.

    M = graph.num_vertices
    coo = graph.coo
    kernel_matrix = np.zeros((M, M))
    mean_vector = np.zeros(M)

    for i in range(M):
        mean_vector[i] = mean(coo[i])
        for j in range(M):
            kernel_matrix[i][j] = RBF_kernel(coo[i], coo[j])

    samples = np.random.multivariate_normal(mean_vector, kernel_matrix, num_samps)

    save_path = '/linear/' if linear_flag else '/nonlinear/'
    np.save(args.root_path + '/' + args.numpy_path + save_path + graph.name +
            '-Gaussian-' + str(num_samps) + '-' + str(M) + '.npy', samples)

    return samples

def generate_uniform_samples(args, graph, num_samps):
    M = graph.num_vertices
    samples = np.random.uniform(-1, 1, (num_samps, M))
    np.save(args.root_path + '/' + args.numpy_path + '/' + graph.name +
            '-Uniform-' + str(num_samps) + '-' + str(M) + '.npy', samples)

def generate_multinomial_samples(args, graph, num_samps):
    M = graph.num_vertices
    samples = np.random.multinomial(1, [1/M]*M, size=num_samps)
    np.save(args.root_path + '/' + args.numpy_path + '/' + graph.name +
            '-Multi-' + str(num_samps) + '-' + str(M) + '.npy', samples)


def f1(x1, x2):
    return np.ones_like(x1)

def f2(x1, x2):
    return np.sin(np.pi*x1) + 2

def get_graph_attributes(func, graph):
    return func(graph.x1, graph.x2)

def generate_deterministic_samples(args, graph, num_samps):
    M = graph.num_vertices
    samples = get_graph_attributes(f2, graph)
    samples = np.expand_dims(samples, axis=0)
    samples = np.repeat(samples, num_samps, axis=0)
    np.save(args.root_path + '/' + args.numpy_path + '/nonlinear/' + graph.name +
            '-Det-' + str(num_samps) + '-' + str(M) + '.npy', samples)

def linear_visual(args, graph):
    model_path = args.root_path + '/' + args.model_path + '/linear/model_6'
    model =  torch.load(model_path)
    source = get_graph_attributes(f1, graph)
    solution = model_prediction(source, graph, model)
    grad_u_1 = np.matmul(graph.gradient_x1, solution)
    grad_u_2 = np.matmul(graph.gradient_x2, solution)
    scalar_field_3D(solution, graph)
    scalar_field_2D(solution, graph)
    vector_field_2D(grad_u_1, grad_u_2, graph)
    np_data = np.load(args.root_path + '/' + args.numpy_path + '/linear/error_11.npy')
    L_inf = np.trim_zeros(np_data[0, :], 'b')
    L_fro = np.trim_zeros(np_data[1, :], 'b')
    plot_training(L_inf, L_fro)


if __name__ == '__main__':
    args = arguments.args
    graph = GraphMSHRTrapezoid(args)
    # generate_deterministic_samples(args, graph, 3000)
    samples = generate_gaussian_samples(args, graph, 1000, False)

    scalar_field_3D(samples[0], graph)
    plt.show()

    # graph = GraphManual(args)
    # exp_visual(args, graph)
    # plt.show()
