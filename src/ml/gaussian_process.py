import numpy as np
import math
from .. import arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from ..graph.domain import GraphManual, GraphMSHR


def generate_samples(graph, num_samps):
    M = graph.num_vertices
    coo = graph.coo
    kernel_matrix = np.zeros((M, M))
    mean_vector = np.zeros(M)

    for i in range(M):
        mean_vector[i] = mean(coo[i])
        for j in range(M):
            kernel_matrix[i][j] = RBF_kernel(coo[i], coo[j])

    samples = np.random.multivariate_normal(mean_vector, kernel_matrix, num_samps)
    samples = np.random.uniform(-1, 1, (num_samps, M))
    samples = np.random.multinomial(1, [1/M]*M, size=num_samps)

    return samples

def RBF_kernel(x1, x2):
    sigma = 10
    l = 0.001
    k = sigma**2 * np.exp( -np.linalg.norm(x1 - x2)**2 / (2*l**2) )
    return k
 
def mean(x):
    return 0

# TODO(Tianju): Do visualization
def visualize(data, N):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(0, 1, N)
    Y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(X, Y)
    Z = data[0]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # plt.axis('off')
    plt.show()

if __name__ == '__main__':
    args = arguments.args
    # graph = GraphMSHR(args)
    graph = GraphManual(args)
    print("generated")
    data = generate_samples(graph=graph, num_samps=30000)


    np.save(args.root_path + '/' + args.numpy_path + '/' + graph.name +
            '-Multi-30000-' + str(graph.num_vertices) + '.npy', data)

    # data = np.identity(graph.num_vertices)
    # np.save(args.root_path + '/' + args.numpy_path + '/' + graph.name +
    #         '-ID-' + str(graph.num_vertices) + '.npy', data)
    # np.save(args.root_path + '/' + args.numpy_path + '/' + graph.name +
    #         '-GP-3000-' + str(graph.num_vertices) + '.npy', data)