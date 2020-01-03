import numpy as np
import math
from .. import arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from ..graph.visualization import *
from ..pde.poisson_square import PoissonSquare
from ..pde.poisson_trapezoid import PoissonTrapezoid
from ..pde.soft_robot import SoftRobot


def generate_gaussian_samples(args, pde, num_samps, case_flag):

    # TODO(Tianju): Make this general
    # def RBF_kernel(x1, x2):
    #     sigma = 1
    #     l = 0.5
    #     k = sigma**2 * np.exp( -np.linalg.norm(x1 - x2)**2 / (2*l**2) )
    #     return k

    def RBF_kernel(x1, x2):
        sigma = 1
        l1 = 0.001
        l2 = 5
        d1 = (x1[0] - x2[0])**2
        d2 = (x1[1] - x2[1])**2
        k = sigma**2 * np.exp( -d1/(2*l1**2) - d2/(2*l2**2))
        return k

    def mean(x):
        return 0.

    M = pde.num_dofs
    coo = pde.coo_dof
    kernel_matrix = np.zeros((M, M))
    mean_vector = np.zeros(M)

    for i in range(M):
        mean_vector[i] = mean(coo[i])
        for j in range(M):
            kernel_matrix[i][j] = RBF_kernel(coo[i], coo[j])

    samples = np.random.multivariate_normal(mean_vector, kernel_matrix, num_samps)
    save_generated_data(args, case_flag, pde, 'Gaussian', samples)
    return samples

def generate_uniform_samples(args, pde, num_samps, case_flag):
    M = pde.num_dofs
    samples = np.random.uniform(-0.1, 0.1, (num_samps, M))
    save_generated_data(args, case_flag, pde, 'Uniform', samples)
    return samples

def generate_multinomial_samples(args, pde, num_samps):
    M = pde.num_dofs
    samples = np.random.multinomial(1, [1/M]*M, size=num_samps)
    save_generated_data(args, case_flag, pde, 'Multi', samples)
    return samples

def save_generated_data(args, case_flag, pde, distribution, samples):
    M = pde.num_dofs
    if case_flag == 0:
        save_path = '/linear/'
    elif case_flag == 1:
        save_path = '/nonlinear/'
    else:
        save_path = '/robot/'
    np.save(args.root_path + '/' + args.numpy_path + save_path + pde.name +
            '-' + distribution + '-' + str(samples.shape[0]) + '-' + str(M) + '.npy', samples)

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
    case_flag = 2
    if case_flag == 0:
        poisson_square = PoissonSquare(args)
        samples = generate_uniform_samples(args, poisson_square, 30000, case_flag)
    elif case_flag == 1:
        poisson_trapezoid = PoissonTrapezoid(args)
        samples = generate_gaussian_samples(args, poisson_trapezoid, 3000, case_flag)
    else:
        soft_robot = SoftRobot(args)
        # samples = generate_gaussian_samples(args, soft_robot, 3000, case_flag)
        samples = generate_uniform_samples(args, soft_robot, 30000, case_flag)

    scalar_field_paraview(args, samples[2], soft_robot, 'debug_source')