import numpy as np
import math
import matplotlib.pyplot as plt
from .demo import get_topology, plot_mesh_general
from ..opt.optimizer_robot import heart_shape
from .. import arguments


def plot_mesh(args):
    path = args.root_path + '/' + args.solutions_path + '/linear/u000000.vtu'  
    plot_mesh_general(path)
 

def plot_sol(args, name):
    path = args.root_path + '/' + args.solutions_path + '/linear/' + name + '000000.vtu' 
    x, u, tri = get_topology(path)
    colors = u
    fig = plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal')
    plt.axis('off')
    tpc = plt.tripcolor(x[:,0], x[:,1], tri, colors, shading='flat', vmin=None, vmax=None)
    cb = plt.colorbar(tpc, aspect=20,  shrink=0.5)
    cb.ax.tick_params(labelsize=20)
    fig.savefig(args.root_path + '/images/linear/' + name + '.png', bbox_inches='tight')


def plot_L(args):
    path_a = args.root_path + '/' + args.numpy_path + '/linear/L_inf_a.npy' 
    path_s = args.root_path + '/' + args.numpy_path + '/linear/L_inf_s.npy' 
    L_inf_a = np.load(path_a)[:-1]
    L_inf_s = np.load(path_s)[:-1]  
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L_inf_a), 1)
    ax.plot(epoch, L_inf_a, linestyle='--', marker='o', color='red', label='AmorFEA')
    ax.plot(epoch, L_inf_s, linestyle='--', marker='o', color='blue', label='Supervised Training')
    ax.set_yscale('log')
    ax.legend(loc='upper right', prop={'size': 12})
    ax.tick_params(labelsize=14)
    fig.savefig(args.root_path + '/images/linear/L.png', bbox_inches='tight')
    # plt.yticks(np.arange(min(L_inf_a), max(L_inf_a)+1, 1.0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


if __name__ == '__main__':
    args = arguments.args
    plot_mesh(args)
    plot_sol(args, 'f')
    plot_sol(args, 'u')
    plot_L(args)
    plt.show()