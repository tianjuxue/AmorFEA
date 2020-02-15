import numpy as np
import math
import matplotlib.pyplot as plt
from .demo import get_topology, plot_mesh_general
from .. import arguments


def plot_mesh(args):
    path = args.root_path + '/' + args.solutions_path + '/dolfin/opt_fem_f000000.vtu'
    plot_mesh_general(path)


def plot_sol(args, name, vmin=None, vmax=None):
    path = args.root_path + '/' + args.solutions_path + \
        '/dolfin/' + name + '000000.vtu'
    x, u, tri = get_topology(path)
    colors = u
    fig = plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal')
    plt.axis('off')
    # shading='gouraud'
    tpc = plt.tripcolor(x[:, 0], x[:, 1], tri, colors,
                        shading='flat', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(tpc, aspect=20,  shrink=0.5)
    cb.ax.tick_params(labelsize=20)
    fig.savefig(args.root_path + '/images/dolfin/' +
                name + '.png', bbox_inches='tight')


if __name__ == '__main__':
    args = arguments.args
    plot_mesh(args)
    # plot_sol(args, 'opt_nn_f', 0, 6)
    # plot_sol(args, 'opt_ad_f', 0, 6)
    # plot_sol(args, 'opt_fem_f', 0, 6)
    plot_sol(args, 'f')
    plot_sol(args, 'u')
    plt.show()
