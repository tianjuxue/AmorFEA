# Clean note: unnessary lines of codes

import numpy as np
import math
import matplotlib.pyplot as plt
from .demo import get_topology, plot_mesh_general
from ..opt.optimizer_robot import heart_shape
from .. import arguments


def plot_mesh(args):
    path = args.root_path + '/' + args.solutions_path + '/robot/deploy/u0000000.vtu'
    plot_mesh_general(path)


def plot_sol(args, case_number):
    data = np.load(args.root_path + '/' + args.numpy_path
                   + '/robot/deploy/case' + str(case_number) + '.npz')
    target_point = data['target_point']
    path = args.root_path + '/' + args.solutions_path + \
        '/robot/deploy/u' + str(case_number) + '000000.vtu'
    x, u, tri = get_topology(path)
    xu = x + u
    fig = plt.figure(figsize=(8, 8))
    colors = np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
    # plt.tripcolor(x[:,0], x[:,1], tri, 0.75*np.ones(len(xu)), vmin=0, vmax=1, alpha=0.5)
    tpc = plt.tripcolor(x[:, 0], x[:, 1], tri, colors,
                        shading='flat', alpha=0.5, vmin=0, vmax=2)
    tpc = plt.tripcolor(xu[:, 0], xu[:, 1], tri, colors,
                        shading='flat', alpha=1, vmin=0, vmax=2)
    plt.scatter(target_point[0] + 0.25, target_point[1] +
                10, marker='*', s=200, color='red')

    t = np.linspace(0, 2 * np.pi, 100)
    x1 = 2 * np.cos(t) + 0.25
    x2 = 2 * np.sin(t) + 10
    plt.plot(x1, x2, linestyle='--', color='blue')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    # cb = plt.colorbar(tpc, aspect=10)
    # cb.ax.tick_params(labelsize=20)
    fig.savefig(args.root_path + '/images/robot/sol' +
                str(case_number) + '.png', bbox_inches='tight')


def plot_demo_sol(args):
    path = args.root_path + '/' + args.solutions_path + '/robot/deploy/gt000000.vtu'
    x, u, tri = get_topology(path)
    xu = x + u
    fig = plt.figure(figsize=(8, 8))
    colors = np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
    # plt.tripcolor(x[:,0], x[:,1], tri, 0.75*np.ones(len(xu)), vmin=0, vmax=1, alpha=0.5)
    tpc = plt.tripcolor(x[:, 0], x[:, 1], tri, colors,
                        shading='flat', alpha=0.5, vmin=0, vmax=2)
    tpc = plt.tripcolor(xu[:, 0], xu[:, 1], tri, colors,
                        shading='flat', alpha=1, vmin=0, vmax=2)
    plt.scatter(3.25 + 0.25, 0.25 + 10, marker='*', s=200, color='red')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    cb = plt.colorbar(tpc, aspect=20,  shrink=0.5)
    cb.ax.tick_params(labelsize=20)
    fig.savefig(args.root_path + '/images/robot/demo.png', bbox_inches='tight')


def plot_robot_and_trajectory(args):
    x_s, y_s = heart_shape()
    x_s, y_s = x_s + 0.25, y_s + 10
    index = 61
    x_r = []
    y_r = []
    for i in range(len(x_s)):
        path = args.root_path + '/' + args.solutions_path + \
            '/robot/time_series_gt/u' + str(i) + '000000.vtu'
        x, u, tri = get_topology(path)
        xu = x + u
        x_s, y_s = heart_shape()
        x_s, y_s = x_s + 0.25, y_s + 10
        fig = plt.figure(figsize=(8, 8))
        colors = np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
        tpc = plt.tripcolor(xu[:, 0], xu[:, 1], tri,
                            colors, shading='flat', vmin=0, vmax=2)
        # plt.tripcolor(xu[:,0], xu[:,1], tri, 0.75*np.ones(len(xu)), vmin=0, vmax=1)
        x_r.append(xu[index][0])
        y_r.append(xu[index][1])
        plt.plot(x_s, y_s, color='blue', label='prescribed')
        plt.plot(x_r, y_r, color='red', label='optimized')
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.xlim(-2, 3.5)
        plt.ylim(-0.5, 11)
        # plt.legend(loc='right')
        fig.savefig(args.root_path + '/images/robot/series/' + f'{i:04}' + '.png', bbox_inches='tight')


def plot_walltime(case_number, trun_ad, trun_mix):
    data = np.load(args.root_path + '/' + args.numpy_path
                   + '/robot/deploy/case' + str(case_number) + '.npz')

    wall_time_ad = data['wall_time_ad']
    objective_ad = data['objective_ad']
    nn_number = data['nn_number'] + 1
    wall_time_mix = data['wall_time_mix']
    objective_mix = data['objective_mix']

    wall_time_ad = wall_time_ad - wall_time_ad[0]
    wall_time_mix = wall_time_mix - wall_time_mix[0]

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(wall_time_ad[:trun_ad], objective_ad[
            :trun_ad], linestyle='--', marker='o', color='blue', label='Adjoint Method')
    ax.plot(wall_time_mix[nn_number - 1:trun_mix], objective_mix[
            nn_number - 1:trun_mix], linestyle='--', marker='o', color='blue')
    ax.plot(wall_time_mix[:nn_number], objective_mix[
            :nn_number], linestyle='--', marker='o', color='red', label='AmorFEA')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center right', prop={'size': 12})
    ax.tick_params(labelsize=14)
    fig.savefig(args.root_path + '/images/robot/walltime' +
                str(case_number) + '.png', bbox_inches='tight')


def plot_walltime_batch():
    truncation_ad = [-2, -11, -4, -1]
    truncation_mix = [-4, -1, -1, -11]
    for i in range(4):
        plot_walltime(i, truncation_ad[i], truncation_mix[i])


def plot_sol_batch():
    for i in range(4):
        plot_sol(args, i)


def plot_step_batch():
    for case_number in range(0, 4):
        data = np.load(args.root_path + '/' + args.numpy_path
                       + '/robot/deploy/case_step' + str(case_number) + '.npz')
        wall_time_ad = data['wall_time_ad']
        objective_ad = data['objective_ad']
        wall_time_nn = data['wall_time_nn']
        objective_nn = data['objective_nn']

        wall_time_ad = wall_time_ad - wall_time_ad[0]
        wall_time_nn = wall_time_nn - wall_time_nn[0]
        print(wall_time_ad[-1])
        print(wall_time_nn[-1])

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.arange(len(objective_ad)), objective_ad, linestyle='-',
                marker='.', color='blue', label='Adjoint Method')
        ax.plot(np.arange(len(objective_nn)), objective_nn,
                linestyle='-', color='red', marker='.', label='AmorFEA')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper right', prop={'size': 12})
        ax.tick_params(labelsize=14)
        fig.savefig(args.root_path + '/images/robot/step' +
                    str(case_number) + '.png', bbox_inches='tight')


if __name__ == '__main__':
    args = arguments.args
    # plot_mesh(args)
    # plot_sol_batch()
    # plot_walltime_batch()
    # plot_demo_sol(args)
    # plot_robot_and_trajectory(args)
    plot_step_batch()
    plt.show()
