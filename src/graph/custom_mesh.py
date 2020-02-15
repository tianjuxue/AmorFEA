"""Warning: mshr generates mesh randomly - with same input parameter the mesh might be different
Better save the mesh and then always load (the same) mesh later on to avoid bugs caused by inconsistency
"""

import numpy as np
import fenics as fa
import mshr
from .. import arguments


def unit_disk():
    mesh = mshr.generate_mesh(mshr.Circle(fa.Point(0, 0), 1), 20)
    return mesh


def irregular_channel():
    radius = 0.35
    resolution = 25
    point_A = fa.Point(0, 0)
    point_B = fa.Point(2, -1)
    point_C = fa.Point(2, 2)
    point_D = fa.Point(0, 1)
    point_E = fa.Point(0.5, 0.5)
    point_F = fa.Point(1.5, 1)
    point_G = fa.Point(1.5, 0)
    outline = mshr.Polygon([point_A, point_B, point_C, point_D])
    circle_1 = mshr.Circle(point_E, radius)
    circle_2 = mshr.Circle(point_F, radius)
    circle_3 = mshr.Circle(point_G, radius)
    mesh = mshr.generate_mesh(
        outline - circle_1 - circle_2 - circle_3, resolution)
    return mesh


def unit_square():
    mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(1, 1), 30, 30)
    return mesh


def slender_rod():
    mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(1, 10), 2, 20)
    return mesh


if __name__ == '__main__':
    args = arguments.args
    case_flag = 0
    if case_flag == 0:
        # mesh = unit_square()
        # file = fa.File(args.root_path + '/' + args.solutions_path + '/saved_mesh/mesh_square.xml')
        mesh = unit_disk()
        file = fa.File(args.root_path + '/' +
                       args.solutions_path + '/saved_mesh/mesh_disk.xml')
    elif case_flag == 1:
        mesh = irregular_channel()
        file = fa.File(args.root_path + '/' +
                       args.solutions_path + '/saved_mesh/mesh_trapezoid.xml')
    else:
        mesh = slender_rod()
        file = fa.File(args.root_path + '/' +
                       args.solutions_path + '/saved_mesh/mesh_robot.xml')

    file << mesh

    # loaded_mesh = fa.Mesh(args.root_path + '/' + args.solutions_path + '/saved_mesh/mesh_robot.xml')
    # print(loaded_mesh.num_vertices())

    # file = fa.File(args.root_path + '/' + args.solutions_path + '/mesh.pvd')
    # mesh.rename('mesh', 'mesh')
    # file << mesh
