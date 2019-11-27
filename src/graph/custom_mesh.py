import numpy as np
import fenics as fa
import mshr
from .. import arguments

def unit_disk():
    mshr.generate_mesh(mshr.Circle(fa.Point(0, 0), 1), 20)
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
    mesh = mshr.generate_mesh(outline - circle_1 - circle_2 - circle_3, resolution)
    return mesh

if __name__ == '__main__':
    args = arguments.args

    mesh = irregular_channel()
    print(mesh.num_vertices())

    file = fa.File(args.root_path + '/' + args.solutions_path + '/mesh.pvd')
    mesh.rename('u', 'u')
    file << mesh