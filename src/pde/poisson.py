"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .pde import PDE


class Poisson(PDE):
    def _build_mesh(self):
        args = self.args
        mesh = fa.RectangleMesh(fa.Point(0, 0), 
                                fa.Point(args.n_cells*args.L0, args.n_cells*args.L0), 
                                args.n_cells,
                                args.n_cells) 
        self.mesh = mesh

    def _build_function_space(self):
        L0 = self.args.L0
        n_cells = self.args.n_cells

        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], L0 * n_cells) or
                    fa.near(x[0], L0 * n_cells) or
                    fa.near(x[0], 0) or
                    fa.near(x[1], 0))

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 0)

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], L0 * n_cells)

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], 0)

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], L0 * n_cells)

        self.exteriors_dic = {'left': Left(), 'right': Right(), 'bottom': Bottom(), 'top': Top()}
        self.exterior = Exterior()
        
        self.V = fa.FunctionSpace(self.mesh, 'P', 1)

        self.sub_domains = fa.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.sub_domains.set_all(0)

        self.boundaries_id_dic = {'left': 1, 'right': 2, 'bottom': 3, 'top': 4}
        self.left = Left()
        self.left.mark(self.sub_domains, 1)
        self.right = Right()
        self.right.mark(self.sub_domains, 2)
        self.bottom = Bottom()
        self.bottom.mark(self.sub_domains, 3)
        self.top = Top()
        self.top.mark(self.sub_domains, 4)

        self.normal = fa.FacetNormal(self.mesh)
        self.ds = fa.Measure("ds")(subdomain_data=self.sub_domains)

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        f = fa.Expression(("sin(2*pi/L*x[0])"), L=self.args.n_cells*self.args.L0, degree=3)
        energy = 0.5*fa.dot(fa.grad(u), fa.grad(u)) - u*f
        return energy
        


