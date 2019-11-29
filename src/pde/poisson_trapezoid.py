"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .poisson import Poisson
from ..graph.custom_mesh import irregular_channel
from .. import arguments


class PoissonTrapezoid(Poisson):
    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()

    def _build_mesh(self):
        self.mesh = irregular_channel()

    def _build_function_space(self):
        L0 = self.args.L0
        n_cells = self.args.n_cells

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 0)

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0] - 2, 0)

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1] + 0.5*x[0], 0)

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1] - 0.5*x[0] - 1, 0)

        class Inner(fa.SubDomain):
            def inside(self, x, on_boundary):
                is_circle_boundary = x[0] > 0 and x[0] < 2 and x[1] + 0.5*x[0] > 0 and x[1] - 0.5*x[0] - 1 < 0
                return on_boundary and is_circle_boundary

        self.V = fa.FunctionSpace(self.mesh, 'P', 1)

        self.sub_domains = fa.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.sub_domains.set_all(0)

        self.boundaries_id_dic = {'left': 1, 'right': 2, 'bottom': 3, 'top': 4, 'inner': 5}
        self.left = Left()
        self.left.mark(self.sub_domains, 1)
        self.right = Right()
        self.right.mark(self.sub_domains, 2)
        self.bottom = Bottom()
        self.bottom.mark(self.sub_domains, 3)
        self.top = Top()
        self.top.mark(self.sub_domains, 4)
        self.inner = Inner()
        self.inner.mark(self.sub_domains, 5)

        self.normal = fa.FacetNormal(self.mesh)
        self.ds = fa.Measure("ds")(subdomain_data=self.sub_domains)

        self.source = fa.Constant(1)
        self.medium = fa.Expression("sin(pi*x[0]) + 2", degree=3)

        # Change your boundary conditions here
        boundary_bc_left = fa.DirichletBC(self.V, fa.Constant(2.), self.left)
        boundary_bc_right = fa.DirichletBC(self.V, fa.Constant(1.), self.right)
        self.bcs = [boundary_bc_left, boundary_bc_right]

    def solve_problem_weak_form(self):
        u = fa.Function(self.V)      
        du = fa.TrialFunction(self.V)
        v  = fa.TestFunction(self.V)
        F  = fa.inner(self.medium*fa.grad(u), fa.grad(v))*fa.dx - self.source*v*fa.dx
        J  = fa.derivative(F, u, du)  

        # Change your boundary conditions here
        boundary_bc_left = fa.DirichletBC(self.V, fa.Constant(2.), self.left)
        boundary_bc_right = fa.DirichletBC(self.V, fa.Constant(1.), self.right)
        bcs = [boundary_bc_left, boundary_bc_right]

        # The problem in this case is indeed linear, but using a nonlinear solver doesn't hurt
        problem = fa.NonlinearVariationalProblem(F, u, self.bcs, J)
        solver  = fa.NonlinearVariationalSolver(problem)
        solver.solve()
        return u

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        energy = 0.5*self.medium*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
        return energy

    def energy(self, u):
        return fa.assemble(self._energy_density(u) * fa.dx)

    def solve_problem_variational_form(self):
        u = fa.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        E = self._energy_density(u)*fa.dx
        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du) 
        fa.solve(dE == 0, u, self.bcs, J=jacE)

        print(self.energy(u))

        return u

if __name__ == '__main__':
    args = arguments.args
    pde = PoissonTrapezoid(args)
    u = pde.solve_problem_variational_form()
    file = fa.File(args.root_path + '/' + args.solutions_path + '/u.pvd')
    u.rename('u', 'u')
    file << u