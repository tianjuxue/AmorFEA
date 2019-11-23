"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa


class Poisson(object):
    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()

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

        self.source = fa.Expression(("sin(2*pi/L*x[0])"), L=self.args.n_cells*self.args.L0, degree=3)

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        energy = 0.5*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
        return energy

    def energy(self, u):
        return fa.assemble(self._energy_density(u) * fa.dx)

    def solve_problem_variational_form(self, boundary_fn=None):
        u = fa.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        E = self._energy_density(u)*fa.dx
        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du) 

        bcs = []
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            bcs = bcs + [boundary_bc]

        fa.solve(dE == 0, u, bcs, J=jacE)

        return u
      
    def solve_problem_weak_form(self, boundary_fn=None):
        u = fa.Function(self.V)      
        du = fa.TrialFunction(self.V)
        v  = fa.TestFunction(self.V)

        C1 = 1
        C2 = 1.1
        u0 = 0
        Te = fa.Expression(("sin(2*pi/L*x[0]) + 1"), L=self.args.n_cells*self.args.L0, degree=3)
        # Te = 1
        # F  = fa.inner(fa.grad(u), fa.grad(v))*fa.dx + (C1 - C2*fa.exp((u - u0)/Te))*v*fa.dx
        # F  = fa.inner(fa.grad(u), fa.grad(v))*fa.dx - self.source*v*fa.dx
        F  = fa.inner(Te*fa.grad(u), fa.grad(v))*fa.dx - v*fa.dx
        J  = fa.derivative(F, u, du)  

        bcs = []
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            bcs = bcs + [boundary_bc]

 
        problem = fa.NonlinearVariationalProblem(F, u, bcs, J)
        solver  = fa.NonlinearVariationalSolver(problem)
        
        # prm = solver.parameters
        # prm['newton_solver']['relaxation_parameter'] = self.args.relaxation_parameter
        # prm['newton_solver']['linear_solver'] =   self.args.linear_solver
        # prm['newton_solver']['maximum_iterations'] = self.args.max_newton_iter
        # prm['newton_solver']['relative_tolerance'] = self.args.rel_tol
        # prm['newton_solver']['absolute_tolerance'] =  self.args.abs_tol

        solver.solve()

        return u