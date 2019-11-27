"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa


class Poisson(object):
    def _build_mesh(self):
        args = self.args
        mesh = fa.RectangleMesh(fa.Point(0, 0), 
                                fa.Point(args.n_cells*args.L0, args.n_cells*args.L0), 
                                args.n_cells,
                                args.n_cells) 
        self.mesh = mesh

    # To be implemented
    def _build_function_space(self):
        pass


    def solve_problem_weak_form(self, boundary_fn=None):
        u = fa.Function(self.V)      
        du = fa.TrialFunction(self.V)
        v  = fa.TestFunction(self.V)

        e = 1.6*1e-19
        eps_0 = 8.85*1e-12
        n_e = 1e18
        C = e*n_e/eps_0
        k_B = 8.6*1e-5

        u0 = 1e3
        Te = 1e5
 
        Te = fa.Expression(("sin(2*pi/L*x[0]) + 1"), L=self.args.n_cells*self.args.L0, degree=3)
        # Te = 1
        F  = fa.inner(fa.grad(u), fa.grad(v))*fa.dx - C*fa.exp(-(u - u0)/(k_B*Te))*v*fa.dx
        # F  = fa.inner(fa.grad(u), fa.grad(v))*fa.dx - self.source*v*fa.dx
        # F  = fa.inner(Te*fa.grad(u), fa.grad(v))*fa.dx - v*fa.dx
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