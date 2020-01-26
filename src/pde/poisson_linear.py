"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .poisson import Poisson
from .. import arguments


class PoissonLinear(Poisson):
    def __init__(self, args):
        super(PoissonLinear, self).__init__(args)
        self.name = 'linear'

    def _build_mesh(self):
        args = self.args
        mesh = fa.Mesh(args.root_path + '/' + args.solutions_path + '/saved_mesh/mesh_square.xml')
        # mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(1, 1), 3, 3)
        self.mesh = mesh

    def _build_function_space(self):
        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], 1) or
                    fa.near(x[0], 1) or
                    fa.near(x[0], 0) or
                    fa.near(x[1], 0))

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 0)

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 1)

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], 0)

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], 1)

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

        self.source = fa.Expression(("sin(2*pi*x[0])"),  degree=3)
        self.source = fa.Constant(1.)

        self.bcs = []
        boundary_fn = fa.Constant(0.)
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            self.bcs = self.bcs + [boundary_bc]

    def _set_detailed_boundary_flags(self):
        self.boundary_flags_list = [self.boundary_flags]

    def set_control_variable(self, dof_data):
        self.source = fa.Function(self.V)
        self.source.vector()[:] = dof_data

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        # energy = 0.5*self.medium*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
        energy = 0.5*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
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
        return u

    def solve_problem_weak_form(self):
        u = fa.Function(self.V)      
        du = fa.TrialFunction(self.V)
        v  = fa.TestFunction(self.V)
        F  = fa.inner(fa.grad(u), fa.grad(v))*fa.dx - self.source*v*fa.dx
        J  = fa.derivative(F, u, du)  

        # The problem in this case is indeed linear, but using a nonlinear solver doesn't hurt
        problem = fa.NonlinearVariationalProblem(F, u, self.bcs, J)
        solver  = fa.NonlinearVariationalSolver(problem)
        solver.solve()
        return u
        
    def solve_problem_matrix_approach(self):
        u = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        a = fa.inner(fa.grad(u), fa.grad(v))*fa.dx 
        L = self.source*v*fa.dx
 
        # equivalent to solve(a == L, U, b)
        A = fa.assemble(a)
        b = fa.assemble(L)
        [bc.apply(A, b) for bc in self.bcs]
        u = fa.Function(self.V)
        U = u.vector()
        fa.solve(A, U, b)
        return u

    def compute_operators(self):
        u = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        form_a = fa.inner(fa.grad(u), fa.grad(v))*fa.dx 
        form_b = u*v*fa.dx 

        A = fa.assemble(form_a)
        B = fa.assemble(form_b)
        A_np = np.array(A.array())
        B_np = np.array(B.array())

        [bc.apply(A) for bc in self.bcs]
        A_np_modified = np.array(A.array())
 
        return A_np, B_np, A_np_modified

    def debug(self):
        v = fa.Function(self.V)
        v.vector()[4] = 1
        u = fa.Function(self.V)
        u.vector()[4] = 1
        # v.vector()[:] = 1
        value = np.array(fa.assemble(fa.inner(fa.grad(u), fa.grad(v))*fa.dx))
        print(value)


if __name__ == '__main__':
    args = arguments.args
    pde = PoissonLinear(args)
    # u = pde.solve_problem_weak_form_explicit()
    # file = fa.File(args.root_path + '/' + args.solutions_path + '/u.pvd')
    # u.rename('u', 'u')
    # file << u