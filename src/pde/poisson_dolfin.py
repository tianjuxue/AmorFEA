import math
import time
import numpy as np
import mshr
import fenics as fa
import dolfin_adjoint as da
from .poisson import Poisson
from ..graph.custom_mesh import irregular_channel
from .. import arguments
from ..graph.visualization import scalar_field_paraview, save_solution


class PoissonDolfin(Poisson):

    def __init__(self, args):
        super(PoissonDolfin, self).__init__(args)
        self.name = 'dolfin'

    def _build_mesh(self):
        # Change log: Ubuntu has da but OS X needs to use fa
        self.mesh = da.Mesh(self.args.root_path + '/' +
                            self.args.solutions_path + '/saved_mesh/dolfin_coarse.xml')

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

        class Interior(fa.SubDomain):

            def inside(self, x, on_boundary):
                return on_boundary and (
                    x[0] > 0.1 and
                    x[0] < 0.9 and
                    x[1] > 0.1 and
                    x[1] < 0.9)

        self.exteriors_dic = {
            'left': Left(), 'right': Right(), 'bottom': Bottom(), 'top': Top()}
        self.exterior = Exterior()
        self.interior = Interior()

        self.V = fa.FunctionSpace(self.mesh, 'P', 1)

        self.source = da.Expression(("100*sin(2*pi*x[0])"),  degree=3)
        # self.source = da.Expression("k*100*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*0.01*l) )",
        #                        k=1, l=1, x0=0.9, x1=0.1, degree=3)
        # self.source = da.Constant(10)
        self.source = da.interpolate(self.source, self.V)

        boundary_fn_ext = da.Constant(1.)
        boundary_fn_int = da.Constant(1.)
        boundary_bc_ext = da.DirichletBC(
            self.V, boundary_fn_ext, self.exterior)
        boundary_bc_int = da.DirichletBC(
            self.V, boundary_fn_int, self.interior)
        self.bcs = [boundary_bc_ext, boundary_bc_int]

    def set_control_variable(self, dof_data):
        self.source = da.Function(self.V)
        self.source.vector()[:] = dof_data

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        energy = 0.5 * fa.dot(fa.grad(u), fa.grad(u)) + 10 * \
            0.5 * u**2 + 10 * 0.25 * u**4 - u * self.source
        return energy

    def energy(self, u):
        return da.assemble(self._energy_density(u) * fa.dx)

    def solve_problem_variational_form(self):
        u = da.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        E = self._energy_density(u) * fa.dx
        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du)
        da.solve(dE == 0, u, self.bcs, J=jacE)
        return u

    def _set_detailed_boundary_flags(self):
        # To check
        x1 = self.coo_dof[:, 0]
        x2 = self.coo_dof[:, 1]
        boundary_flags_list = [np.zeros(self.num_dofs) for i in range(2)]
        for i in range(self.num_dofs):
            if self.boundary_flags[i] == 1:
                if x1[i] < 1e-10 or x1[i] > 1 - 1e-10 or x2[i] < 1e-10 or x2[i] > 1 - 1e-10:
                    boundary_flags_list[0][i] = 1
                else:
                    boundary_flags_list[1][i] = 1
        self.boundary_flags_list = boundary_flags_list

    def compute_operators(self):
        u = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        form_a = fa.inner(fa.grad(u), fa.grad(v)) * fa.dx
        form_b = u * v * fa.dx
        A = da.assemble(form_a)
        B = da.assemble(form_b)
        A_np = np.array(A.array())
        B_np = np.array(B.array())
        return A_np, B_np

    def debug(self):
        _, B_np = self.compute_operators()
        dof_data = np.random.rand(self.num_dofs)
        u = da.Function(self.V)
        u.vector()[:] = dof_data
        L1 = da.assemble(u * u * fa.dx)
        L2 = (np.matmul(B_np, dof_data) * dof_data).sum()
        print(L1)
        print(L2)


if __name__ == '__main__':
    args = arguments.args
    pde = PoissonDolfin(args)
    # adjacency_matrix = pde.get_adjacency_matrix()

    u = pde.solve_problem_variational_form()
    # save_solution(args, u, '/dolfin/u')
    # save_solution(args, pde.source, 'dolfin/f')
