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
        super(PoissonTrapezoid, self).__init__(args)
        self.name = 'trapezoid'

    def _build_mesh(self):
        self.mesh = fa.Mesh(self.args.root_path + '/' + 
                            self.args.solutions_path + '/saved_mesh/mesh_trapezoid.xml')

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

        self.source = fa.Constant(0)
        self.medium = fa.Expression("sin(pi*x[0]) + 2", degree=3)

        # Change your boundary conditions here
        boundary_bc_left = fa.DirichletBC(self.V, fa.Constant(0.), self.left)
        boundary_bc_right = fa.DirichletBC(self.V, fa.Constant(0.), self.right)
        self.bcs = [boundary_bc_left, boundary_bc_right]

    def set_control_variable(self, dof_data):
        self.source = fa.Function(self.V)
        self.source.vector()[:] = dof_data

    # Constitutive relationships
    def _energy_density(self, u):
        # variational energy density of u
        # energy = 0.5*self.medium*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
        energy = 0.5*fa.dot(fa.grad(u), fa.grad(u)) + 1*0.5*u**2 + 0.25*u**4 - u*self.source
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

    def _set_detailed_boundary_flags(self):
        # To check
        x1 = self.coo_dof[:, 0]
        x2 = self.coo_dof[:, 1]
        boundary_flags_list = [np.zeros(self.num_dofs) for i in range(3)] 
        for i in range(self.num_dofs):
            if self.boundary_flags[i] == 1:
                if x1[i] < 1e-10:
                    boundary_flags_list[0][i] = 1
                if x1[i] > 2 - 1e-10:
                    boundary_flags_list[1][i] = 1
                if np.abs(x2[i] - 0.5*x1[i] - 1) < 1e-10 or \
                   np.abs(x2[i] + 0.5*x1[i]) < 1e-10 or \
                   x1[i] > 1e-10 and x1[i] < 2 - 1e-10:
                   boundary_flags_list[2][i] = 1
        self.boundary_flags_list =  boundary_flags_list

    def compute_operators(self):
        u = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)
        form_a = fa.inner(fa.grad(u), fa.grad(v))*fa.dx 
        form_b = u*v*fa.dx 
        A = fa.assemble(form_a)
        B = fa.assemble(form_b)
        A_np = np.array(A.array())
        B_np = np.array(B.array())
        return A_np, B_np

    def debug(self):
        v = fa.Function(self.V)
        v.vector()[4] = 1
        u = fa.Function(self.V)
        u.vector()[4] = 1
        print(value)


if __name__ == '__main__':
    args = arguments.args
    pde = PoissonTrapezoid(args)
    adjacency_matrix = pde.get_adjacency_matrix()

    print(adjacency_matrix.sum())

    # print(pde.energy(u))
    # file = fa.File(args.root_path + '/' + args.solutions_path + '/u.pvd')
    # u.rename('u', 'u')
    # file << u