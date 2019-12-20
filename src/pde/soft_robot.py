"""Poisson problem PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .poisson import Poisson
from .. import arguments


class SoftRobot(Poisson):
    def __init__(self, args):
        super(SoftRobot, self).__init__(args)
        self.name = 'robot'

    def _build_mesh(self):
        args = self.args
        mesh = fa.Mesh(args.root_path + '/' + args.solutions_path + '/saved_mesh/mesh_robot.xml')
        # mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(1, 10), 2, 20)
        self.mesh = mesh

        # file = fa.File(args.root_path + '/' + args.solutions_path + '/mesh.pvd')
        # mesh.rename('mesh', 'mesh')
        # file << mesh

    def _build_function_space(self):
        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], 10) or
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
                return on_boundary and fa.near(x[1], 10)

        self.exteriors_dic = {'left': Left(), 'right': Right(), 'bottom': Bottom(), 'top': Top()}
        self.exterior = Exterior()

        self.V = fa.VectorFunctionSpace(self.mesh, 'P', 1)
        self.W = fa.FunctionSpace(self.mesh, 'DG', 0)

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

        # self.source = fa.Expression(("1 + 0.1*x[1]"),  degree=1)
        # self.source = fa.Constant(1.)

        self.bcs = [ self.bottom]

        boundaries = [self.bottom, self.left, self.right]
        boundary_fn = [fa.Constant((0., 0.)),
                       fa.Expression(("0", ".3808*x[1]"), degree=1),
                       fa.Expression(("0", ".3808*x[1]"), degree=1)]

        self.bcs = []
        for i in range(len(boundaries)):
            boundary_bc = fa.DirichletBC(self.V, boundary_fn[i], boundaries[i])
            self.bcs = self.bcs + [boundary_bc]

    def _set_detailed_boundary_flags(self):
        x1 = self.coo_dof[:, 0]
        x2 = self.coo_dof[:, 1]
        # [bottom, left_x, left_y, right_x, right_y]
        boundary_flags_list = [np.zeros(self.num_dofs) for i in range(5)] 
        counter_left = 0
        counter_right = 0
        for i in range(self.num_dofs):
            if x2[i] < 1e-10:
                boundary_flags_list[0][i] = 1
            else:
                if x1[i] < 1e-10:
                    if counter_left%2 == 0:
                        boundary_flags_list[1][i] = 1
                    else:
                        boundary_flags_list[2][i] = 1
                    counter_left += 1

                if x1[i] > 1 - 1e-10:
                    if counter_right%2 == 0:
                        boundary_flags_list[3][i] = 1
                    else:
                        boundary_flags_list[4][i] = 1
                    counter_right += 1
        self.boundary_flags_list =  boundary_flags_list

    # Deformation gradient
    def DeformationGradient(self, u):
        I = fa.Identity(u.geometric_dimension())
        return I + fa.grad(u)

    # Right Cauchy-Green tensor
    def RightCauchyGreen(self, F):
        return F.T * F

    # Neo-Hookean Energy
    def _energy_density(self, u):
        young_mod = 100
        poisson_ratio = 0.3
        shear_mod = young_mod / (2 * (1 + poisson_ratio))
        bulk_mod = young_mod / (3 * (1 - 2*poisson_ratio))
        d = u.geometric_dimension()
        F = self.DeformationGradient(u)
        F = fa.variable(F)
        J = fa.det(F)
        I1 = fa.tr(self.RightCauchyGreen(F))

        # Plane strain assumption
        Jinv = J**(-2 / 3)
        energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
                  (bulk_mod / 2) * (J - 1)**2)  
        return energy

    def set_control_variable(self, dof_data):
        self.source = fa.Function(self.W)
        self.source.vector()[:] = dof_data

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

    def compute_operators(self):
        v = fa.Function(self.V)
        w = fa.Function(self.W)
        F00 = []
        F01 = []
        F10 = []
        F11 = []
        for i in range(self.num_dofs):
            v.vector()[:] = 0
            v.vector()[i] = 1
            F = self.DeformationGradient(v)
            f00 = fa.project(F[0, 0], self.W)
            f01 = fa.project(F[0, 1], self.W)
            f10 = fa.project(F[1, 0], self.W)
            f11 = fa.project(F[1, 1], self.W)
            F00.append(np.array(f00.vector()))
            F01.append(np.array(f01.vector()))
            F10.append(np.array(f10.vector()))
            F11.append(np.array(f11.vector()))
        # Do not forget to add 1 later
        F00 = np.transpose(np.array(F00)) - 1
        F01 = np.transpose(np.array(F01))
        F10 = np.transpose(np.array(F10))
        F11 = np.transpose(np.array(F11)) - 1

        return F00, F01, F10, F11

    def compute_areas(self):
        w = fa.Function(self.W)
        area = np.zeros(self.W.dim())
        for i in range(self.W.dim()):
            w.vector()[:] = 0
            w.vector()[i] = 1
            area[i] = fa.assemble(w*fa.dx)
        return area

    def debug(self):
        v = fa.Function(self.V)
        v.vector()[0] = 1
        w = fa.Function(self.W)
        w.vector()[20] = 1
        print(np.array(test.vector()))


if __name__ == '__main__':
    args = arguments.args
    pde = SoftRobot(args)

    u = pde.solve_problem_variational_form()
    print(pde.energy(u))
    file = fa.File(args.root_path + '/' + args.solutions_path + '/u.pvd')
    u.rename('u', 'u')
    file << u