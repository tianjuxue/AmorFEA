#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:38:13 2019

@author: alexanderniewiarowski
"""

from dolfin import *

try:
    from dolfin_adjoint import *
    ADJOINT = True
except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False
    import contextlib
    def stop_annotating():
        return contextlib.nullcontext()

from simple_membranes.materials import *
from simple_membranes.gas import *
#from fenicsmembranes.liquid import *
#from fenicsmembranes.solvers import *
from .io import InputOutputHandling
from simple_membranes.boundary_conditions import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize":True, "quadrature_degree":5}
parameters["form_compiler"]["quadrature_degree"] = 5
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

class ParametricMembrane(object):
    """
    Parametric membrane class
    """
    def __init__(self, kwargs):

        self.material = None
        self.thickness = None
        self.solver = None
        self.data = {}
        self.kwargs = kwargs

        geo = kwargs.get("geometry")
        self.gamma = geo.gamma
        self.Gsub1 = geo.Gsub1
        self.Gsub2 = geo.Gsub2

        self.nsd = self.gamma.ufl_function_space().ufl_element().value_size()
        self._get_mesh()

        # Define function spaces
        self.Ve = VectorElement("CG", self.mesh.ufl_cell(), degree=2, dim=self.nsd)
        self.V = FunctionSpace(self.mesh, self.Ve)
        self.Vs = FunctionSpace(self.mesh, 'CG', 2)  # should this be 2?

        # Construct spaces for plotting discontinuous fields
        self.W = VectorFunctionSpace(self.mesh, 'DG', 0, dim=self.nsd)
        self.Z = FunctionSpace(self.mesh, 'DG', 1)

        # Define trial and test function
        self.du = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u = Function(self.V, name="u")

        self.initial_position = interpolate(self.gamma, self.V)
        self.p_ext = Function(self.Vs, name="External pressure")  # external pressure field
        self.l1 = Function(self.Vs, name="lambda 1")
        self.l2 = Function(self.Vs, name="lambda 2")
        self.l3 = Function(self.Vs, name="lambda 3")
        self.s11 = Function(self.Vs, name="stress1")
        self.s22 = Function(self.Vs, name="stress2")
        self.normals = Function(self.W, name="Surface unit normals")

        self.data = {'u': self.u,
                     'gamma': self.gamma,
                     'p_ext': self.p_ext,
                     'n': self.normals,
                     'l1': self.l1,
                     'l2': self.l2,
                     'l3': self.l3,
                     's11': self.s11,
                     's22': self.s22}

        # setup thickness
        if type(kwargs['thickness']) is float:
            self.thickness = Constant(kwargs['thickness'], name='thickness')
        else:
            if kwargs['thickness']['type'] == 'Constant':
                self.thickness = Constant(kwargs['thickness']['value'], name='thickness')
            elif kwargs['thickness']['type'] == 'Expression':
                self.thickness = Expression(kwargs['thickness']['value'], degree=1)
            elif kwargs['thickness']['type'] == 'Function_constant':
                self.thickness = Function(self.Vs, name="thickness")
                self.thickness.assign(interpolate(Expression(kwargs['thickness']['value'], degree=1), self.Vs))
            elif kwargs['thickness']['type'] == 'Function':
                self.thickness = Function(self.Vs, name="thickness")
                self.thickness.assign(kwargs['thickness']['value'])

        self.setup_kinematics()

        self.bc_type = kwargs['Boundary Conditions']
        if self.bc_type == 'Pinned':
            self.bc = pinBC(self)
        elif self.bc_type == 'Roller':
             self.bc = rollerBC(self)
        else:
            raise(f'Boundary condition {self.bc_type} not implemented!')

        # Initialize material, constitutive law, and internal potential energy
        material_name = kwargs["material"]
        material_class = get_material(material_name)
        self.material = material_class(self, kwargs)
        self.Pi = self.thickness*self.material.psi*self.J_A*dx(self.mesh)  #sqrt(dot(self.Gsub3, self.Gsub3))*dx(self.mesh)

        if self.nsd==3:
            self.PK2 = self.get_PK2()
            self.cauchy = self.F_n*self.PK2*self.F_n.T  #  Bonet eq(39)  = J^{-1} F S F^T Here J == 1

        # Create input/ouput instance and save initial states
        self.io = InputOutputHandling(self)
        self.output_file_path = kwargs["output_file_path"]
        self.io.setup()

        self.phases = []    
        # Initialize internal gas (if any)
        self.p_0 = Constant(kwargs['pressure'], name="initial pressure")
        gas_name = kwargs.get("Gas_law", "Isentropic Gas")
        if gas_name is not None:
            gas_class = get_gas_law(gas_name)
            self.gas = gas_class(self)
            self.phases.append(self.gas)
            
        # Initialize internal fluid (if any)
        self.free_surface = self.kwargs.get("free_surface", False)

        liquid_name = kwargs.get("Liquid", None)
        if liquid_name is not None:
            liquid_class = get_liquid_law(liquid_name)
            self.liquid = liquid_class(self)
            self.phases.append(self.liquid)
        
        with stop_annotating():
            self.io.write_fields()
    
        self.inflate(self.p_0)
        with stop_annotating():
            self.vol_0 = assemble((1/self.nsd)*dot(self.gamma, self.Gsub3)*dx(self.mesh))
            self.area_0 = assemble(sqrt(dot(self.Gsub3, self.Gsub3))*dx(self.mesh))
            print(f"Initial volume: {self.vol_0}")
            print(f"Initial area: {self.area_0}")


    def _get_mesh(self):
        res = self.kwargs.get("resolution")
        assert len(res)==self.nsd-1, "Mesh resolution does not match dimension" 
        
        mesh_type = self.kwargs.get("mesh_type", "Default")
       
        if self.nsd==2:
            self.mesh = UnitIntervalMesh(res[0])

        elif self.nsd==3:
            self.mesh = UnitSquareMesh(res[0], res[1], 'crossed')

    def setup_kinematics(self):
        from dolfin import cross, sqrt, dot
        if self.nsd==2:
            # Get the dual basis
            self.Gsup1 = self.Gsub1/dot(self.Gsub1, self.Gsub1)
            R = as_tensor([[0, -1],
                           [1, 0]])

            self.gsub1 = self.Gsub1 + self.u.dx(0)
            gradu = outer(self.u.dx(0), self.Gsup1)

            I = Identity(self.nsd)
            self.F = I + gradu
            self.C = dot(self.F.T, self.F)

            self.Gsub3 = dot(R, self.Gsub1)
            self.gsub3 = dot(R, self.gsub1)
            lmbda = sqrt(dot(self.gsub1, self.gsub1)/dot(self.Gsub1, self.Gsub1))
            self.lambda1, self.lambda2 = lmbda, lmbda
            self.lambda3 = 1/self.lambda1

        elif self.nsd==3:
            from dolfin import cross, sqrt, dot
            from fenicsmembranes.calculus_utils import contravariant_base_vector
            
            # Get the contravariant tangent basis
            self.Gsup1 = contravariant_base_vector(self.Gsub1, self.Gsub2)
            self.Gsup2 = contravariant_base_vector(self.Gsub2, self.Gsub1)

            # Reference normal 
            self.Gsub3 = cross(self.Gsub1, self.Gsub2)
            self.Gsup3 = self.Gsub3/dot(self.Gsub3, self.Gsub3)


            # Construct the covariant convective basis
            self.gsub1 = self.Gsub1 + self.u.dx(0)
            self.gsub2 = self.Gsub2 + self.u.dx(1)

            # Construct the contravariant convective basis
            self.gsup1 = contravariant_base_vector(self.gsub1, self.gsub2)
            self.gsup2 = contravariant_base_vector(self.gsub2, self.gsub1)

            # Deformed normal
            self.gsub3 = cross(self.gsub1, self.gsub2)
            self.gsup3 = self.gsub3/dot(self.gsub3, self.gsub3)

            # Deformation gradient
            gradu = outer(self.u.dx(0), self.Gsup1) + outer(self.u.dx(1), self.Gsup2)
            I = Identity(self.nsd)

            self.F = I + gradu
            self.C = self.F.T*self.F # from initial to current

            # 3x2 deformation tensors
            # TODO: check/test
            self.F_0 = as_tensor([self.Gsub1, self.Gsub2]).T
            self.F_n = as_tensor([self.gsub1, self.gsub2]).T

            # 2x2 surface metrics
            self.C_0 = self.get_metric(self.Gsub1, self.Gsub2)
            self.C_0_sup = self.get_metric(self.Gsup1, self.Gsup2)
            self.C_n = self.get_metric(self.gsub1, self.gsub2)

            # TODO: not tested. do we need these?
            self.det_C_0 = dot(self.Gsub1, self.Gsub1)*dot(self.Gsub2, self.Gsub2) - dot(self.Gsub1,self.Gsub2)**2
            self.det_C_n = dot(self.gsub1, self.gsub1)*dot(self.gsub2, self.gsub2) - dot(self.gsub1,self.gsub2)**2

            self.lambda1, self.lambda2, self.lambda3 = self.get_lambdas()

            self.I1 = inner(inv(self.C_0), self.C_n)
            self.I2 = det(self.C_n)/det(self.C_0)

        else:
            raise Exception("Could not infer spatial dimension")
        
        # Unit normals
        self.J_A = sqrt(dot(self.Gsub3, self.Gsub3))
        self.N = self.Gsub3/self.J_A
        self.j_a = sqrt(dot(self.gsub3, self.gsub3))
        self.n = self.gsub3/self.j_a

    def get_metric(self, i,j):
        return as_tensor([[dot(i, i), dot(i, j)], [dot(j, i), dot(j, j)]])

    def get_PK2(self):
        '''
        2nd Piola-Kirchhoff Stress
        '''
        # S = mem.material.mu*(dolfin.inv(C_0) - (dolfin.det(C_0)/dolfin.det(C_n))*dolfin.inv(C_n))
        A = 1/self.det_C_0
        B = self.det_C_0/(det(self.C_n)**2)
        G1 = self.Gsub1
        G2 = self.Gsub2
        g1 = self.gsub1
        g2 = self.gsub2

        G1G1 = dot(G1, G1)
        G2G2 = dot(G2, G2)
        G1G2 = dot(G1, G2)

        g1g1 = dot(g1, g1)
        g2g2 = dot(g2, g2)
        g1g2 = dot(g1, g2)

        mu = self.material.mu
        S = mu*as_matrix([[A*G2G2 - B*g2g2, -A*G1G2 + B*g1g2], [-A*G1G2 + B*g1g2, A*G1G1 - B*g1g1]])
        return S

    def get_lambdas(self):
        C_n = self.F_n.T*self.F_n
        C_0 = self.F_0.T*self.F_0
        I1 = inner(inv(C_0), C_n)
        I2 = det(C_n)/det(C_0)
        delta = sqrt(I1**2 -4*I2)

        lambda1 = sqrt(0.5*(I1 + delta))
        lambda2 = sqrt(0.5*(I1 - delta))
        lambda3 = sqrt(det(self.C_0)/det(self.C_n))

        return lambda1, lambda2, lambda3

    def get_I1(self):
        return inner(inv(self.C_0), self.C_n)

    def get_I2(self):
        return det(self.C_n)/det(self.C_0)

    def get_position(self):
        return self.gamma + self.u

    def calculate_volume(self, u):  # FIXME - need to fix volume calc for open membranes!!!
        # FIXME: Why is this a function of u???
        '''
        Calculates the current volume of the membrane with a deformation given by dolfin function u.
        We do this by integrating a function with unit divergence:
        V = \int dV = \frac{1}{dim} x \cdot n dA

        Args:
            u (Type): Dolfin function

        '''
        volume = assemble((1/self.nsd)*dot(self.gamma + u, self.gsub3)*dx(self.mesh))
        return volume

    def get_c(self):
        '''
        Create scalar indicator function c:
            c=1 if wetted surface
            c=0 if dry surface
        '''
        x = self.get_position()
        self.X_H = Constant(self.kwargs.get("X_H"))
        self.x_H = Constant(self.kwargs.get("X_H"))
        if self.nsd==3:
            self.k_hat = Constant(('0', '0', '1'), name="k_hat")
        elif self.nsd==2:
            self.k_hat = Constant(('0', '1'), name="k_hat")
        
        # If the z-component of the position vector is greater than X_H, 0, else 1
        self.c = conditional(ge(dot(x, self.k_hat), self.x_H), 0, 1)
        self.x_g = (1-self.c)*x
        self.x_f = self.c*x

    def inflate(self, pressure):
        """
        Inflate a membrane to a given pressure, no external forces or tractions

        Args:
            pressure (float): The pressure of the membrane
        """

        if not hasattr(pressure, "values"):
            p = Constant(pressure)
        else:
            p = pressure

        solver_name = self.kwargs.get("inflation_solver", None)
        if ADJOINT or solver_name is None:

            # Compute first variation of Pi (directional derivative about u in the direction of v)
            
            
            F = self.DPi_int() - self.DPi_air(p)
            
            if self.kwargs.get("free_surface"):
                F -= self.liquid.DPi_0
    
            # Compute Jacobian of F
            K = derivative(F, self.u, self.du)
            problem = NonlinearVariationalProblem(F, self.u, bcs=self.bc, J=K, form_compiler_parameters={"optimize": True})

            # Create solver and call solve
            solver = NonlinearVariationalSolver(problem)
            prm = solver.parameters
    #        prm['newton_solver']['report']=True
#            prm['newton_solver']['maximum_iterations'] =200
#            prm['newton_solver']["krylov_solver"]["maximum_iterations"] =2000
    
    
#        # works for unmodified material
            prm["nonlinear_solver"] = "snes"
            prm["snes_solver"]["linear_solver"] = "cg"
    
            solver.parameters.update(prm)       
            solver.solve()
            self.inflation_solver=solver
        
        else:
            self.inflation_problem = inflation.InflationProblem(self, p)
    
            solver_class = get_solver(solver_name)
            self.inflation_solver = solver_class(self)
            self.inflation_solver.parameters["maximum_iterations"] = 150
            self.inflation_solver.solve(self.inflation_problem, self.u.vector())
            
#            import matplotlib.pyplot as plt
#            plt.plot(self.inflation_problem.PI, label=r'$\Pi_{int}$')
#            plt.legend()
        
        for phase in self.phases:
            phase.setup()

        with stop_annotating():
            self.io.write_fields()

    def DPi_int(self):
        '''
        directional derivative of \Pi_int in an arbitrary direction \delta v
        DPi_int(\phi)[\delta v]
        '''
        
#        A = diff(self.material.psi, variable(self.C_n))
#        A = self.F_n*self.PK2
#        B =  as_tensor([self.v.dx(0), self.v.dx(1)])
#        from ufl import indices
#        i,j = indices(2)
#        return A[i,j]*B[j,i]*self.thickness*self.J_A*dx(self.mesh)
#        
        return derivative(self.Pi, self.u, self.v)

    def DPi_ext(self):
        return dot(self.tractions, self.v)*dx(self.mesh)  #FIXME not implemented yet

    def DPi_air(self, p):
        '''
        p*dot(self.v, dot(self.R, self.gsub1))*dx(self.mesh)

        Args:
            p (): p should be a Constant
        '''
        print()
        return p * dot(self.v, self.gsub3)*dx(self.mesh)

    def k_load(self):
        '''
        Need to multiply by p! eq. 27 in 
        Rumpel, T., & Schweizerhof, K. (2003). 
        Volume-dependent pressure loading and its influence 
        on the stability of structures. 
        doi.org/10.1002/nme.561
        '''
        # Construct skew symmetric tensors
        from fenicsmembranes.calculus_utils import wedge
        self.W1 = wedge(self.gsub3, self.gsup1)  # outer(self.gsub3, self.gsup1) - outer(self.gsup1, self.gsub3)
        self.W2 = wedge(self.gsub3, self.gsup2)  # outer(self.gsub3, self.gsup2) - outer(self.gsup2, self.gsub3)

        k_load = (0.5)*(dot(self.W1.T*self.du, self.v.dx(0)) + \
                  dot(self.W2.T*self.du, self.v.dx(1)))*dx(self.mesh) + \
                (0.5)*dot((self.W1*self.du.dx(0) + self.W2*self.du.dx(1)),self.v)*dx(self.mesh)
        return k_load

    def solve(self, external_load, output=True):
        solver_name = self.kwargs["solver"]
        solver_class = get_solver(solver_name)
        self.solver = solver_class(self, external_load, output=output)
        return self.solver.solve()
