#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:28:14 2019

@author: alexanderniewiarowski
"""

from dolfin import * #Constant, det, tr, inner, inv, as_matrix, sqrt, dot,as_tensor, split,as_vector, outer,cos, sin, exp
import ufl
from . import Material, register_material

try:
    from dolfin_adjoint import * #Constant
    ADJOINT = True
except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False


@register_material('Incompressible NeoHookean')
class IncompressibleNeoHookean(Material):

    def __init__(self, membrane, kwargs):
        mem = membrane

        # Define material - incompressible Neo-Hookean
        self.nu = 0.5  # Poisson ratio, incompressible

        E = kwargs.get("E", None)
        if E is not None:
            self.E = Constant(E, name="Elastic modulus")

        mu = kwargs.get("mu", None)
        if mu is not None:
            self.mu = Constant(mu, name="Shear modulus")
        else:
            self.mu = Constant(self.E/(2*(1 + self.nu)), name="Shear modulus")


        mem.I_C = tr(mem.C) + 1/det(mem.C) - 1

        '''
        # Alternative formulation:
        C_n = mem.C_n  #mem.F_n.T*mem.F_n
        C_0 = mem.C_0  #mem.F_0.T*mem.F_0
        C_0_sup = mem.C_0_sup
        i,j = ufl.indices(2)
        I1 = C_0_sup[i,j]*C_n[i,j]
            # TODO: why are these not equal to the third
#            print(assemble(det(membrane.C_n)/det(membrane.C_0)*dx(membrane.mesh)))
#            print(assemble((membrane.det_C_n)/(membrane.det_C_0)*dx(membrane.mesh)))
#            print(assemble(det(membrane.C)*dx(membrane.mesh)))
            # blows up if use other definiton of C
        mem.I_C = I1 + 1/det(mem.C) #det(C_0)/det(C_n)# mem.det_C_0/det(mem.C_n)
        '''
        self.psi = 0.5*self.mu*(mem.I_C - Constant(mem.nsd))


@register_material('Incompressible NeoHookean Mosler Weighted')
class IncompressibleNeoHookeanMosler(Material):

    def __init__(self, membrane, kwargs):
        self.mem = mem = membrane

        # Define material - incompressible Neo-Hookean
        self.nu = 0.5  # Poisson ratio, incompressible

        E = kwargs.get("E", None)
        if E is not None:
            self.E = Constant(E, name="Elastic modulus")

        mu = kwargs.get("mu", None)
        if mu is not None:
            self.mu = Constant(mu, name="Shear modulus")
        else:
            self.mu = Constant(self.E/(2*(1 + self.nu)), name="Shear modulus")
        mem.I_C = tr(mem.C) + 1/det(mem.C) - 1
        self.psi  = 0.5*self.mu*(mem.I_C - Constant(mem.nsd))
        self.alpha = Constant(kwargs.get("alpha", None))
        self.update()
    
    def update(self):
        mem = self.mem
        mesh = mem.mesh
        
        F = outer(mem.gsub1, mem.Gsup1) + outer(mem.gsub2, mem.Gsup2)
        A = as_tensor([mem.Gsub1, mem.Gsub2, mem.Gsub3]).T
        
        C = inv(A) * F.T*F * A
        C = as_tensor([[C[0, 0], C[0, 1]], 
                       [C[1, 0], C[1, 1]]])       
        
        S = FunctionSpace(mesh, "CG", 2)
        We = VectorElement("CG", mesh.ufl_cell(), degree=2, dim=3)
        W = FunctionSpace(mesh, We)
    
        w = Function(W)
        a_sq, b_sq, alpha = split(w)
    
        v = TestFunction(W)
        dw = TrialFunction(W)
        
        N = as_vector([cos(alpha), sin(alpha)])
        M = as_vector([-sin(alpha), cos(alpha)])
        
        C_r  = C + a_sq*outer(N, N) + b_sq*outer(M, M)
        
        # Lagrangian
        psi_r = 0.5*self.mu*(tr(C_r) + det(C_r)**-1 - Constant(3))*dx
    
        # First derivative
        J = derivative(psi_r, w, v)
        
        # second derivative
        H = derivative(J, w, dw)
    
        # set up a petsc snes nonlinear solver
        snes_solver_parameters = {"nonlinear_solver": "snes",
                                  "snes_solver": {"linear_solver": "lu",
                                                  "maximum_iterations": 100,
                                                  "report": True,
                                                  "error_on_nonconvergence": False}}
        # create the problem (no boundary conditions)
        problem = NonlinearVariationalProblem(J, w, bcs=None, J=H)
    
        # We need to impose handle the bounds
        lower = Function(W) # 0, 0, 0
        upper = Function(W)  # 0, 0, pi
    
        zero = Function(S)
        zero.vector()[:] = 0
    
        from numpy import infty
        pinfty = Function(S)
        pinfty.vector()[:] = 10
    
        # assign the bound values to the functions    
        fa = FunctionAssigner(W, [S, S, S])
        fa.assign(lower, [zero, zero, zero])
        fa.assign(upper, [pinfty, pinfty, interpolate(Constant(pi), S)])
    
        # set bounds and solve
        problem.set_bounds(lower, upper)
    
        solver  = NonlinearVariationalSolver(problem)
        solver.parameters.update(snes_solver_parameters)
        solver.solve()

        self.psi = (self.alpha*self.psi + (1-self.alpha)*0.5*self.mu*(tr(C_r) + det(C_r)**-1 - Constant(3)))