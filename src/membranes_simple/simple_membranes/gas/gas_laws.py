#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:23:17 2020

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

from . import register_gas_law

@register_gas_law('Isentropic Gas')
class IsentropicGas():
    """
        p = p_0(V_0/V)^k

        Calculates and saves the gas constant by inflating membrane to p_0.

        Used for initialization or when internal air mass changes.

    """
    def __init__(self, membrane):
        self.ready = False
        self.mem = mem = membrane
        self.kappa = mem.kwargs.get("kappa", 1)
        self.free_surface = mem.kwargs.get("free_surface", False)
        # inflate structure to p_0
        self.p_0 = mem.p_0

        
        # save initial inflated volume
        if self.free_surface:
            mem.get_c()
            mem.data["free_srf"] = Function(mem.W, name="free_srf", label="free surface")
            mem.data["c"] = Function(mem.Z, name="c", label="c")
            
            with stop_annotating():
                # visualize free surface
                # matrix to project onto the x axis
                if mem.nsd==2:
                    P = as_matrix([[1,0],
                                   [0,0]])
                    mem.fs = P*mem.x_g - (1 - mem.c)*Expression(('x[0]',  '-x_H'), degree=2, x_H=mem.x_H)
                    
                elif mem.nsd==3:    
                    P = as_matrix([[1,0,0],
                                   [0,0,0],
                                   [0,0,0]])
              
                    mem.fs = P*mem.x_g - (1 - mem.c)*Expression(('x[0]', '-x[1]',  '-x_H'), degree=2, x_H=mem.x_H)

                def _write(v, V, f):
                    return project(v, V, function=f)
                mem.io.add_extra_output_function(project(mem.fs, mem.W, function=mem.data["free_srf"]))
                mem.io.add_extra_output_function(project(mem.c, mem.Z, function=mem.data["c"]))
        
    def setup(self):
        mem = self.mem
        if self.free_surface:
#                mem.io.write_fields()
                
#            mem.inflate(self.p_0)
#            with stop_annotating():
#                project(mem.c, mem.Z, function=mem.data["c"])
#                project(fs, mem.W, function=mem.data["free_srf"])
#                mem.io.write_fields()
            
            self.mag_n0 = dot((1-mem.c)*mem.gsub3, mem.k_hat)
            self.n0 = self.mag_n0*mem.k_hat
            
            if mem.nsd ==2:
                self.V_0 = assemble((1/mem.nsd)*(dot(mem.x_g, (1-mem.c)*mem.gsub3) - dot(Constant(('0', mem.x_H)), mem.gas.n0))*dx(mem.mesh, degree=5))
            elif mem.nsd ==3:
                self.V_0 = assemble((1/mem.nsd)*(dot(mem.x_g, (1-mem.c)*mem.gsub3) - dot(Constant((0,0, mem.x_H)), mem.gas.n0))*dx(mem.mesh, degree=5))
            # assemble((1/mem.nsd)*dot(mem.x_g - (1 - mem.c)*Constant(('0', mem.X_H)), (mem.gsub3))*dx(mem.mesh, degree=5))

            
            self.S = assemble(sqrt(dot(self.mag_n0, self.mag_n0))*dx(mem.mesh))  # calcualte size of free surface area by projecting normal onto z dire Rumpel 2005 eq 26
            
            # initial displacement of free surface associated with inflation, subtract out?
            self.delta_0 = (-1/self.S)*assemble(dot(mem.c*mem.u, mem.c*mem.gsub3)*dx(mem.mesh))
            
        else:
#            mem.inflate(self.p_0)
            self.V_0 = mem.calculate_volume(mem.u)
            
        # TODO: why doesn't dolfin adjoint like the Constant? try AdjFloat?
        if ADJOINT:
            self.constant = self.p_0*self.V_0**self.kappa
        else:
            self.constant = Constant(self.p_0*self.V_0**self.kappa, name='gas_constant')

    def update_pressure(self):
        mem = self.mem
        
        if self.free_surface:
            if mem.nsd ==2:
                self.V = assemble((1/mem.nsd)*dot(mem.x_g - (1 - mem.c)*Constant(('0', mem.x_H)), (mem.gsub3))*dx(mem.mesh, degree=5))
            elif mem.nsd ==3:
                self.V = assemble((1/mem.nsd)*dot(mem.x_g - (1 - mem.c)*Constant(('0',0, mem.x_H)), (mem.gsub3))*dx(mem.mesh, degree=5))
            self.S = assemble(sqrt(dot(self.mag_n0, self.mag_n0))*dx(mem.mesh))
        else:
            self.V = mem.calculate_volume(mem.u)
        
        self.p = self.constant/(self.V**self.kappa)

        # update dpDV 
#        self.dpdV = -self.kappa*self.constant/(self.V**(self.kappa + 1)) ##
        self.dpdV = -self.kappa*self.p/self.V
        if ADJOINT:
            return self.p
        else:
            return Constant(self.p)


@register_gas_law('Boyle')
class Boyle():
    """
    Calculates and saves the Boyle constant by inflating membrane to p_0.

    Used for initialization or when internal air mass changes.
    """
    def __init__(self, membrane, **kwargs):
        self.mem = mem = membrane
        self.p_0 = mem.p_0
        
    def setup(self):
        mem = self.mem
        # save initial inflated volume
        self.V_0 = mem.calculate_volume(mem.u)
        
        if ADJOINT:
            self.boyle = self.V_0*self.p_0
        else:
            self.boyle = Constant(self.V_0*self.p_0, name='Boyle_constant')
        
    def update_pressure(self):
        mem = self.mem
        self.V = mem.calculate_volume(mem.u)
        self.p = self.boyle/self.V
        self.dpdV = -self.boyle/(self.V**2)
#        self.dpdV = -(self.constant)/(V**2)  # boyle 
        
        
    

    
    