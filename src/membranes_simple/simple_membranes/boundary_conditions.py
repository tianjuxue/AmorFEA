#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:49:47 2020

@author: alexanderniewiarowski
"""

from dolfin import Constant, CompiledSubDomain, DirichletBC

try:
    from dolfin_adjoint import *
    ADJOINT = True
except ModuleNotFoundError:
    ADJOINT = False


# UNIT INTERVAL MESH BOUNDARIES
bd_all_1D = CompiledSubDomain("near(x[0], 0) || near(x[0], 1)")
bd_x_mid_1D = CompiledSubDomain("near(x[0], 0.5)")


# UNIT SQUARE MESH BOUNDARIES
bd_all = CompiledSubDomain("on_boundary")

# left and right
bd_x = CompiledSubDomain("(near(x[0], 0) && on_boundary) || (near(x[0], 1) && on_boundary)")

# bottom and top
bd_y = CompiledSubDomain("(near(x[1], 0) && on_boundary) || (near(x[1], 1) && on_boundary)")

# x=0.5
bd_x_mid = CompiledSubDomain("near(x[0], 0.5)")

# y=0.5
bd_y_mid = CompiledSubDomain("near(x[1], 0.5)")

# These are provided for convenience 
def rollerBC(membrane):
    bc =[]
    if membrane.nsd==2:
        # 
        bc.append(DirichletBC(membrane.V.sub(1), Constant(0), bd_all_1D))
        bc.append(DirichletBC(membrane.V.sub(0), Constant(0), bd_x_mid_1D))

    elif membrane.nsd==3:
        # This constrains the thickness at the support but is not plane strain
        bc.append(DirichletBC(membrane.V.sub(1), Constant(0), bd_all))  
        bc.append(DirichletBC(membrane.V.sub(0), Constant(0), bd_x_mid))
        bc.append(DirichletBC(membrane.V.sub(2), Constant(0), bd_x))
    return bc

def pinBC(membrane):
    bc = []
    if membrane.nsd==2:
        bc.append(DirichletBC(membrane.V, Constant((0,0)), bd_all_1D))

    elif membrane.nsd==3:
        # This constrains the thickness at the support but is not plane strain
        bc.append(DirichletBC(membrane.V.sub(1), Constant(0), bd_all))
        bc.append(DirichletBC(membrane.V, Constant((0,0,0)), bd_x))
#        bc.append(DirichletBC(self.V, Constant((0,0,0)), self.bd_all))
    return bc


def bc_junk():
    # Define boundary conditions
    # out of plane is y so V.sub(1)
    # bc.append( DirichletBC(V.sub(2), Constant(0), Mid_bd) ) #this keeps the structure symmetric
    # bc.append( DirichletBC(V.sub(1), Constant("0"), "fabs(x[0] - 0.5) < DOLFIN_EPS") )

    #bc.append( DirichletBC(V.sub(1), Constant(0), Pin_bd) )
    #bc.append( DirichletBC(V.sub(0), Constant(0), Pin_bd) )

    # bc.append( DirichletBC(V.sub(1), Constant(0), All_bd)) #add this to enforce plane strain in the out of plane directional
    #which forces u.dx(1) to be equal to zero. We also have plane stress in the normal direction meaning only membrane action because the
    # term u.dx(2) does not appear in the grad(u) term.
    pass




    