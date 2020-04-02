#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:31:14 2020

@author: alexanderniewiarowski
"""
from dolfin import *
from simple_membranes.parametric_membrane import *
from simple_membranes.ligaro import AdjointGeoWEta as Cylinder

# First we need geometry information. For the geometrically exact membrane, it is 
# necessary to provide a class that contains the mapping gamma and the analytical tangent basis.::

# generate intitial parametric geometry (semicircle)
# This uses the parametrization from my paper 
# First arg width, second is eta=L/R (circumference to radius ratio as explained in the paper)
rad = 0.21     # radius of cylinder
geo = Cylinder(2*rad, -pi, dim=3)

# we set up a dictionary with some desired settings.::
input_dict = {
        "resolution": [25,5],  # num elements in x and y directions
        "geometry": geo,
        "thickness": 0.01,
        "Gas_law": "Boyle",  # This is the simple gas, default is IsentropicGas
        "pressure": 0.045,
        "material": 'Incompressible NeoHookean',
        "mu": 1.0,
        "output_file_path": 'membrane-demo',
        "Boundary Conditions": "Pinned",   # Can also try "Roller"
        "solver": 'Naive'
        }

# Finally, we create a membrane object by calling:
membrane = ParametricMembrane(input_dict)
# This call created an instance of the ParametricMembrane and calculated the gas constant. It is ready for external loading. 

# You can inspect the result in Paraview - select "Warp by Vector, with the orientation array being "u_vector"

# To visualize the displacement, color with "u", stretch is "l1"