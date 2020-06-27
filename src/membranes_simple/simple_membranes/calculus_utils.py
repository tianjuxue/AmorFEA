#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:43:18 2019

@author: alexanderniewiarowski
"""
from dolfin import outer, dot

def wedge(a,b):
    '''
    Calculates the wedge product of vectors a and b

    .. math::

        a \wedge b : = a \otimes b - b\otimes a

    '''
    return outer(a, b) - outer(b, a)

def contravariant_base_vector(i, j):
    #TODO: documentation
    '''
    Gets :math:`g^i` the contravariant base vector of :math:`g_i` in terms of :math:`g_j`

    .. math::
        g^i = \\frac{(g_i \wedge g_j) \cdot g_j} { g_i \cdot ((g_i \wedge g_j) \cdot g_j)}
    '''
    supi = dot(wedge(i, j), j) / dot(i, dot(wedge(i, j), j))
    return supi

