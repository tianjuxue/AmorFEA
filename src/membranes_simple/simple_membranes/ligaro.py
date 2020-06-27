#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:16:49 2019

@author: alexanderniewiarowski


This module contains various classes and functions to generate a parametric 
geometry based on circular arches.
The 

"""
import sympy as sp
import numpy as np
from dolfin import *

try:
    from dolfin_adjoint import Expression, Constant
    ADJOINT = True
except ModuleNotFoundError:
    ADJOINT = False

DEGREE = 3


def ccode(z):
    return sp.printing.ccode(z)


def ligaro(W, eta):
    x1 = sp.symbols('x[0]')
    A = -0.5*W
    B = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))
    C = 0.5*W
    D = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))

    X = A*sp.cos(x1*eta) + B*sp.sin(x1*eta) + C
    Z = A*sp.sin(x1*eta) - B*sp.cos(x1*eta) + D
    return X, Z


def get_L_from_eta(w, eta):
    c = np.sqrt(2*(1-np.cos(eta)))/-eta
    return w/c


class AdjointGeoWEta(object):

    def build_derivative(self, diff_args=None, name=None):
        return Expression(tuple([ccode(i.diff(*diff_args)) for i in self.gamma_sp]),
                                eta=self.eta,
                                w=self.w,
                                degree=DEGREE,
                                name=name)

    def __init__(self, w=None, eta=None, dim=None):
        self.w = Constant(w, name="width")
        self.eta = Constant(eta, name="eta")


        W, eta, x1, x2 = sp.symbols('w, eta, x[0], x[1]')
        self.dependencies = [W, eta]
        X, Z = ligaro(W, eta)
#        R = L/eta

        if dim==2:
            self.gamma_sp = gamma_sp=[X, Z]
            self.Gsub2 = None
        if dim==3:
            self.gamma_sp = gamma_sp = [X, x2, Z]

        # for dim==2 or dim ==3:
        self.gamma = Expression(tuple([ccode(i) for i in gamma_sp]),
                                eta=self.eta,
                                w=self.w,
                                degree=DEGREE,
                                name="gamma")

        self.Gsub1 = self.build_derivative([x1], name='Gsub1')

        if ADJOINT:
            # adjoint gamma
            self.gamma.dependencies = [self.w, self.eta]
            self.gamma.user_defined_derivatives = {}
            self.gamma.user_defined_derivatives[self.w] = self.build_derivative([W],
                                                           name='d_gamma_dw')
            self.gamma.user_defined_derivatives[self.eta] = self.build_derivative([eta],
                                                           name='d_gamma_deta')
            # adjoint Gsub1
            self.Gsub1.dependencies = [self.w, self.eta]
            self.Gsub1.user_defined_derivatives = {}
            self.Gsub1.user_defined_derivatives[self.w] = self.build_derivative([x1, W],
                                                           name='d_Gsub1_dw')
            self.Gsub1.user_defined_derivatives[self.eta] = self.build_derivative([x1, eta],
                                                           name='d_Gsub1_deta')


        if dim==3:
            self.Gsub2 = self.build_derivative([x2], name='Gsub2')

            if ADJOINT:
                self.Gsub2.dependencies = [self.w, self.eta]
                self.Gsub2.user_defined_derivatives = {}
                self.Gsub2.user_defined_derivatives[self.w] = self.build_derivative([x2, W],
                                                               name='d_Gsub2_dw')
                self.Gsub2.user_defined_derivatives[self.eta] = self.build_derivative([x2, eta],
                                                               name='d_Gsub2_deta')


class ParametricAdjointGeo(object):

    def __init__(self, w=None, L=None, dim=3):

        w_val = w
        self.length = L

        W, eta, L = sp.symbols('w, eta, length')
        x1, x2 = sp.symbols('x[0], x[1]')

        CC = W/L
        a, b,c,d,e,f = sp.symbols('a, b,c,d,e,f')
        eta = a + b*CC + c*CC**2 + d*CC**3 + e*sp.cos(f*CC)

        eta_fit= np.linspace(-6.28,0.001,100)
        x_data  = np.sqrt(2*(1-np.cos(eta_fit))/eta_fit**2)

        def test_func(x, a, b,c,d,e,f):
            return a + b*x + c*x**2 + d*x**3 + e*np.cos(f*x)

        from scipy import optimize
        params, params_covariance = optimize.curve_fit(test_func, x_data, eta_fit, maxfev=10000)

        X, Z = ligaro(W, eta)

        R = L/eta

        if dim==2:
            self.Gsub2 = None
            gamma_sp = [None, None]
            gamma_sp[0] = X
            gamma_sp[1] = Z

            self.gamma = Expression((ccode(gamma_sp[0]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1]).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="gamma")

            # spatial derivative of above gamma (needed for PDE), similarly for \partial\gamma / \partial x2
            self.Gsub1 = Expression((ccode(gamma_sp[0].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="Gsub1")


            # Now need the derivatives of the above wrt length for dolfin-adjoint
            self.d_gamma = Expression((ccode(gamma_sp[0].diff(L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(L)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="d_gamma")

            self.d_Gsub1 = Expression((ccode(gamma_sp[0].diff(x1, L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1, L)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="d_Gsub1")

            # per the example, store the derivatives as dictionaries
            self.gamma.dependencies = [self.length]
            self.gamma.user_defined_derivatives = {self.length: self.d_gamma}

            self.Gsub1.dependencies = [self.length]
            self.Gsub1.user_defined_derivatives = {self.length: self.d_Gsub1}

        if dim==3:
            gamma_sp = [None, None, None]
            gamma_sp[0] = X
            gamma_sp[1] = x2
            gamma_sp[2] = Z

            self.gamma = Expression((ccode(gamma_sp[0]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2]).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="gamma")

            # spatial derivative of above gamma (needed for PDE), similarly for \partial\gamma / \partial x2
            self.Gsub1 = Expression((ccode(gamma_sp[0].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x1)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="Gsub1")

            self.Gsub2 = Expression((ccode(gamma_sp[0].diff(x2)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x2)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x2)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="Gsub2")

            # Now need the derivatives of the above wrt length for dolfin-adjoint
            self.d_gamma = Expression((ccode(gamma_sp[0].diff(L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(L)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="d_gamma")

            self.d_Gsub1 = Expression((ccode(gamma_sp[0].diff(x1, L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1, L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x1, L)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="d_Gsub1")

            self.d_Gsub2 = Expression((ccode(gamma_sp[0].diff(x2, L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x2, L)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x2, L)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=self.length,
                                    w=w_val,
                                    a=params[0],
                                    b=params[1],
                                    c=params[2],
                                    d=params[3],
                                    e=params[4],
                                    f=params[5],
                                    degree=DEGREE,
                                    name="d_Gsub2")

            # per the example, store the derivatives as dictionaries
            self.gamma.dependencies = [self.length]
            self.gamma.user_defined_derivatives = {self.length: self.d_gamma}

            self.Gsub1.dependencies = [self.length]
            self.Gsub1.user_defined_derivatives = {self.length: self.d_Gsub1}

            self.Gsub2.dependencies = [self.length]
            self.Gsub2.user_defined_derivatives = {self.length: self.d_Gsub2}


class ParametricGeometryLigaro(object):

    def __init__(self, c=None, length=1, dim=3):

        eta, L = sp.symbols('eta, length')
        eta = sp.nsolve(c**2 * eta**2 - 2*(1-sp.cos(eta)), eta, -5)

        W = c*L

        X, Z = ligaro(W, eta)

        # center at -cL/2,
        R = L/eta
        A = -0.5*W
        cx = A
        cy = W*sp.sin(eta)/(2*(1-sp.cos(eta)))
        self.eta = eta
        self.radius = R.evalf(subs={L: 1})
        self.center = [cx.evalf(subs={L: length}), cy.evalf(subs={L: length})]

        x1, x2 = sp.symbols('x[0], x[1]')

        gamma_sp = [None, None, None]
        gamma_sp[0] = X
        gamma_sp[1] = x2
        gamma_sp[2] = Z

        if dim==2:
            del gamma_sp[1]
        self._build_geo(gamma_sp, length, x1, x2, dim)

    def _build_geo(self, gamma_sp, length, x1, x2, dim):

        if dim==2:
            self.Gsub2 = None
            # Construct the actual fenics map
            self.gamma = Expression((ccode(gamma_sp[0]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1]).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=length,
                                    degree=DEGREE,
                                    name="gamma")

            # Get the induced base from the mapping to the referrential geometry
            # G_2 = ∂X/xi^2
            self.Gsub1 = Expression((ccode(gamma_sp[0].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=length,
                                    degree=DEGREE,
                                    name="Gsub1")

        if dim==3:
            # Construct the actual fenics map
            self.gamma = Expression((ccode(gamma_sp[0]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1]).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2]).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=length,
                                    degree=DEGREE,
                                    name="gamma")

            # Get the induced base from the mapping to the referrential geometry
            # G_2 = ∂X/xi^2
            self.Gsub1 = Expression((ccode(gamma_sp[0].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x1)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x1)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=length,
                                    degree=DEGREE,
                                    name="Gsub1")

            self.Gsub2 = Expression((ccode(gamma_sp[0].diff(x2)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[1].diff(x2)).replace('M_PI', 'pi'),
                                     ccode(gamma_sp[2].diff(x2)).replace('M_PI', 'pi')),
                                    pi=pi,
                                    length=length,
                                    degree=DEGREE,
                                    name="Gsub2")


class LigaroVaried(ParametricGeometryLigaro):

    def __init__(self, c=None, length=1, T=1, a=0.15, omega=2):
        super().__init__(c=c, length=length)
        x1, x2 = sp.symbols('x[0], x[1]')
        gamma_sp = [None, None, None]

        gamma_sp[0] = (a*self.radius*sp.sin(omega*pi*x2)  - self.radius)*sp.cos(pi-x1*pi) - self.center[0]
        gamma_sp[1] = x2*T
        gamma_sp[2] = (a*self.radius*sp.sin(omega*pi*x2)  - self.radius)*sp.sin(pi-x1*pi) - self.center[1]
        self._build_geo(gamma_sp, length, x1, x2)

'''
class LigaroVaried(ParametricGeometryLigaro):

    def __init__(self, length, a):
        x1, x2 = sp.symbols('x[0], x[1]')
        eta, L = sp.symbols('eta, length')


        eta = -sp.sin(a*pi*x2) - 3
        c = sp.sqrt((2-sp.cos(eta))/eta**2)
        L=  -c*eta
        W = c*L

        A = -0.5*W
        B = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))
        C = 0.5*W
        D = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))

        # center at -cL/2,
        R = L/eta
        cx = A
        cy = W*sp.sin(eta)/(2*(1-sp.cos(eta)))
        self.eta = eta
        self.radius = R.evalf(subs={L: 1})
        self.center = [cx.evalf(subs={L: L}), cy.evalf(subs={L: L})]


        gamma_sp = [None, None, None]
        gamma_sp[0] = A*sp.cos(x1*L/R) + B*sp.sin(x1*L/R) #+  C
        gamma_sp[1] = x2
        gamma_sp[2] = A*sp.sin(x1*L/R) - B*sp.cos(x1*L/R) + D


#        x1, x2 = sp.symbols('x[0], x[1]')
#        gamma_sp = [None, None, None]
#
#        gamma_sp[0] = (a*self.radius*sp.sin(2*pi*x2)  - self.radius)*sp.cos(pi-x1*pi) - self.center[0]
#        gamma_sp[1] = x2*2
#        gamma_sp[2] = (a*self.radius*sp.sin(2*pi*x2)  - self.radius)*sp.sin(pi-x1*pi) - self.center[1]
        self._build_geo(gamma_sp, length, x1, x2)
'''
