import fenics as fa
import dolfin_adjoint as da
import numpy as np

n = 64

mesh = da.UnitSquareMesh(n, n)

V = fa.FunctionSpace(mesh, "CG", 1)
W = fa.FunctionSpace(mesh, "DG", 0)

f = da.interpolate(da.Expression("x[0]+x[1]", name='Control', degree=1), W)
u = da.Function(V, name='State')
v = fa.TestFunction(V)

F = (fa.inner(fa.grad(u), fa.grad(v)) - f * v) * fa.dx
bc = da.DirichletBC(V, 0.0, "on_boundary")
da.solve(F == 0, u, bc)

x = fa.SpatialCoordinate(mesh)
w = da.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1 / (2 * np.pi ** 2)
d = da.Expression("d*w", d=d, w=w, degree=3)

alpha = da.Constant(1e-6)
J = da.assemble((0.5 * fa.inner(u - d, u - d)) * fa.dx + alpha / 2 * f ** 2 * fa.dx)
control = da.Control(f)

dJdm = da.compute_gradient(J, control)
k = np.array(dJdm.vector()[:])
print(k.shape)