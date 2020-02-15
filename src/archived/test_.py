import fenics as fa
import dolfin_adjoint as da
import numpy as np
import moola


n = 64
alpha = da.Constant(0)
mesh = da.UnitSquareMesh(n, n)

# case_flag = 0 runs the default case by
# http://www.dolfin-adjoint.org/en/latest/documentation/poisson-mother/poisson-mother.html
# change to any other value runs the other case 
case_flag = 1

x = fa.SpatialCoordinate(mesh)
w = da.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)

V = fa.FunctionSpace(mesh, "CG", 1)
W = fa.FunctionSpace(mesh, "DG", 0)

# g is the ground truth for source term
# f is the control variable
if case_flag == 0:
    g = da.interpolate(da.Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3), W)
    f = da.interpolate(da.Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3), W)
else:
    g = da.interpolate(da.Expression(("sin(2*pi*x[0])"),  degree=3), W)
    f = da.interpolate(da.Expression(("sin(2*pi*x[0])"),  degree=3), W)

u = da.Function(V, name='State')
v = fa.TestFunction(V)

F = (fa.inner(fa.grad(u), fa.grad(v)) - f * v) * fa.dx
bc = da.DirichletBC(V, 0.0, "on_boundary")
da.solve(F == 0, u, bc)

d = da.Function(V)
d.vector()[:] = u.vector()[:]

J = da.assemble((0.5 * fa.inner(u - d, u - d)) * fa.dx + alpha / 2 * f ** 2 * fa.dx)
control = da.Control(f)
rf = da.ReducedFunctional(J, control)

# set the initial value to be zero for optimization
f.vector()[:] = 0
# N = len(f.vector()[:])
# f.vector()[:] = np.random.rand(N)*2 -1

problem = da.MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(f)
solver = moola.BFGS(problem, f_moola, options={'jtol': 0,
                                               'gtol': 1e-9,
                                               'Hinit': "default",
                                               'maxiter': 100,
                                               'mem_lim': 10})
sol = solver.solve()
f_opt = sol['control'].data

file = fa.File('f.pvd')
f_opt.rename('f', 'f')
file << f_opt

file = fa.File('g.pvd')
g.rename('g', 'g')
file << g