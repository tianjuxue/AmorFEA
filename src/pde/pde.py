"""Base class for PDEs"""
import fenics as fa

class PDE(object):
    """Base class for PDEs.
    """

    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()

    def _build_mesh(self):
        raise NotImplementedError()

    def _build_function_space(self):
        raise NotImplementedError()

    def _energy_density(self, u):
        raise NotImplementedError()

    def energy(self, u):
        return fa.assemble(self._energy_density(u) * fa.dx)

    def solve_problem(self,
                      boundary_fn=None):

        u = fa.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)

        E = self._energy_density(u) * fa.dx

        bcs = []

        # If boundary functions are defined using one global function
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            bcs = bcs + [boundary_bc]

        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du)
   
        fa.solve(dE == 0, u, bcs, J=jacE)

        return u
      

