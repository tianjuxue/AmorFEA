def sparse_representation(A):
    # seems stupid... first compute the size
    counter = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j]**2 > 1e-10:
                counter += 1
    index = torch.zeros((2, counter), dtype=torch.long)
    value = torch.zeros(counter, dtype=torch.float)

    counter = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j]**2 > 1e-10:
                index[0, counter] = i
                index[1, counter] = j
                value[counter] = A[i][j]
                counter += 1

    A_sp = torch.sparse.FloatTensor(index, value, torch.Size([len(A), len(A)]))
    print("Create sparse representation of A")
    return A_sp


# Constitutive relationships
def _energy_density(self, u):
    # variational energy density of u
    energy = 0.5*fa.dot(fa.grad(u), fa.grad(u)) - u*self.source
    return energy

def energy(self, u):
    return fa.assemble(self._energy_density(u) * fa.dx)

def solve_problem_variational_form(self, boundary_fn=None):
    u = fa.Function(self.V)
    du = fa.TrialFunction(self.V)
    v = fa.TestFunction(self.V)
    E = self._energy_density(u)*fa.dx
    dE = fa.derivative(E, u, v)
    jacE = fa.derivative(dE, u, du) 

    bcs = []
    if boundary_fn is not None:
        boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
        bcs = bcs + [boundary_bc]

    fa.solve(dE == 0, u, bcs, J=jacE)

    return u

# prm = solver.parameters
# prm['newton_solver']['relaxation_parameter'] = self.args.relaxation_parameter
# prm['newton_solver']['linear_solver'] =   self.args.linear_solver
# prm['newton_solver']['maximum_iterations'] = self.args.max_newton_iter
# prm['newton_solver']['relative_tolerance'] = self.args.rel_tol
# prm['newton_solver']['absolute_tolerance'] =  self.args.abs_tol