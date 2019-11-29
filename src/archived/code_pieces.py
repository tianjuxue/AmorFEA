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


# Old loss_function NOT using variational energy
def loss_function(self, x_control, x_state):
    # loss function is defined so that PDE is satisfied
    # x_control should be torch tensor with shape (batch, input_size)
    # x_state should be torch tensor with shape (batch, input_size)
    assert(x_control.shape == x_state.shape and len(x_control.shape) == 2)
    x_state = x_state.unsqueeze(2)
    x_control = x_control.unsqueeze(2)

    grad_x1 = torch.matmul(self.gradient_x1_operator, x_state)
    tmp_x1 = grad_x1*x_control
    lap_x1 = torch.matmul(self.gradient_x1_operator, tmp_x1)

    grad_x2 = torch.matmul(self.gradient_x2_operator, x_state)
    tmp_x2 = grad_x2*x_control
    lap_x2 = torch.matmul(self.gradient_x2_operator, tmp_x2)

    lhs = (-lap_x1 -lap_x2).squeeze()
    rhs = self.source
    loss_interior = ((lhs - rhs)**2 * self.weight_area).sum()

    # print(x_state[0,:,0].data.numpy())
    # exit()

    grad_n = grad_x1.squeeze()*self.normal_x1 + grad_x2.squeeze()*self.normal_x2
    loss_boundary = ((grad_n - self.traction)**2 * self.weight_length).sum()

    # print(loss_interior)
    # print(loss_boundary)

    loss = loss_interior

    return loss

def initialization(self):
    # Can be much more general
    self.gradient_x1_operator = torch.tensor(self.graph.gradient_x1).float()
    self.gradient_x2_operator = torch.tensor(self.graph.gradient_x2).float()
    self.weight_area = torch.tensor(self.graph.weight_area).float()
    
    normal_x1, normal_x2, weight_length = self.graph.assemble_normal(self.graph.boundary_flags_list[2])
    self.normal_x1 =  torch.tensor(normal_x1).float()
    self.normal_x2 =  torch.tensor(normal_x2).float()
    self.weight_length =  torch.tensor(weight_length).float()

    self.source = torch.ones(self.graph.num_vertices)
    self.traction = torch.zeros(self.graph.num_vertices)

    bc_flag_1 = torch.tensor(self.graph.boundary_flags_list[0]).float()
    bc_value_1 = 2.*bc_flag_1
    bc_flag_2 = torch.tensor(self.graph.boundary_flags_list[1]).float()
    bc_value_2 = 1.*bc_flag_2
    bc_value = bc_value_1 + bc_value_2

    interior_flag = torch.ones(self.graph.num_vertices) - bc_flag_1 - bc_flag_2

    self.graph_info = [bc_value, interior_flag]