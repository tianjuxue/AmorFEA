class OptimizerDolfinIdentification(OptimizerDolfin):
    """
    Level 2
    """

    def __init__(self, args):
        super(OptimizerDolfinIdentification, self).__init__(args)
        self.target_dof_noise_free, self.target_u_noise_free = target_solution_id(
            self.args, self.poisson)
        self.sigma = 0.
        self.add_noise()

    def optimize(self):
        x_initial = np.array([0., 0., 0.5, 0.5])
        # x_initial = np.zeros(4)
        options = {'eps': 1e-15, 'maxiter': 1000,
                   'disp': True} 
        start = time.time()
        res = opt.minimize(fun=self._objective,
                           x0=x_initial,
                           method='CG',
                           jac=self._derivative,
                           callback=None,
                           options=options)
        end = time.time()
        print("Time elasped {}".format(end - start))
        return res.x

    def add_noise(self):
        noise = self.sigma * \
            torch.normal(torch.zeros(self.target_dof_noise_free.shape[1]), 1)
        self.target_dof = self.target_dof_noise_free + noise
        self.target_u = da.Function(self.poisson.V)
        self.target_u.vector()[:] = self.target_u_noise_free.vector()[
            :] + noise.data.numpy()

    def debug(self):
        truth_map = []
        coo = torch.tensor(self.poisson.coo_dof, dtype=torch.float)
        for i in range(self.args.input_size):
            L = self._obj(coo[i])
            truth_map.append(L.data.numpy())
        truth_map = np.array(truth_map)
        scalar_field_paraview(self.args, truth_map, self.poisson, "opt_tm")
        

class OptimizerDolfinIdentificationSurrogate(OptimizerDolfinIdentification, OptimizerDolfinSurrogate):
    """
    Level 3
    """

    def __init__(self, args):
        super(OptimizerDolfinIdentificationSurrogate, self).__init__(args)

    def _obj(self, p):
        coo = torch.tensor(self.poisson.coo_dof, dtype=torch.float)
        source = 100 * \
            p[2] * torch.exp((-(coo[:, 0] - p[0])**2 -
                              (coo[:, 1] - p[1])**2) / (2 * 0.01 * p[3]))
        solution = self.model(source.unsqueeze(0))
        diff = solution - self.target_dof

        tmp = batch_mat_vec(self.B_sp, diff)
        L_diff = diff * tmp
        # diff = (solution - self.target_dof)**2

        L = 0.5 * L_diff.sum()
        return L


class OptimizerDolfinIdentificationAdjoint(OptimizerDolfinIdentification, OptimizerDolfinAdjoint):
    """
    Level 3
    """

    def __init__(self, args):
        super(OptimizerDolfinIdentificationAdjoint, self).__init__(args)

    def _obj(self, x):
        p = da.Constant(x)
        x = fa.SpatialCoordinate(self.poisson.mesh)
        self.poisson.source = 100 * p[2] * fa.exp((-(x[0] - p[0]) * (
            x[0] - p[0]) - (x[1] - p[1]) * (x[1] - p[1])) / (2 * 0.01 * p[3]))
        u = self.poisson.solve_problem_variational_form()
        L_tape = da.assemble(
            (0.5 * fa.inner(u - self.target_u, u - self.target_u)) * fa.dx)
        L = float(L_tape)
        return L, L_tape, p

    def _objective(self, x):
        L, _, _ = self._obj(x)
        return L

    def _derivative(self, x):
        _, L_tape, p = self._obj(x)
        control = da.Control(p)
        J_tape = da.compute_gradient(L_tape, control)
        J = J_tape.values()
        return J



def target_solution_id(args, pde):
    pde.source = da.Expression("k*100*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*0.01*l) )",
                               k=1, l=1, x0=0.1, x1=0.1, degree=3)
    pde.source = da.interpolate(pde.source, pde.V)
    save_solution(args, pde.source, 'opt_fem_f')
    u = pde.solve_problem_variational_form()
    save_solution(args, u, 'opt_fem_u')
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)
    return dof_data, u

def run_id(args):
    sigma_list = [1e-3, 1e-2, 1e-1]
    optimizer_nn = OptimizerDolfinIdentificationSurrogate(args)
    optimizer_ad = OptimizerDolfinIdentificationAdjoint(args)
    for sigma in sigma_list:
        optimizer_nn.sigma = sigma
        optimizer_nn.add_noise()
        x_nn = optimizer_nn.optimize()
        print('nn optimized x is', x_nn)

        optimizer_ad.sigma = sigma
        optimizer_ad.add_noise()
        x_ad = optimizer_ad.optimize()
        print('ad optimized x is', x_ad)
        print('\n')
