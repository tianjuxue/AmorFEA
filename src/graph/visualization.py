import fenics as fa


def save_solution(args, solution, name):
    file = fa.File(args.root_path + '/' +
                   args.solutions_path + '/' + name + '.pvd')
    solution.rename('u', 'u')
    file << solution


def scalar_field_paraview(args, attribute, pde, name):
    solution = fa.Function(pde.V)
    solution.vector()[:] = attribute
    save_solution(args, solution, name)
