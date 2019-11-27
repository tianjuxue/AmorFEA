"""Poisson problem definition"""


class Poisson(object):
    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()

    def _build_mesh(self):
        raise NotImplementedError()

    def _build_function_space(self):
        raise NotImplementedError()

    def solve_problem_weak_form(self, boundary_fn=None):
        raise NotImplementedError()