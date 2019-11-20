import numpy as np
import fenics as fa
import mshr
from .. import arguments


class Graph(object):
    """Base class for Graph.
    """
    def __init__(self, args):
        self.args = args
        self.adjacency_list = self._adjacency_list_from_adjacency_matrix()
        self.ordered_adjacency_list = self._order_adjacency()
        self.triangle_area_sum = self._compute_area()
        self.weighted_area = self._get_weighted_area()
        self.gradient_x1 = self._assemble_gradient(self.x2, 0)
        self.gradient_x2 = self._assemble_gradient(self.x1, 1)
        self.reset_matrix_boundary = np.diag(self._boundary_flags)
        self.reset_matrix_interior = np.identity(self.num_vertices) - self.reset_matrix_boundary
        # print(self.coo)
        # print(self.ordered_adjacency_list)
        # print(self.triangle_area_sum)
        # print(self.gradient_x1)
        # print(self.reset_matrix_boundary)
        # print(self.reset_matrix_interior)

    def _assemble_gradient(self, x, flag):
        mult_1 = 1 if flag == 1 else -1
        mult_2 = -1 if flag == 1 else 1
        gradient_operator = np.zeros((self.num_vertices, self.num_vertices))
        for i in range(self.num_vertices):
            ordered_adjacency = self.ordered_adjacency_list[i]
            range_j = len(ordered_adjacency) - 1 if self._boundary_flags[i] == 1 else len(ordered_adjacency)
            for j in range(range_j):
                idx_crt = ordered_adjacency[j]
                idx_next = ordered_adjacency[(j + 1) % len(ordered_adjacency)]
                gradient_operator[i][idx_next] += mult_1/(2*self.triangle_area_sum[i])*(x[idx_crt] - x[i])
                gradient_operator[i][i] += mult_2/(2*self.triangle_area_sum[i])*(x[idx_crt] - x[i])
                gradient_operator[i][idx_crt] += mult_2/(2*self.triangle_area_sum[i])*(x[idx_next] - x[i])
                gradient_operator[i][i] += mult_1/(2*self.triangle_area_sum[i])*(x[idx_next] - x[i])
        return gradient_operator

    def _compute_area(self):
        area = np.zeros(self.num_vertices)
        for i in range(self.num_vertices):
            ordered_adjacency = self.ordered_adjacency_list[i]
            range_j = len(ordered_adjacency) - 1 if self._boundary_flags[i] == 1 else len(ordered_adjacency)
            for j in range(range_j):
                area[i] = area[i] + self._signed_tri_area(self.coo[i], 
                                                   self.coo[ordered_adjacency[j]], 
                                                   self.coo[ordered_adjacency[(j + 1) % len(ordered_adjacency)]])
        return area

    def _signed_tri_area(self, coo_0, coo_1, coo_2):
        return 0.5*((coo_2[1] - coo_0[1])*(coo_1[0] - coo_0[0]) 
                   -(coo_1[1] - coo_0[1])*(coo_2[0] - coo_0[0]))

    def _order_adjacency(self):
        ordered_adjacency_list = []
        for i, adjacency in enumerate(self.adjacency_list):
            ordered_adjacency = self._order_adjacency_substep(i, adjacency)
            ordered_adjacency_list.append(ordered_adjacency)
        return ordered_adjacency_list

    def _adjacency_list_from_adjacency_matrix(self):
        adjacency_list = []
        for i in range(self.num_vertices):
            adjacency_list.append([])
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] == 1:
                    adjacency_list[i].append(j)
        return adjacency_list

    def _order_adjacency_substep(self, center, adjacency):
        coo = [ [self.x1[i] - self.x1[center], self.x2[i] - self.x2[center]] for i in adjacency ]
        angles = [self._cart2pol(coo[i]) for i in range(len(coo))]
        dic = dict(zip(adjacency, angles))
        ordered_list = sorted(dic.items(), key=lambda x: x[1]) # sort by value, return a list of tuples
        ordered_adjacency = [ordered_list[i][0] for i in range(len(ordered_list))]
        ordered_adjacency = self._order_adjacency_subsubstep(center, ordered_adjacency)
        return ordered_adjacency

    def _cart2pol(self, coo):
        # helper function
        x = coo[0]
        y = coo[1]
        phi = np.arctan2(y, x) 
        phi = phi + 2*np.pi if phi < 0 else phi
        return phi


class GraphMSHR(Graph):
    """Graph obtained from mshr.
    """
    def __init__(self, args, mesh_parameters=None):
        # replaced by something like namedtuple
        # self.mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(1, 1), 3, 3) 
        self.mesh = mshr.generate_mesh(mshr.Circle(fa.Point(0, 0), 1), 10)
        self.coo = self.mesh.coordinates()
        self._cells = self._make_oriented(self.mesh.cells())
        self.num_vertices = self.mesh.num_vertices()
        self.adjacency_matrix = self._adjacency_matrix_from_cells()
        self.x1 = self.coo[:, 0]
        self.x2 = self.coo[:, 1]
        self._boundary_flags = self._get_boundary_flags()
        self.name = 'mshr' 
        super(GraphMSHR, self).__init__(args)

    def _get_weighted_area(self):
        return 1./3.*self.triangle_area_sum

    def _make_oriented(self, cells):
        for i, cell in enumerate(cells):
            if self._signed_tri_area(self.coo[cell[0]], 
                              self.coo[cell[1]],
                              self.coo[cell[2]]) < 0:
                tmp = cells[i][1]
                cells[i][1] = cells[i][2]
                cells[i][2] = tmp
        return cells

    def _get_boundary_flags(self):
        bmesh = fa.BoundaryMesh(self.mesh, "exterior", True)
        bmesh_coos = bmesh.coordinates()
        boundary_flags = np.zeros(self.num_vertices)
        for i in range(self.num_vertices):
            for bcoo in bmesh_coos:
                if np.linalg.norm(bcoo - self.coo[i]) < 1e-8:
                    boundary_flags[i] = 1
        return boundary_flags

    def _adjacency_matrix_from_cells(self):
        adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices))
        for cell in self._cells:
            for i in cell:
                for j in cell:
                    if i != j:
                        adjacency_matrix[i][j] = 1.
        return adjacency_matrix 

    def _order_adjacency_subsubstep(self, center, ordered_adjacency):
        # PDEs are made by god, yet boundary conditions are made by demon
        # Special care given to boundary nodes
        if self._boundary_flags[center] == 1:
            counter = 0
            for i, index in enumerate(ordered_adjacency):
                cell_t = np.array([center, 
                                   ordered_adjacency[i],
                                   ordered_adjacency[(i + 1) % len(ordered_adjacency)]])
                if not self._valid_cell(cell_t):
                    break
                counter = counter + 1
            assert(counter < len(ordered_adjacency))
            ordered_adjacency = ordered_adjacency[counter + 1:] + ordered_adjacency[:counter + 1]
        return ordered_adjacency

    def _order_cell(self, cell):
        return np.concatenate((cell[np.argmin(cell):], cell[:np.argmin(cell)]))

    def _valid_cell(self, cell_t):
        for cell in self._cells:
            if np.linalg.norm(self._order_cell(cell) - self._order_cell(cell_t)) < 1e-8:
                return True
        return False


class GraphManual(Graph):
    """Graph created by ourselves.
    """
    def __init__(self, args):
        self.num_ver_per_line = 15
        self.num_vertices =  self.num_ver_per_line**2
        self.adjacency_matrix = self._get_adjacency_matrix()
        self.coo = self._get_coo()
        self.x1 = self.coo[:, 0]
        self.x2 = self.coo[:, 1]
        self._boundary_flags = self._get_boundary_flags() 
        self.name = 'manual' 
        super(GraphManual, self).__init__(args)

    def _get_weighted_area(self):
        return 0.5*self.triangle_area_sum

    def _get_coo(self):
        total_length = 1.
        cell_length = total_length / (self.num_ver_per_line - 1)
        coo = np.zeros((self.num_vertices, 2))
        for i in range(self.num_vertices):
            coo[i][0] = (i % self.num_ver_per_line) * cell_length
            coo[i][1] = (i // self.num_ver_per_line) * cell_length
        return coo

    def _get_boundary_flags(self):
        boundary_flags = np.zeros(self.num_vertices)
        for i in range(self.num_vertices):
            if i < self.num_ver_per_line or \
               i >= self.num_vertices - self.num_ver_per_line or \
               i % self.num_ver_per_line == 0 or \
               (i + 1) % self.num_ver_per_line == 0:
                boundary_flags[i] = 1
        return boundary_flags

    def _get_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices))
        for i in range(self.num_vertices):
            if not i < self.num_ver_per_line:
                adjacency_matrix[i][i - self.num_ver_per_line] = 1.
            if not i >= self.num_vertices - self.num_ver_per_line:
                adjacency_matrix[i][i + self.num_ver_per_line] = 1.
            if not i % self.num_ver_per_line == 0:
                adjacency_matrix[i][i -1] = 1.
            if not (i + 1) % self.num_ver_per_line == 0:
                adjacency_matrix[i][i + 1] = 1.
        return adjacency_matrix

    def _order_adjacency_subsubstep(self, center, ordered_adjacency):
        # PDEs are made by god, yet boundary conditions are made by demon
        # Special care given to boundary nodes
        if self._boundary_flags[center] == 1:
            if center < self.num_ver_per_line and center != 0:
                counter = -1
            if center >= self.num_vertices - self.num_ver_per_line and center != self.num_vertices - 1:
                counter = 0
            if center % self.num_ver_per_line == 0 and center != self.num_vertices - self.num_ver_per_line:
                counter = 1
            if (center + 1) % self.num_ver_per_line == 0 and center != self.num_ver_per_line - 1:
                counter = -1
            ordered_adjacency = ordered_adjacency[counter + 1:] + ordered_adjacency[:counter + 1]
        return ordered_adjacency


if __name__ == '__main__':
    args = arguments.args

    graph_mshr = GraphMSHR(args)
    # graph_manual = GraphManual(args)

    file = fa.File(args.root_path + '/' + args.solutions_path + '/mesh.pvd')
    graph_mshr.mesh.rename('u', 'u')
    file << graph_mshr.mesh