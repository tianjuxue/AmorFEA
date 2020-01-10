"""Poisson problem definition"""
import numpy as np
import fenics as fa


class Poisson(object):
    def __init__(self, args):
        self.args = args
        self._build_mesh()
        self._build_function_space()
        self._build_transformer()

    def _build_mesh(self):
        raise NotImplementedError()

    def _build_function_space(self):
        raise NotImplementedError()

    def _build_transformer(self):
        self.num_dofs = self.V.dim()
        self.num_vertices = self.mesh.num_vertices()
        self.v_d = fa.vertex_to_dof_map(self.V)
        self.d_v = fa.dof_to_vertex_map(self.V)
        self.coo_dof = self.V.tabulate_dof_coordinates()
        self.coo_ver = self.mesh.coordinates()
        self.cells = self.mesh.cells()  
        self._set_boundary_flags()
        self._set_detailed_boundary_flags()

        # print(self.num_dofs)
        # print(self.num_vertices)
        # print(self.coo_dof)
        # print(self.coo_ver)

        # for cell_no in range(self.mesh.num_cells()):
        #     dofs = self.V.dofmap().cell_dofs(cell_no)
        #     print(dofs)

    def _set_boundary_flags(self):
        bmesh = fa.BoundaryMesh(self.mesh, "exterior", True)
        bmesh_coos = bmesh.coordinates()
        boundary_flags = np.zeros(self.num_vertices)
        for i in range(self.num_vertices):
            for bcoo in bmesh_coos:
                if np.linalg.norm(bcoo - self.coo_dof[i]) < 1e-8:
                    boundary_flags[i] = 1
        self.boundary_flags = boundary_flags

    def _tri_area(self, coo_0, coo_1, coo_2):
        return np.absolute(0.5*((coo_2[1] - coo_0[1])*(coo_1[0] - coo_0[0]) 
                          -(coo_1[1] - coo_0[1])*(coo_2[0] - coo_0[0])))

    def _cell_area(self, cell):
        return self._tri_area(self.coo_ver[cell[0]], self.coo_ver[cell[1]], self.coo_ver[cell[2]])

    def get_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.num_dofs, self.num_dofs))
        for cell in self.cells:
            for i in cell:
                for j in cell:
                    if i != j:
                        adjacency_matrix[self.v_d[i]][self.v_d[j]] = 1.
        return adjacency_matrix

    def get_adjacency_list(self):
        adjacency_matrix = self.get_adjacency_matrix()
        adjacency_list = []
        for i in range(self.num_vertices):
            adjacency_list.append([])
            for j in range(self.num_vertices):
                if adjacency_matrix[i][j] == 1:
                    adjacency_list[i].append(j)
        return adjacency_list

    def get_weight_area(self):
        weight_area = np.zeros(self.num_vertices)
        for index in range(self.num_vertices):
            for cell in self.cells:
                for i in cell:
                    if self.d_v[index] == i:
                        weight_area[index] += self._cell_area(cell)
        return 1./3.*weight_area
