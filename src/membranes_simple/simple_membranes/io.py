#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:42:40 2019

@author: alexanderniewiarowski
"""

import os
import dolfin


# Default values, can be changed in the input file
XDMF_WRITE_INTERVAL = 0

class InputOutputHandling:
    def __init__(self, membrane):
        """
        This class handles reading and writing the simulation state such as
        velocity and presure fields. Files for postprocessing (xdmf) are also
        handled here
        """
        self.membrane = mem = membrane
        self.ready = False


        # Initialise the main plot file types
        self.xdmf = XDMFFileIO(membrane)

        # Set up periodic output of plot files. Other parts of Ocellaris may
        # also add their plot writers to this list
        self._plotters = []
        self.add_plotter(
            self.xdmf.write, 'output/xdmf_write_interval', XDMF_WRITE_INTERVAL
        )

        def close(success):
            return self._close_files()  # @UnusedVariable - must be named success

        # When exiting due to a signal kill/shutdown a savepoint file will be
        # written instead of an endpoint file, which is the default. This helps
        # with automatic restarts from checkpointing jobs. The only difference
        # is the name of the file
        self.last_savepoint_is_checkpoint = False

        # Last savepoint written. To be deleted after writing new save point
        # if only keeping the last save point file
        self.prev_savepoint_file_name = ''

    def setup(self):
        mem = self.membrane

#        # Make sure functions have nice names for output
        for name, description in (
            ('gamma', 'Parametric mapping'),
            ('n', 'Surface unit normals'),
            ('u', 'Displacement'),
            ('x', 'Position'),
            ('p_ext', 'External pressure'),
            ('l1', 'lambda1'),
            ('l2', 'lambda2'),
            ('l3', 'lambda3'),
            ('s11', 'sigma_11'),
            ('s22', 'sigma_22')
        ):
            if name not in mem.data:
                continue
            func = mem.data[name]
            if hasattr(func, 'rename'):
                func.rename(name, description)
        self.ready = True

    def add_plotter(self, func, interval_inp_key, default_interval):
        """
        Add a plotting function which produces IO output every timestep
        """
        self._plotters.append((func, interval_inp_key, default_interval))

    def add_extra_output_function(self, function):
        """
        The output files (XDMF) normally only contain u, p and potentially rho or c. Other
        custom fields can be added
        """
        self.xdmf.extra_functions.append(function)


    def _close_files(self):
        """
        Save final restart file and close open files
        """
        if not self.ready:
            return

        mem = self.membrane
        self.xdmf.close()

    def write_fields(self):
        """
        Write fields to file after end of time step
        """
        mem = self.membrane

        # Call the output functions at the right intervals
        for func, interval_inp_key, default_interval in self._plotters:
            # Check this every timestep, it might change
            write_interval = 1

            # Write plot file this time step if it aligns with the interval
            if write_interval > 0:
                func()


# Default values, can be changed in the input file
XDMF_FLUSH = True


class XDMFFileIO:
    def __init__(self, membrane):
        """
        Xdmf output using dolfin.XDMFFile
        """
        self.membrane = membrane
        self.extra_functions = []
        self.xdmf_file = None

    def close(self):
        """
        Close any open xdmf file
        """
        if self.xdmf_file is not None:
            self.xdmf_file.close()
        self.xdmf_file = None

    def write(self):
        """
        Write a file that can be used for visualization. The fields will
        be automatically downgraded (interpolated) into something that
        dolfin.XDMFFile can write, typically linear CG elements.
        """
        with dolfin.Timer('Save xdmf'):
            if self.xdmf_file is None:
                self._setup_xdmf()
            self._write_xdmf()
        return self.file_name

    def _setup_xdmf(self):
        """
        Create XDMF file object
        """
        mem = self.membrane
        xdmf_flush = XDMF_FLUSH

#        fn = sim.input.get_output_file_path('output/xdmf_file_name', '.xdmf')
        fn = mem.output_file_path
        file_name = get_xdmf_file_name(mem, fn)

        print('Creating XDMF file %s' % file_name)
        comm = mem.mesh.mpi_comm()
        self.file_name = file_name
        self.xdmf_file = dolfin.XDMFFile(comm, file_name)
        self.xdmf_file.parameters['flush_output'] = xdmf_flush
        self.xdmf_file.parameters['rewrite_function_mesh'] = False
        self.xdmf_file.parameters['functions_share_mesh'] = True
        self.xdmf_first_output = True



#        def create_vec_func(V):
#            "Create a vector function from the components"
#            family = V.ufl_element().family()
#            degree = V.ufl_element().degree()
#            cd = mem.data['constrained_domain']
#            V_vec = dolfin.VectorFunctionSpace(
#                sim.data['mesh'], family, degree, constrained_domain=cd
#            )
#            vec_func = dolfin.Function(V_vec)
#            assigner = dolfin.FunctionAssigner(V_vec, [V] * mem.ndim)
#            return vec_func, assigner

        # XDMF cannot save functions given as "as_vector(list)"
#        self._vel_func, self._vel_func_assigner = create_vec_func(sim.data['Vu'])
#        self._vel_func.rename('u', 'Velocity')

    def _write_xdmf(self):
        """
        Write plot files for Paraview and similar applications
        """
        mem = self.membrane

        if self.xdmf_first_output:
            self.t = 0

            # Construct the identity map used for plotting only
            if mem.nsd==2:
                self.idn = dolfin.Expression(('x[0]','0'), degree=1)
            else:
                self.idn = dolfin.Expression(('x[0]','x[1]','0'), degree=1)

            self.delta_gammah =  dolfin.interpolate(mem.gamma, mem.V)

            # Get the displacement from the parametric to the reference for plotting
            self.gamma_paraview = dolfin.Function(mem.V)
            self.gamma_paraview.rename('gamma','Parametric mapping')
            self.gamma_paraview.vector()[:] = self.delta_gammah.vector()[:] - dolfin.interpolate(self.idn, mem.V).vector()[:]
            self.xdmf_file.write(self.gamma_paraview, 0)

            self.gamma_paraview_DG =  dolfin.interpolate(self.idn, mem.W)

            self.uh = dolfin.Function(mem.V)
            self.uh.rename('u_vector','Displacement')
#            self.W = dolfin.TensorFunctionSpace(mem.mesh, "CG", 2, shape=(3, 3))

        t = self.t

        # Write the shifted displacements for Paraview "warp by vector"
        # TODO: should uh be in mem.data?
        self.uh.vector()[:] = self.gamma_paraview.vector()[:] + mem.u.vector()[:]
        self.xdmf_file.write(self.uh, t)

        # Assign stretches to functions l1, l2 (lambda is a dolfin.Form, l is dolfin.Function)
        mem.l1.assign(dolfin.project(mem.lambda1, mem.Z))
        mem.l2.assign(dolfin.project(mem.lambda2, mem.Z))
        mem.l3.assign(dolfin.project(mem.lambda3, mem.Z))

        # Assign the unit normals
        mem.normals.assign(dolfin.project(mem.n, mem.W))

        if hasattr(mem, "free_surface"):
            if mem.free_surface:
                dolfin.project(mem.fs, mem.W, function=mem.data["free_srf"])
                dolfin.project(mem.c, mem.Z, function=mem.data["c"])

#        if mem.nsd==3:  #FIXME: write stresses for 2D case too!
#            # project sigma into g1 and g2 directions - # FIXME, experimental
#            s11 = dolfin.dot(mem.gsub1, dolfin.dot(mem.cauchy, mem.gsub1))
#            mem.s11.assign(dolfin.project(s11, mem.Vs))
#            s22 = dolfin.dot(mem.gsub2, dolfin.dot(mem.cauchy, mem.gsub2))
#            mem.s22.assign(dolfin.project(s22, mem.Vs))
#            self.xdmf_file.write(mem.s11, t)
#            self.xdmf_file.write(mem.s22, t)

#        Stress = dolfin.project(self.cauchy, self.W)
#        S1 = dolfin.project(Stress[0,0], mem.Vs)
#        S1.rename('S1_11', 'S1_11')
#        self.xdmf_file.write(S1, t)

#        S2 = dolfin.project(Stress[1,1], mem.Vs)
#        S2.rename('S2_22', 'S2_22')
#        self.xdmf_file.write(S2, t)

        # Write default functions
        for name in ('u', 'p_ext', 'n', 'l1', 'l2', 'l3'):
            if name in mem.data:
                func = mem.data[name]
                if isinstance(func, dolfin.Function):
                    self.xdmf_file.write(func, t)

        # Write extra functions
        for func in self.extra_functions:
            self.xdmf_file.write(func, t)

        self.xdmf_first_output = False
        self.t+=1

def get_xdmf_file_name(membrane, file_name_suggestion):
    """
    Deletes any previous files with the same name unless
    the simulation is restarted. If will then return a new
    name to avoid overwriting the existing files
    """
    base = os.path.splitext(file_name_suggestion)[0]

    file_name = base + '.xdmf'
    file_name2 = base + '.h5'

    # Remove any existing files with the same base file name
    if os.path.isfile(file_name):
        os.remove(file_name)
    if os.path.isfile(file_name2):
        os.remove(file_name2)

    return file_name
