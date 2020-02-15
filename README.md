# Requirements 
- python 3.7.3
- numpy
- scipy
- matplotlib
- fenics
- dolfin-adjoint
- mshr
- pytorch

We recommend using `conda`

    conda install python==3.6.3
    conda install numpy scipy matplotlib
    conda install -c conda-forge fenics
    conda install -c conda-forge dolfin-adjoint
    conda install -c conda-forge mshr
    conda install pytorch torchvision cudatoolkit=$YOUR_TOOLKIT_VERSION -c pytorch

# Usage
To run a runner module, do:

    python -m src.ml.trainer_dolfin

or similar. The package currently contains three examples (_linear_, _dolfin_ and _robot_) and module names should be indicative.

# Descriptions
The following descriptions use _dolfin_ example for instructions.

To compute traditional finite element analysis (FEA) solutions, do:

```
python -m src.pde.poisson_dolfin
```

To draw i.i.d. data for control parameters, do:

```
python -m src.ml.generator
```

To train a neural network model using amortized finite element analysis (AmorFEA), do:

```
python -m src.ml.trainer_dolfin
```

To perform PDE-constrained optimization with AmorFEA enabled surrogate model, do:

```
python -m src.opt.optimizer_dolfin
```

To visualize results, do:

```
python -m src.vis.demo_dolfin
```

# Dev

To define hyperparameters / arguments / defaults for a module, create a file `src/arguments/MODULENAME_args.py`, with a method `add_args(parser)` which adds argparse arguments to the parser. 

To access arguments for all modules:

```
from src import arguments
args = arguments.args
```

Modules use relative imports (e.g. to import `arguments` from a module in `src/`, `from . import arguments`, to do it from a module in `src/pde/`,  `from .. import arguments`, ...).

Generally follow PEP8.