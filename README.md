# Requirements 
- python 3.6.3
- mshr
- numpy
- scipy
- matplotlib
- jupyter
- pytorch

For Linux or Mac users:

    conda install python==3.6.3
    conda install -c conda-forge fenics
    conda install -c conda-forge mshr
    conda install pytorch torchvision cudatoolkit=$YOUR_TOOLKIT_VERSION -c pytorch
    conda install numpy scipy matplotlib jupyter

# Usage
To run a runner module, do:

    python -m src.ml.trainer

or similar.

# Dev
To define hyperparameters / arguments / defaults for a module, create a file `src/arguments/MODULENAME_args.py`, with a method add_args(parser) which adds argparse arguments to the parser. 

To access arguments for all modules:

    from src import arguments
    args = arguments.args

Modules use relative imports (e.g. to import fa from a module in src/, `from . import arguments`, to do it from a module in src/pde/, `from .. import arguments`, ..).

Generally follow pep8.

# Todo
- From linear problems to non-linear problems
- More complicated boundary conditions (both Dirirchlet and Neumann conditions imposed weakly in the form of penalty terms in loss functions). Can be input if needed.
- Test on a real plasma asccosicated problem
