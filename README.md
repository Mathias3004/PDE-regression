# PDE-regression
Regression to reconstruct effective partial differential equation from numerical data.

Here applied to numerical simulations of two-point correlators, using MPS library TeNPy (https://tenpy.github.io/).
The technique is applied to bosonic and fermionic lattice models (Bose(Fermi)-Hubbard model)

The datgen_boson(fermion).py is used to generate the data, it takes command-line arguments and generates a timeline using TEBD to obtain two-point correlators.
The analyze Jupyter notebooks are used to read in the data again perform the PDE regression and reconstruction (defined in functions_regression.py, functions_PDE_solver.py respectively)

A PDE is recovered that contains an effective rescaling of the coefficients found in a perturbative (analytical) approach.
