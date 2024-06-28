## Quantum Annealing and Tensor Networks: a Powerful Combination to Solve Optimization Problems

![Overview](https://github.com/MiquelAlberti2/Quantum-annealing-and-tensor-networks/blob/aebd7ed56cd7f887a3126a6a75ad3bc51c9d39aa/Visualizations/project_overview.png)

This repository contains the code for my Bachelor's thesis, which explores the application of quantum annealing and tensor networks to solve the Quadratic Knapsack Problem (QKP). For detailed information about the project, refer to the main thesis document `thesis.pdf`; for a broader overview, see `thesis_presentation.pdf`.

### Simplified UML Diagram

![UML Diagram](https://github.com/MiquelAlberti2/Quantum-annealing-and-tensor-networks/blob/fbd2cd590756a2154817d6405cd636d452433619/Visualizations/UML_diagram.png)

### Solvers

This project implements several solvers for the QKP, all inheriting from the base class `solver.py` which defines the problem formulation. 

- **Simulated Annealing:**
    - `qibo_solver.py`: Implements simulated annealing using the Qibo library.
    - `neal_solver.py`: Implements simulated annealing using the Neal library.
- **Dynamic Programming:**
    - `dp_solver.py`: Solves the QKP using dynamic programming.
- **Tensor Networks:**
    - `dmrg_solver.py`: Solves the QKP using the Density Matrix Renormalization Group (DMRG) algorithm.
- **Exact Solver:**
    - `exact_solver.py`: Implements an exact solver using brute force and exact diagonalization for testing purposes.

### Notebooks

- `solutions_enhanced_annealing.ipynb`: Demonstrates how to use the DMRG to estimate the gap of the annealing evolution for creating a custom annealing schedule, followed by testing with Qibo's solver.
- `solutions_large_instances.ipynb`: Provides examples of solving large QKP instances using dynamic programming and Neal's solver.

### Other Files

- `dmrg_hyperparameter_tuning.py`: Script for finding optimal DMRG parameters for a specific QKP instance.
- **DMRG/** folder: Contains subroutines used by `dmrg_solver.py`:
    - `annealing_ham_to_mpo.py`: Creates the MPO for the annealing Hamiltonian of an Ising model.
    - `module.py`: Implements the DMRG algorithm.
    - `module_penalty.py`: Variation of the DMRG for computing the first excited state.
    - `samplingMPS.py`: Class that efficiently samples the state represented by a given MPS.
- **Tests/** folder:
    - Contains methods for defining QKP test instances. 
    - Stores results from `solutions_large_instances.ipynb`.
- **Visualizations/** folder:
    - Contains notebooks used for understanding the algorithms.
    - `symbolic_X_ZZ_interactions.ipynb` and `symbolic_annealing_ham.ipynb` were used to develop the Matrix Product Operator (MPO) for the annealing Hamiltonian in `DMRG/annealing_ham_to_mpo.py`
