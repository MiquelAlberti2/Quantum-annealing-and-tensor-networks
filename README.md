## QUANTUM ANNEALING AND TENSOR NETWORKS: A POWERFUL COMBINATION TO SOLVE OPTIMIZATION PROBLEMS

This repository contains the code for my Bachelor's thesis, which explores the application of quantum annealing and tensor networks to solve the Quadratic Knapsack Problem (QKP). Review the memory thesis `memory.pdf` for further information about the project.

### Solvers

This project implements several solvers for the QKP, all inheriting from the base class `Solver.py` which defines the problem formulation. 

- **Simulated Annealing:**
    - `qibo_solver.py`: Implements simulated annealing using the Qibo library.
    - `neal_solver.py`: Implements simulated annealing using the Neal library.
- **Dynamic Programming:**
    - `dp_solver.py`: Solves the QKP using dynamic programming.
- **Tensor Networks:**
    - `dmrg_solver.py`: Solves the QKP using the Density Matrix Renormalization Group (DMRG) algorithm with tensor networks.
- **Exact Solver:**
    - `exact_solver.py`: Implements an exact solver using brute force and exact diagonalization for testing purposes.

### Notebooks

- `solutions_enhanced_annealing.ipynb`: Demonstrates how to use DMRG to estimate the gap of the annealing evolution for creating a custom annealing schedule, followed by testing with Qibo's solver.
- `solutions_large_instances.ipynb`: Provides examples of solving large QKP instances using dynamic programming and Neal's simulated annealing.

### Other Files

- `dmrg_hyperparameter_tuning.py`: Script for finding optimal DMRG parameters for a specific QKP instance.
- **DMRG/** folder: Contains subroutines used by `DMRG_solver.py`:
    - `annealing_ham_to_mpo.py`: Creates the MPO for the Ising model's annealing Hamiltonian.
    - `module.py`: Implements the DMRG algorithm.
    - `module_penalty.py`: Variation of DMRG for computing the first excited state.
    - `samplingMPS.py`: Class that efficiently samples the state represented by a given MPS.
- **tests/** folder:
    - Contains methods for defining QKP test instances. 
    - Stores results from `solutions_large_instances.ipynb`.
- **Visualizations/** folder:
    - Contains Jupyter notebooks used for understanding the algorithms.
    - `symbolic_X_ZZ_interactions.ipynb` and `symbolic_annealing_ham.ipynb` were used to create the Matrix Product Operator (MPO) for the annealing Hamiltonian in `DMRG/annealing_ham_to_mpo.py`
