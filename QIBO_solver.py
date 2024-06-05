from overrides import override
import numpy as np
import matplotlib.pyplot as plt
from qibo import callbacks, hamiltonians, models
from qibo.symbols import Z, X
from qibo import hamiltonians

from solver import Solver

class Qibo_annealing_solver(Solver):
    def __init__(self, W, wt, val):
        super().__init__(W, wt, val)

        # problem hamiltonian
        ham = sum(self.h_coeffs[term] * Z(term[0]) for term in np.ndindex(self.h_coeffs.shape)) # lineal terms
        ham += sum(self.J_coeffs[term] * Z(term[0]) * Z(term[1]) for term in np.ndindex(self.J_coeffs.shape)) # quadratic terms

        self.h1 = hamiltonians.SymbolicHamiltonian(ham)

        self.annealing_schedule = lambda t: t
        self.custom_time = False

    def get_ham(self):
        return self.h1

    def eigenvalues(self):
        return self.h1.eigenvalues()
    
    def ground_state(self):
        return self.h1.ground_state()
    
    def exact_solution(self):
        print('Exact solution has items: ', self.convert_state_to_items(self.h1.ground_state()))
    
    def ham_matrix(self):
        return self.h1.matrix
    
    def set_annealing_schedule(self, s):
        self.annealing_schedule = s
        self.custom_time = True

    @override
    def run(self, time = 50):
        '''
        time (float): Total time of the adiabatic evolution.
        '''
        
        ham = (-1/self.N)*sum(X(i) for i in range(self.N))
        h0 = hamiltonians.SymbolicHamiltonian(ham)

        bac = self.h1.backend

        # Calculate target values (h1 ground state)
        target_state = self.h1.ground_state()
        target_energy = bac.to_numpy(self.h1.eigenvalues()[0]).real
        print('Target energy', target_energy)
        print(f'+ offset ({self.offset}) = {target_energy + self.offset}')

        # Check ground state
        state_energy = bac.to_numpy(self.h1.expectation(target_state)).real
        np.testing.assert_allclose(state_energy.real, target_energy)

        '''
        dt (float): Time step used for integration.
        solver (str): Solver used for integration.
        '''
        dt = 1e-2
        solver = "exp"

        energy = callbacks.Energy(self.h1)
        overlap = callbacks.Overlap(target_state)
        evolution = models.AdiabaticEvolution(
            h0, self.h1, self.annealing_schedule, dt=dt, solver=solver, callbacks=[energy, overlap]
        )
        self.final_psi = evolution(final_time=time)
        self.solution_items = self.convert_state_to_items(self.final_psi)

        print('final annealing energy: ', energy[-1])

        # Plots
        tt = np.linspace(0, time, int(time / dt) + 1)
        plt.figure(figsize=(12, 4))
        if self.custom_time:
            plt.title('Custom time schedule')
        else:
            plt.title('Ordinary time schedule')
        plt.subplot(121)
        plt.plot(tt, energy[:], linewidth=2.0, label="Evolved state")
        plt.axhline(y=target_energy, color="red", linewidth=2.0, label="Ground state")
        plt.xlabel("$t$")
        plt.ylabel("$H_1$")
        plt.legend()

        plt.subplot(122)
        plt.plot(tt, overlap[:], linewidth=2.0)
        plt.xlabel("$t$")
        plt.ylabel("Overlap")

    def convert_state_to_items(self, state):
        max_i = -1
        max_amplitud = 0
        for i, amplitud in enumerate(state):
            if abs(amplitud) > max_amplitud:
                max_i = i
                max_amplitud = abs(amplitud)

        bin_solution = bin(max_i)
        bin_solution = str(bin(max_i))[2:] # remove '0b' start
        bin_solution = '0'*(self.N - len(bin_solution)) + bin_solution
        
        # the MSB is the item 0
        items = []
        for item, b in enumerate(bin_solution):
            if b=='0': # due to the encoding we are using (dimod)
                items.append(item)

        return items