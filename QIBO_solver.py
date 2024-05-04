import numpy as np
import matplotlib.pyplot as plt
from qibo import callbacks, hamiltonians, models
from qibo.symbols import Z, X
from qibo import hamiltonians


class QKP_Qibo_solver:
    def __init__(self, W, wt, val):
        self.W = W # Weight capacity
        self.wt = wt # array containing the weights of each item
        self.val = val # array containing the profit of each pair of items
        self.N = len(wt) # total number of items

        def S(i): # spin variable
            return (1-Z(i))/2

        ham = - sum(self.val[i][j] * S(i) * S(j) for i in range(self.N) for j in range (i+1)) # values
        ham -= 0.9603*(self.W - sum(self.wt[i] * S(i) for i in range(self.N)))
        ham += 0.0371*(self.W - sum(self.wt[i] * S(i) for i in range(self.N)))**2

        self.h1 = hamiltonians.SymbolicHamiltonian(ham)

    def eigenvalues(self):
        return self.h1.eigenvalues()
    
    def ground_state(self):
        return self.h1.ground_state()
    
    def exact_solution(self):
        print('Exact solution has items: ', self.convert_state_to_items(self.h1.ground_state()))
    
    def ham_matrix(self):
        return self.h1.matrix

    def run_simulated_annealing(self, T = 50):
        '''
        T (float): Total time of the adiabatic evolution.
        '''
        
        ham = sum(X(i) for i in range(self.N))
        h0 = hamiltonians.SymbolicHamiltonian(ham)

        bac = self.h1.backend

        # Calculate target values (h1 ground state)
        target_state = self.h1.ground_state()
        target_energy = bac.to_numpy(self.h1.eigenvalues()[0]).real
        print('Target energy', target_energy)

        # Check ground state
        state_energy = bac.to_numpy(self.h1.expectation(target_state)).real
        np.testing.assert_allclose(state_energy.real, target_energy)

        '''
        dt (float): Time step used for integration.
        solver (str): Solver used for integration.
        '''
        dt = 1e-1
        solver = "exp"

        energy = callbacks.Energy(self.h1)
        overlap = callbacks.Overlap(target_state)
        evolution = models.AdiabaticEvolution(
            h0, self.h1, lambda t: t, dt=dt, solver=solver, callbacks=[energy, overlap]
        )
        self.final_psi = evolution(final_time=T)

        print('final annealing energy: ', energy[-1])

        # Plots
        tt = np.linspace(0, T, int(T / dt) + 1)
        plt.figure(figsize=(12, 4))
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

    def show_results(self):
        print('Solution has items: ', self.convert_state_to_items(self.final_psi))

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
            if b=='1':
                items.append(item)

        return items