import dimod
from neal import SimulatedAnnealingSampler
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from qibo.backends import matrices
from qibo.hamiltonians.hamiltonians import Hamiltonian, SymbolicHamiltonian, AbstractHamiltonian
from qibo.hamiltonians.terms import HamiltonianTerm
from qibo.hamiltonians.models import multikron
from qibo import callbacks, hamiltonians, models


class QKP_Hamiltonian:
    def __init__(self, W, wt, val):
        self.W = W # Weight capacity
        self.wt = wt # array containing the weights of each item
        self.val = val # array containing the profit of each pair of items
        self.N = len(wt) # total number of items

        self.items_in_sol = None

        # Define cost function with binary variables
        x = [dimod.Binary(i) for i in range(self.N)]

        cost = - dimod.quicksum(val[i][j] * x[i] * x[j] for i in range(self.N) for j in range (i+1)) # values
        cost -= 0.9603*(W - dimod.quicksum(wt[i] * x[i] for i in range(self.N)))
        cost += 0.0371*(W - dimod.quicksum(wt[i] * x[i] for i in range(self.N)))**2

        # Convert to Ising model (spin variables)
        self.model = dimod.BinaryQuadraticModel(cost.linear, cost.quadratic, cost.offset, vartype='BINARY')
        H_dict = self.model.to_ising()

        self.h_coeffs = [H_dict[0][site] for site in range(self.N)] # H[0] contains linear interactions

        self.J_coeffs = np.zeros((self.N,self.N))
        for term in H_dict[1]: # quadratic interacions
            # TODO in formulation i>j, but in MPO i<j (triangular superior, la meitat buida)
            self.J_coeffs[min(term[0], term[1]), max(term[0], term[1])] = H_dict[1][term]

        self.offset = H_dict[2]

        print('offset: ', self.offset)

    def build_Hamiltonian(self):
        Id = np.array([[1,0],[0,1]])
        Z = np.array([[1,0],[0,-1]])

        ham = np.zeros((2**self.N, 2**self.N))

        for i in range(self.N):
            ham += self.h_coeffs[i]*multikron(Z if k==i else Id for k in range(self.N))

        for i in range(self.N):
            for j in range(i+1, self.N):
                ham += self.J_coeffs[i,j]*multikron(Z if (k==i or k==j) else Id for k in range(self.N))

        self.H_target = Hamiltonian(self.N, ham)

    def run_simulated_annealing_neal(self):
        sampleset = SimulatedAnnealingSampler().sample(self.model, num_reads=1000)
        self.items_in_sol = []
        for key in sampleset.first.sample:
            if sampleset.first.sample[key] == 1:
                self.items_in_sol.append(key)

        return self.items_in_sol
    
    def get_ham_coeffs(self):
        return self.h_coeffs, self.J_coeffs, self.offset
    
    def eigenvalues(self):
        return self.H_target.eigenvalues()
    
    def ham_matrix(self):
        return self.H_target.matrix
    
    def run_simulated_annealing_qibo(self, T = 50):
        '''
        T (float): Total time of the adiabatic evolution.
        '''
        
        H_init = hamiltonians.X(self.N)
        bac = self.H_target.backend

        # Calculate target values (H_target ground state)
        target_state = self.H_target.ground_state()
        target_energy = bac.to_numpy(self.H_target.eigenvalues()[0]).real
        print('Target energy', target_energy)
        print('Target energy with offset', target_energy + self.offset)

        # Check ground state
        state_energy = bac.to_numpy(self.H_target.expectation(target_state)).real
        np.testing.assert_allclose(state_energy.real, target_energy)

        '''
        dt (float): Time step used for integration.
        solver (str): Solver used for integration.
        '''
        dt = 1e-1
        solver = "exp"

        energy = callbacks.Energy(self.H_target)
        overlap = callbacks.Overlap(target_state)
        evolution = models.AdiabaticEvolution(
            H_init, self.H_target, lambda t: t, dt=dt, solver=solver, callbacks=[energy, overlap]
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
        if not self.items_in_sol:
            raise Exception('Call run method first')

        print(' - Solution has items: ', self.items_in_sol)

        value = 0
        weight = 0

        for i in self.items_in_sol:
            value += self.val[i][i]
            weight += self.wt[i]

        pairs = list(combinations(self.items_in_sol,2))
        for p in pairs:
            value += self.val[max(p[0], p[1])][min(p[0], p[1])]

        energy = - value - 0.9603*(self.W - weight) + 0.0371*(self.W - weight)**2

        print(f'Profit: {value}')
        if weight <= self.W:
            print(f'Weight: {weight} (satisfies constraint W={self.W})')
        else:
            print(f'Weight: {weight} (does NOT satisfy constraint W={self.W})')
        print(f'Energy: {energy}')

    def show_qibo_results(self):
        print('final state: ', self.final_psi)

        max_i = -1
        max_amplitud = 0
        for i, amplitud in enumerate(self.final_psi):
            if abs(amplitud) > max_amplitud:
                max_i = i
                max_amplitud = abs(amplitud)

        print('most probable candidate: ', bin(max_i))

    def energy_of_items(self, items):
        print(' - Evaluating candidate ', items)
        value = 0
        weight = 0

        for i in items:
            value += self.val[i][i]
            weight += self.wt[i]

        pairs = list(combinations(items,2))
        for p in pairs:
            value += self.val[max(p[0], p[1])][min(p[0], p[1])]

        energy = - value - 0.9603*(self.W - weight) + 0.0371*(self.W - weight)**2

        print(f'Profit: {value}')
        if weight <= self.W:
            print(f'Weight: {weight} (satisfies constraint W={self.W})')
        else:
            print(f'Weight: {weight} (does NOT satisfy constraint W={self.W})')
        print(f'Energy: {energy}')