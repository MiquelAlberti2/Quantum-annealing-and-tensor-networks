import dimod
from neal import SimulatedAnnealingSampler
from itertools import combinations
import numpy as np
from qibo.backends import matrices
from qibo.hamiltonians.hamiltonians import SymbolicHamiltonian
from qibo.hamiltonians.terms import HamiltonianTerm
from qibo.hamiltonians.models import multikron


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

        h_coeffs = [H_dict[0][site] for site in range(self.N)] # H[0] contains linear interactions

        J_coeffs = np.zeros((self.N,self.N))
        for term in H_dict[1]: # quadratic interacions
            # TODO in formulation i>j, but in MPO i<j (triangular superior, la meitat buida)
            J_coeffs[min(term[0], term[1]), max(term[0], term[1])] = H_dict[1][term]

        self.offset = H_dict[2]

        terms = []

        for i in range(self.N):
            terms.append(HamiltonianTerm(h_coeffs[i]*matrices.Z, i))

        m = multikron([matrices.Z, matrices.Z])

        for i in range(self.N):
            for j in range(i+1, self.N):
                terms.append(HamiltonianTerm(J_coeffs[i, j]*m, i, j))

        self.H_target = SymbolicHamiltonian()
        self.H_target.terms = terms

    def run_simulated_annealing_neal(self):
        sampleset = SimulatedAnnealingSampler().sample(self.model, num_reads=1000)
        self.items_in_sol = []
        for key in sampleset.first.sample:
            if sampleset.first.sample[key] == 1:
                self.items_in_sol.append(key)

        return self.items_in_sol
    
    def run_simulated_annealing_qibo(self):
        

        return self.items_in_sol

    def convert_to_MPO(self):
        pass

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

    def impose_penalty_to_state():
        pass