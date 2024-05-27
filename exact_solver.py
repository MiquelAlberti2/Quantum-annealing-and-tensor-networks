from overrides import override
from solver import Solver
from qibo import callbacks, hamiltonians, models
from qibo.symbols import Z, X
from qibo import hamiltonians

import numpy as np
import matplotlib.pyplot as plt
import itertools

class Exact_solver(Solver):
	def __init__(self, W, wt, val):
		super().__init__(W, wt, val)

		# initial hamiltonian
		ham = (-1)*sum(X(i) for i in range(self.N))
		h0 = hamiltonians.SymbolicHamiltonian(ham)
		self.h0_matrix = h0.matrix

		# problem hamiltonian
		ham = sum(self.h_coeffs[term] * Z(term[0]) for term in np.ndindex(self.h_coeffs.shape)) # lineal terms
		ham += sum(self.J_coeffs[term] * Z(term[0]) * Z(term[1]) for term in np.ndindex(self.J_coeffs.shape)) # quadratic terms

		h1 = hamiltonians.SymbolicHamiltonian(ham)
		self.h1_matrix = h1.matrix

		self.target_ham = self.h1_matrix

	def evaluate_combination(self, combination):
		items = []

		for i, decision in enumerate(combination):
			if decision == 1: 
				items.append(i)

		return self.evaluate_items(items)

	@override
	def run(self, time = 0):
		best_feasible_profit = float('-inf')
		minimum_energy = float('inf')

		# Generate all possible combinations of binary decision variables
		for combination in itertools.product([0, 1], repeat=self.N):
			profit, weight, energy = self.evaluate_combination(combination)

			if energy < minimum_energy:
				minimum_energy = energy
			if weight < self.W and profit > best_feasible_profit:
				best_feasible_profit = profit
					
		return best_feasible_profit, minimum_energy


	def compute_target_gap(self):
		eigenvalues, eigenvectors = np.linalg.eig(self.target_ham)
		gap = sorted(eigenvalues)[:2]
		return gap
	
	def build_annealing_ham_time_s(self, s):
		self.target_ham = (1-s)*self.h0_matrix + s*self.h1_matrix
	
	def annealing_run(self, step = 10):
		E_0 = []
		E_1 = []

		for s in range(step+1):
			self.build_annealing_ham_time_s(s / step)
			gap = self.compute_target_gap()
			print(f's = {s} : {gap}')
			E_0.append(gap[0])
			E_1.append(gap[1])

		plt.figure()
		plt.plot(E_0, label='Ground energies', marker='o')
		plt.plot(E_1, label='First excited energies', marker='x')

		plt.title('Exact annealing run')
		plt.xlabel('step')
		plt.ylabel('Energy')
		plt.legend()
		plt.show()

		# plot gap
		plt.figure()
		plt.plot([E_1[i] - E_0[i] for i in range(len(E_0))], label='gap')

		plt.title('Exact gap')
		plt.xlabel('step')
		plt.ylabel('gap')
		plt.legend()
		plt.show()

		# compute estimated minimum gap
		gap = E_1[0] - E_0[0]
		for i in range(1, len(E_0)):
			aux = E_1[i] - E_0[i]
			if aux < gap:
				gap = aux

		print('estimated minimum gap: ', gap)
		
		self.target_ham = self.h1_matrix
