from abc import ABC, abstractmethod
from itertools import combinations
import dimod
import numpy as np

class Solver(ABC):
    def __init__(self, W, wt, val):
        self.W = W # Weight capacity
        self.wt = wt # array containing the weights of each item
        self.val = val # array containing the profit of each pair of items
        self.N = len(wt) # total number of items

        a = 3 # parameter to play with if solutions are not satisfying the contraint

        self.mult = a*np.log2(self.N*self.W) # the bigger this factor, the greater the penalization for overweight candidates,
                                            # but also the bigger the penalization for light weight candidates
                                            # and the bigger the bonus for valid candidates close to the limit
                 
        # Define cost function with binary variables
        x = [dimod.Binary(i) for i in range(self.N)]

        cost = - dimod.quicksum(self.val[i][j] * x[i] * x[j] for i in range(self.N) for j in range (i+1)) # values
        cost -= self.mult*0.9603*(W - dimod.quicksum(wt[i] * x[i] for i in range(self.N)))
        cost += self.mult*0.0371*(W - dimod.quicksum(wt[i] * x[i] for i in range(self.N)))**2

        self.model = dimod.BinaryQuadraticModel(cost.linear, cost.quadratic, cost.offset, vartype='BINARY')

        # https://docs.ocean.dwavesys.com/en/latest/docs_dimod/reference/generated/dimod.binary.BinaryQuadraticModel.to_ising.html
        # Convert to Ising model (spin variables)
        self.H_dict = self.model.to_ising()

        max_coeff = float('-inf') # normalizing factor

        # H_dict[0] contains linear interactions
        self.h_coeffs = np.zeros((self.N))
        for site in self.H_dict[0]:
            coeff = self.H_dict[0][site]
            self.h_coeffs[site] = coeff

            if abs(coeff) > max_coeff: 
                max_coeff = coeff
        
        # H_dict[1] contains quadratic interactions
        self.J_coeffs = np.zeros((self.N,self.N))
        for term in self.H_dict[1]:
            # In formulation i>j, but in MPO i<j
            # Future work: use the same convention
            coeff = self.H_dict[1][term]
            self.J_coeffs[min(term[0], term[1]), max(term[0], term[1])] = coeff

            if abs(coeff) > max_coeff: 
                max_coeff = coeff

        # normalize the Ising Hamiltonian
        self.h_coeffs = self.h_coeffs / max_coeff
        self.J_coeffs = self.J_coeffs / max_coeff

        self.offset = self.H_dict[2]

        self.solution_items = None

    @abstractmethod
    def run(self, time):
        '''
        Method to be implemented by the childs of this class
        It should compute the solution of the given QKP 
        '''
        pass

    def show_solution(self):
        if self.solution_items is None:
            raise Exception('Call run method first')

        print('-------- Solution has items: ', self.solution_items, '--------')
        solution_dict = self.stats_of_items(self.solution_items)
        print('-------------------------------------------------')

        return solution_dict


    def stats_of_items(self, items):
        print(' - Evaluating candidate ', items)
        value, weight, energy = self.evaluate_items(items)

        print(f'Profit: {value}')
        if weight <= self.W:
            print(f'Weight: {weight} (satisfies constraint W={self.W})')
        else:
            print(f'Weight: {weight} (does NOT satisfy constraint W={self.W})')
        print(f'Energy: {energy}')

        solution_dict = {'N': self.N,
                         'W': self.W,
                         'profit': value,
                         'weight': weight,
                         'energy': energy}

        return solution_dict

    def evaluate_items(self, items):
        '''
        Given a set of items of the QKP,
        returns its profit, weight and energy of the QUBO formulation
        '''
        value = 0
        weight = 0

        for i in items:
            value += self.val[i][i]
            weight += self.wt[i]

        pairs = list(combinations(items,2))
        for p in pairs:
            value += self.val[max(p[0], p[1])][min(p[0], p[1])]

        energy = - value - self.mult*0.9603*(self.W - weight) + self.mult*0.0371*(self.W - weight)**2
        return value, weight, energy