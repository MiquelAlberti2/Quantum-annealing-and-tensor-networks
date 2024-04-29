
# This code is contributed by Suyash Saxena
# https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/
# Modificiation 1: keep track of the items in the final solution
# Modification 2: the value matrix is a triangular matrix, to solve QKP
from itertools import combinations

class Dynamic_Programming_QKP_solver:
	def __init__(self, W, wt, val):
		self.W = W # Weight capacity
		self.wt = wt # array containing the weights of each item
		self.val = val # array containing the profit of each pair of items
		self.N = len(wt) # total number of items

		self.items_in_sol = None

	def run(self): 

		solutions_dict = {} # to keep track of items in the solution for each weight
		for w in range(self.W+1):
			solutions_dict[w] = []
		
		# Create array that carries the value of the current solution for each weight 
		dp = [0 for i in range(self.W+1)] 

		# Taking first i elements 
		for i in range(self.N): 
			
			# Starting from back, so that we also have data of 
			# previous computation when taking i-1 items 
			for w in range(self.W, 0, -1): 

				if self.wt[i] <= w: # if the item fits...
					# compute the value of the solution if we take the item
					new_value = dp[w-self.wt[i]] + self.val[i][i]

					for j in solutions_dict[w-self.wt[i]]:
						new_value += self.val[max(i,j)][min(i,j)] # val is a triangular matrix

					# check if the item improves the solution
					if dp[w] < new_value:
						# update the value
						dp[w] = new_value
						# update the items in the solution
						solutions_dict[w] = solutions_dict[w-self.wt[i]].copy()
						solutions_dict[w].append(i)
		
		self.items_in_sol = solutions_dict[self.W]
		return self.items_in_sol
	
	def show_results(self):
		if not self.items_in_sol:
			raise Exception('Call run method first')
		
		print('Solution has items: ', self.items_in_sol)

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

		

		

  
  

		