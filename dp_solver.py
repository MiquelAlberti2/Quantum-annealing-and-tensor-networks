from overrides import override
from solver import Solver

class DP_solver(Solver):
	def __init__(self, W, wt, val):
		# assumes integer values
		super().__init__(W, wt, val)

	@override
	def run(self, time = 0): 
		'''
		Solves the QKP using dynammic programming. The optimal solution is not guaranteed.

        This Solver does not use time
        '''

		solutions_dict = {} # to keep track of items in the solution for each weight
		for w in range(self.W+1):
			solutions_dict[w] = []
		
		# Create array that carries the value of the current solution for each weight 
		V = [0 for i in range(self.W+1)] 

		# Taking first i elements 
		for i in range(self.N): 
			
			# Starting from back, so that we also have data of 
			# previous computation when taking i-1 items 
			for w in range(self.W, 0, -1): 

				if self.wt[i] <= w: # if the item fits...
					# compute the value of the solution if we take the item
					new_value = V[w-self.wt[i]] + self.val[i][i]

					for j in solutions_dict[w-self.wt[i]]:
						new_value += self.val[max(i,j)][min(i,j)] # val is a triangular matrix

					# check if the item improves the solution
					if V[w] < new_value:
						# update the value
						V[w] = new_value
						# update the items in the solution
						solutions_dict[w] = solutions_dict[w-self.wt[i]].copy()
						solutions_dict[w].append(i)
		
		self.solution_items = solutions_dict[self.W]
		

		

  
  

		