
# This code is contributed by Suyash Saxena
# https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/
# Modificiation 1: keep track of the items in the final solution
# Modification 2: the value matrix is a triangular matrix, to solve QKP


def QKP(W, wt, val, n): 

	items_in_sol = {} # to keep track of items in the solution for each weight
	for w in range(W+1):
		items_in_sol[w] = []
	
	# Making the dp array 
	dp = [0 for i in range(W+1)] 

	# Taking first i elements 
	for i in range(n): 
		
		# Starting from back, so that we also have data of 
		# previous computation when taking i-1 items 
		for w in range(W, 0, -1): 

			if wt[i] <= w: # if the item fits...
				# compute the value of the solution if we take the item
				new_value = dp[w-wt[i]] + val[i][i]

				for j in items_in_sol[w-wt[i]]:
					new_value += val[max(i,j)][min(i,j)] # val is a triangular matrix

				# check if the item improves the solution
				if dp[w] < new_value:
					# update the value
					dp[w] = new_value
					# update the items in the solution
					items_in_sol[w] = items_in_sol[w-wt[i]].copy()
					items_in_sol[w].append(i)
	
	# Returning the maximum value of knapsack 
	return dp[W], items_in_sol[W]