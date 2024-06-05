from dmrg_solver import DMRG_solver
from itertools import product


def optimize(W_capacity, weights, values, chi_list, opts_maxit_list, opts_krydim_list, numsweeps_list, penalties_list, normalization_list):
	lowest_error = float('inf')
	best_combo = None

	# compute total number of combinations to create a "progress bar"
	n_comb = len(chi_list)*len(opts_maxit_list)*len(opts_krydim_list)*len(numsweeps_list)*len(penalties_list)*len(normalization_list)

	counter = 1
	for combo in product(chi_list, opts_maxit_list, opts_krydim_list, numsweeps_list, penalties_list, normalization_list):
		print(f'Combination {counter}/{n_comb}: {combo}')
		counter += 1

		chi = combo[0]
		opts_maxit = combo[1]
		opts_krydim = combo[2]
		numsweeps = combo[3]
		penalty = combo[4]
		normalization = combo[5]

		qkp_DMRG = DMRG_solver(W_capacity, weights, values, chi, opts_maxit, opts_krydim)
		error = qkp_DMRG.annealing_run(penalty, step=5, numsweeps = numsweeps, normalization=normalization)

		if error < lowest_error:
			lowest_error = error
			best_combo = combo
			print('IMPROVEMENT:')
			print(lowest_error)

	print('-------------------- Final solution --------------------')
	print(lowest_error)
	print(best_combo)

	chi = best_combo[0]
	opts_maxit = best_combo[1]
	opts_krydim = best_combo[2]
	numsweeps = best_combo[3]
	penalty = best_combo[4]
	normalization = best_combo[5]

	qkp_DMRG = DMRG_solver(W_capacity, weights, values, chi, opts_maxit, opts_krydim)
	qkp_DMRG.show_run_plots = True
	error = qkp_DMRG.annealing_run(penalty, step=5, numsweeps = numsweeps, normalization=normalization)
	print('Final error: ', error)

	


		
	
  


  

