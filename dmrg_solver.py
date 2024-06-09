from overrides import override
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from ncon import ncon
from DMRG.module import doDMRG_MPO 
from DMRG.module_penalty import doDMRG_MPO_penalty 
from DMRG.annealing_ham_to_MPO import from_ham_coeff

from DMRG.samplingMPS import SamplingMPS
from solver import Solver


class DMRG_solver(Solver):
    def __init__(self, W, wt, val, chi, opts_maxit = 2, opts_krydim = 4):
        super().__init__(W, wt, val)

        self.chi = chi # bond dimension

        self.OPTS_dispon = 0 # level of output display
        self.OPTS_updateon = True # level of output display
        self.OPTS_maxit = opts_maxit # iterations of Lanczos method
        self.OPTS_krydim = opts_krydim # dimension of Krylov subspace

        self.M = None # list of tensors forming the MPO
        self.ML = None #left MPO boundary
        self.MR = None #right MPO boundary

        #### Initialize MPS tensors
        chid = 2 # local dimension
        self.A = [0 for x in range(self.N)]
        self.A[0] = np.random.rand(1,chid,min(chi,chid))
        for k in range(1,self.N):
            self.A[k] = np.random.rand(self.A[k-1].shape[2],chid,min(min(chi,self.A[k-1].shape[2]*chid),chid**(self.N-k-1)))

        self.solution_energy = None
        self.solution_MPS = None
        self.show_run_plots = True

    def build_MPO_time_s(self, s):
        '''
        Builds the MPO for the instance of the QKP given at the annealing time s \in [0,1]
        '''
        self.M = from_ham_coeff(self.N, self.J_coeffs, self.h_coeffs, s)

        # Dummy MPO boundary matrices
        L_dim = self.M[0].shape[0]
        R_dim = self.M[-1].shape[2]

        # Define the left and right MPO boundaries
        self.ML = np.zeros((L_dim, 1, 1))
        self.ML[0, 0, 0] = 1

        self.MR = np.zeros((R_dim, 1, 1))
        self.MR[-1, 0, 0] = 1

    @override
    def run(self, time = 50):
        '''
        Runs DMRG for the MPO built in the last execution of the method self.build_MPO_time_s()

        time: number of DMRG sweeps
        '''
        if self.M is None:
            raise Exception('Call build_MPO_time_s method first')
        
        #### Do DMRG sweeps (2-site approach)
        energies, left_MPS, sWeight, right_MPS = doDMRG_MPO(self.A, self.ML, self.M, self.MR, self.chi,
                                                numsweeps = time, dispon = self.OPTS_dispon, 
                                                updateon = self.OPTS_updateon, maxit = self.OPTS_maxit,
                                                krydim = self.OPTS_krydim)
        
        self.solution_energy = energies[-1]
        self.solution_MPS = right_MPS

        if self.show_run_plots:
            plt.plot(energies)
            plt.xlabel('Half sweeps')
            plt.ylabel('Energy')
            plt.title('Run DMRG plot')
            plt.show()
            print(f'Solution energy = {self.solution_energy} + {self.offset} (offset) = {self.solution_energy + self.offset}')

        # Obtain items in the result
        samplingMPS = SamplingMPS()
        result_dict = samplingMPS.sampling(right_MPS, 1000)

        # get the state that appeared the most in the sampling
        max_state = max(result_dict, key=lambda k: result_dict[k])

        self.solution_items = []

        for i, ch in enumerate(max_state):
            if ch == '0': # dimod mapping
                self.solution_items.append(i)

        return left_MPS

    def run_penalty(self, w_penalty, MPS_penalti_L, numsweeps = 50, normalization = False):
        '''
        Runs DMRG for the MPO built in the last execution of the method self.build_MPO_time_s(),
        but adding a penalty to the MPS_penalti_L given.

        If MPS_penalty_L encodes the ground state, the method will return the first excited state
        (if the penalty w_penalty is appropiate).
        '''
        if self.M is None:
            raise Exception('Call build_MPO_time_s method first')
        
        # Custom initialitzation of the MPS for the DMRG
        init_MPS = [0 for x in range(self.N)]

        matrix_0 = np.array([[0, 1]])  # dimod mapping

        for k in range(self.N):
            init_MPS[k] = np.zeros((MPS_penalti_L[k].shape[0],2,MPS_penalti_L[k].shape[2]))
            # initialize tensors with 0
            for i in range(init_MPS[k].shape[0]):
                for j in range(init_MPS[k].shape[2]):  
                    init_MPS[k][i,:,j] = matrix_0
        
        # Uncomment to check the overlap
        '''overlap = ncon(
            [np.conj(MPS_penalti_L[0]), init_MPS[0]], [[1, 2, -1], [1, 2, -2]]
        )
        for i in range(1, self.N - 1):
            overlap = ncon([overlap, np.conj(MPS_penalti_L[i]), init_MPS[i]], [[1, 2], [1, 3, -1], [2, 3, -2]])

        overlap = ncon([overlap, np.conj(MPS_penalti_L[self.N-1]), init_MPS[self.N-1]], [[1, 2], [1, 3, 4], [2, 3, 4]])

        print(overlap)'''

        #### Do DMRG sweeps (2-site approach)
        energies, left_MPS, sWeight, right_MPS = doDMRG_MPO_penalty(init_MPS, self.ML, self.M, self.MR, self.chi,
                                                w_penalty, MPS_penalti_L,
                                                numsweeps = numsweeps, dispon = self.OPTS_dispon, 
                                                updateon = self.OPTS_updateon, maxit = self.OPTS_maxit,
                                                krydim = self.OPTS_krydim, normalization=normalization)
        
        if self.show_run_plots:
            plt.plot(energies)
            plt.xlabel('Half sweeps')
            plt.ylabel('Energy')
            plt.title('Run_penalty plot')
            plt.show()
        
        return energies[-1]

    def annealing_run(self, penalty = 100, step = 10, numsweeps = 50, normalization = False):
        '''
        Simulates the annealing evolution of the Hamiltonian representing the instance of the QKP given for
		a discrete set of steps. For each time step of the evolution, it computes the gap with DMRG and
        compares it with the real gap obtained with exact diagonalization, to check the accuracy of the results.
        '''

        gs_energies = []
        exc_energies = []

        real_gaps = []

        self.show_run_plots = False

        for s in range(step+1):
            print(' ---- s=', s / step, ' ----')

            self.build_MPO_time_s(s / step)
            MPS_L = self.run()
            gs_energies.append(self.solution_energy)

            # compute first excited state
            e = self.run_penalty(penalty, MPS_L, numsweeps = numsweeps, normalization = normalization)
            exc_energies.append(e)

            if self.show_run_plots:
                print(f'DMRG gap: [{gs_energies[-1]}, {exc_energies[-1]}]')

            # build the ham matrix to test results
            aux_M = self.M.copy()

            aux_M[0] = aux_M[0][0, :] 
            aux_M[-1] = aux_M[-1][:, -1]

            ham = aux_M[0]
            for i in range(1, len(self.M)-1):
                ham = ncon([ham, aux_M[i]], [[1, -2, -4], [1, -1, -3, -5]])
                ham = ham.reshape(ham.shape[0], 2**(i+1), 2**(i+1))
            ham = ncon([ham, aux_M[-1]], [[1, -1, -3], [1, -2, -4]])
            ham = ham.reshape(2**self.N, 2**self.N)

            # perform exact diagonalization to test results
            eigenvalues, eigenvectors = np.linalg.eig(ham)
            gap = sorted(eigenvalues)[:2]
            real_gaps.append(gap[1]-gap[0])
            if self.show_run_plots:
                print('Real gap: ', gap)

        self.show_run_plots = True

        self.dmrg_gaps = [exc_energies[i] - gs_energies[i] for i in range(len(gs_energies))]

        error = 0
        for i in range(len(self.dmrg_gaps)):
            error += (self.dmrg_gaps[i] - real_gaps[i])**2
        
        if self.show_run_plots:
            x_values = [s/step for s in range(step+1)]

            plt.figure()
            plt.plot(x_values, gs_energies, label='Ground energies', marker='o')
            plt.plot(x_values, exc_energies, label='First excited energies', marker='x')

            y_ticks = np.linspace(min(gs_energies), max(exc_energies), num=5) # create evenly spaced ticks
            plt.xticks(fontsize=14) 
            plt.yticks(y_ticks, fontsize=14)
            plt.title('DMRG annealing run')
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Energy', fontsize=16)
            plt.legend(fontsize=15)
            plt.show() 

            plt.figure()
            plt.plot(x_values, self.dmrg_gaps, label='DMRG gaps', marker='o')
            plt.plot(x_values, real_gaps, label='Real gaps', marker='x')
            
            y_ticks = np.linspace(min(real_gaps), max(real_gaps), num=5) # create evenly spaced ticks
            plt.xticks(fontsize=14) 
            plt.yticks(y_ticks, fontsize=14)
            plt.title('Annealing gaps')
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Gap', fontsize=16)
            plt.legend(fontsize=15)
            plt.show()

        return error
    
    def annealing_time_estimation(self):
        '''
        Computes a custom annealing schedule time based on the gap
        approximation computed with DMRG.
        '''
        if self.dmrg_gaps is None:
            raise Exception('Call annealing_run method first')
        
        step = len(self.dmrg_gaps)

        x = np.linspace(0, 1, num = len(self.dmrg_gaps))
        def velocity(t):
            g_min = min(self.dmrg_gaps)
            g_max = max(self.dmrg_gaps)

            return (self.dmrg_gaps[t] - g_min + 0.05)/(g_max - g_min)
        y = np.array([velocity(s) for s in range(step)])

        # Fit a polynomial of degree 3
        p = np.polynomial.polynomial.Polynomial.fit(x, y, 3)

        # Integrate the polynomial
        p_integ = p.integ()

        # Shifted and scaled function so that s(0) = 0 and s(1) = 1
        s_0 = p_integ(0)
        s_1 = p_integ(1)

        def scaled_polynomial(t):
            return (p_integ(t) - s_0) / (s_1 - s_0)
        
        schedule_points = scaled_polynomial(x)
        # Plot the data points, fitted polynomial, and its integral
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Time (t)', fontsize=16)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_ylabel('Scheduled time s(t)', color='tab:red', fontsize=16)
        ax1.plot(x, schedule_points, color='tab:red')
        ax1.plot(x, x, color='tab:green', label=r'$s(t)=t$', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
        ax1.legend(fontsize=15)

        ax2 = ax1.twinx()
        ax2.set_ylabel('DMRG gap', color='tab:blue', fontsize=16)  # we already handled the x-label with ax1
        ax2.plot(x, self.dmrg_gaps, label='DMRG gaps', color='tab:blue', marker='o')
        ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=14) 
        
        ax1.locator_params(nbins=5)
        ax2.locator_params(nbins=6)
        
        plt.title('Polynomial Fit and Its Integral')
        plt.show()
        
        return scaled_polynomial


    def energy_of_items(self, items):
        '''
        Method with testing purposes. Given a selection of items of the QKP,
        it computes the energy of their corresponding state.
        '''
        print(' - Evaluating candidate ', items)

        aux_M = self.M.copy() 
        aux_M[0] = aux_M[0][0, :]
        aux_M[-1] = aux_M[-1][:, -1] # M with no dummy indices

        # Create the MPS corresponding to the state that represents the items given
        custom_MPS = []

        matrix_0 = np.array([[0, 1]]) # dimod mapping
        reshaped_matrix_0 = matrix_0.reshape(1,2,1)

        matrix_1 = np.array([[1, 0]])
        reshaped_matrix_1 = matrix_1.reshape(1,2,1)

        for s in items:
            if s=='0':
                custom_MPS.append(reshaped_matrix_0)
            elif s=='1':
                custom_MPS.append(reshaped_matrix_1)

        # Do the contraction <\psi|H|\psi> to obtain the energy
        S_left = ncon(
            [custom_MPS[0], aux_M[0], custom_MPS[0]], [[1, 2, -1], [-2, 2, 3], [1, 3, -3]]
        )
        for i in range(1, self.N - 1):
            tensor = custom_MPS[i]
            S_left = ncon([S_left, tensor, aux_M[i], np.conj(tensor)], [[1, 2, 3], [1, 4, -1], [2, -2, 4, 5], [3, 5, -3]])
        S_left = ncon(
            [S_left, custom_MPS[-1], aux_M[-1], np.conj(custom_MPS[-1])],
            [[1, 2, 3], [1, 4, 5], [2, 4, 6], [3, 6, 5]],
        )

        print(f'Energy: {S_left}')
        print(f'+ offset ({self.offset}) = {S_left+self.offset}')