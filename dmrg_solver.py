from overrides import override
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from ncon import ncon
from DMRG.module import doDMRG_MPO 
from DMRG.annealing_ham_to_MPO import from_ham_coeff

from DMRG.samplingMPS import SamplingMPS
from solver import Solver


class DMRG_solver(Solver):
    def __init__(self, W, wt, val, chi):
        super().__init__(W, wt, val)

        self.chi = chi # set bond dimension

        self.OPTS_dispon = 2 # level of output display
        self.OPTS_updateon = True # level of output display
        self.OPTS_maxit = 2 # iterations of Lanczos method
        self.OPTS_krydim = 4 # dimension of Krylov subspace

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
        time (float): # number of DMRG sweeps
        '''
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
            plt.show()

        print(f'Solution energy = {self.solution_energy} + {self.offset} (offset) = {self.solution_energy + self.offset}')

        samplingMPS = SamplingMPS()
        result_dict = samplingMPS.sampling(right_MPS, 1000)

        # get the state that appeared the most in the sampling
        max_state = max(result_dict, key=lambda k: result_dict[k])

        self.solution_items = []

        for i, ch in enumerate(max_state):
            if ch == '0': # dimod mapping
                self.solution_items.append(i)

    def annealing_run(self, step = 10):

        self.show_run_plots = False

        energies = []

        for s in range(step+1):
            print(' ---- s=', s, ' ----')

            self.build_MPO_time_s(s / step)
            self.run()
            energies.append(self.solution_energy)

            print('DMRG energy: ', self.solution_energy)

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

            # perform exact diagonalization
            eigenvalues, eigenvectors = np.linalg.eig(ham)
            gap = sorted(eigenvalues)[:2]

            print('Real gap: ', gap)


        plt.plot(energies)
        plt.xlabel('step')
        plt.ylabel('Energy')
        plt.show()

        self.show_run_plots = True

    def energy_of_items(self, items):
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

    def show_results(self):
        print('Energy: ', self.solution_energy)
        
        # TODO this method is hardcoded for N=3
        # contract the MPS to get the state
        E = ncon(self.solution_MPS, [[-1,-2,1],[1,-3,2],[2,-4,-5]])
        bulk_E = E.reshape(2,2,2)

        max_item = '-1-1-1'
        max_amplitud = 0

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    print(f'state {i}{j}{k}: {bulk_E[i,j,k]}')

                    if abs(bulk_E[i,j,k]) > max_amplitud:
                        max_item = f'{i}{j}{k}'
                        max_amplitud = abs(bulk_E[i,j,k])

        print('Solution: ', max_item)



'''#### Compare with exact results (computed from free fermions)
H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
D = LA.eigvalsh(H)
EnExact = 2*sum(D[D < 0])

##### Plot results
plt.figure(1)
plt.yscale('log')
plt.plot(range(len(En1)), En1 - EnExact, 'b', label="chi = 16")
plt.plot(range(len(En2)), En2 - EnExact, 'r', label="chi = 32")
plt.legend()
plt.title('DMRG for XX model')
plt.xlabel('Update Step')
plt.ylabel('Ground Energy Error')
plt.show()'''