# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from ncon import ncon
from DMRG.module import doDMRG_MPO 
from DMRG.annealing_ham_to_MPO import from_ham_coeff


class QKP_DMRG:
    def __init__(self, N, chi, numsweeps, h_coeffs, J_coeffs, offset):
        self.Nsites = N
        self.chi = chi # set bond dimension
        self.OPTS_numsweeps = numsweeps # number of DMRG sweeps

        self.J = J_coeffs
        self.h = h_coeffs
        self.offset = offset

        self.OPTS_dispon = 2 # level of output display
        self.OPTS_updateon = True # level of output display
        self.OPTS_maxit = 2 # iterations of Lanczos method
        self.OPTS_krydim = 4 # dimension of Krylov subspace

        self.M = None # list of tensors forming the MPO
        self.ML = None #left MPO boundary
        self.MR = None #right MPO boundary

        #### Initialize MPS tensors
        chid = 2 # local dimension
        self.A = [0 for x in range(self.Nsites)]
        self.A[0] = np.random.rand(1,chid,min(chi,chid))
        for k in range(1,self.Nsites):
            self.A[k] = np.random.rand(self.A[k-1].shape[2],chid,min(min(chi,self.A[k-1].shape[2]*chid),chid**(self.Nsites-k-1)))

        self.solution_energy = None
        self.solution_MPS = None

    def build_MPO_time_s(self, s):
        self.M = from_ham_coeff(self.Nsites, self.J, self.h, s)

        # Dummy MPO boundary matrices
        L_dim = self.M[0].shape[0]
        R_dim = self.M[-1].shape[2]

        # Define the left MPO boundary
        self.ML = np.zeros((L_dim, 1, 1))
        self.ML[0, 0, 0] = 1

        self.MR = np.zeros((R_dim, 1, 1))
        self.MR[-1, 0, 0] = 1

    def run(self):
        #### Do DMRG sweeps (2-site approach)
        En1, left_MPS, sWeight, right_MPS = doDMRG_MPO(self.A, self.ML, self.M, self.MR, self.chi,
                                        numsweeps = self.OPTS_numsweeps, dispon = self.OPTS_dispon, 
                                        updateon = self.OPTS_updateon, maxit = self.OPTS_maxit,
                                        krydim = self.OPTS_krydim)
        
        self.solution_energy = En1[-1] + self.offset
        self.solution_MPS = right_MPS

    def energy_of_items(self, items):
        print(' - Evaluating candidate ', items)

        aux_M = self.M.copy() # M with no dummy indices
        aux_M[0] = aux_M[0][0, :]
        aux_M[-1] = aux_M[-1][:, -1]

        custom_MPS = []

        matrix_0 = np.array([[1, 0]])
        reshaped_matrix_0 = matrix_0.reshape(1,2,1)

        matrix_1 = np.array([[0, 1]])
        reshaped_matrix_1 = matrix_1.reshape(1,2,1)

        for s in items:
            if s=='0':
                custom_MPS.append(reshaped_matrix_0)
            elif s=='1':
                custom_MPS.append(reshaped_matrix_1)

        S_left = ncon(
            [custom_MPS[0], aux_M[0], custom_MPS[0]], [[1, 2, -1], [-2, 2, 3], [1, 3, -3]]
        )
        for i in range(1, self.Nsites - 1):
            tensor = custom_MPS[i]
            S_left = ncon([S_left, tensor, aux_M[i], np.conj(tensor)], [[1, 2, 3], [1, 4, -1], [2, -2, 4, 5], [3, 5, -3]])
        S_left = ncon(
            [S_left, custom_MPS[-1], aux_M[-1], np.conj(custom_MPS[-1])],
            [[1, 2, 3], [1, 4, 5], [2, 4, 6], [3, 6, 5]],
        )

        print(f'Energy: {S_left + self.offset}')

    def show_results(self):
        print('Energy: ', self.solution_energy)

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