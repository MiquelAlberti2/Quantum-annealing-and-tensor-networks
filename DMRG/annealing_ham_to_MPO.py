import numpy as np

def from_ham_coeff(N, J_coeffs, h_coeffs, s):
    """
    Construct an MPO from a annealing Hamiltonian for QUBO problems

    Args:
        N: number of qubits
        J_coeffs: coefficients of the long range interactions
        h_coeffs: coefficients of the short range interactions
        s: transverse field strength

    Raises:
        NotImplementedError: Type of Hamiltonian not supported.

    Returns:
        MPO (List[array]): List of tensors representing the MPO (1 tensor for each qubit)
        offset (float): Offset for the energy of the Hamiltonian
    """
    # TODO Simplify MPOs if certain given coefficients are 0

    # Start building the Tensor Network
    i_matrix = np.array([[1, 0], [0, 1]])
    x_matrix = np.array([[0, 1], [1, 0]])
    z_matrix = np.array([[1, 0], [0, -1]])
    
    N_by_2 = int(N / 2)

    a_list = [np.zeros((k + 2, k + 3, 2,2)) for k in range(1, N_by_2)]

    aux = 2 if N % 2 == 0 else 3
    a_list += [np.zeros((N_by_2 + 2, N_by_2 + aux, 2,2))]

    a_list += [np.zeros((N - k + 3, N - k + 2, 2,2)) for k in range(N_by_2 + 1, N + 1)]

    for k in range(1, N+1):
        a_list[k-1][0,0] = i_matrix
        a_list[k-1][-1,-1] = i_matrix
        a_list[k-1][-2,-1] = z_matrix
        a_list[k-1][0,-1] = (-1)*(1-s)*x_matrix + s*h_coeffs[k-1]*z_matrix
        
        if k < N_by_2:
            a_list[k-1][0,1] = z_matrix 
            a_list[k-1][0,k+1] = s*J_coeffs[k-1,k]*z_matrix

            for m in range(2, k+1):
                a_list[k-1][m-1, m] = i_matrix
                a_list[k-1][m-1, k+1] = s*J_coeffs[k-m, k]*i_matrix

        elif k == N_by_2:
            for n in range(2, N_by_2 + aux):
                a_list[k-1][0, n-1] = s*J_coeffs[N_by_2-1, N-n+1]*z_matrix
                for m in range(2, k+1):
                    a_list[k-1][m-1, n-1] = s*J_coeffs[N_by_2-m,N-n+1]*i_matrix

        else: # k+1 > N_by_2:
            for m in range(2, N-k+2):
                a_list[k-1][0, m-1] = s*J_coeffs[k-1,N-m+1]*z_matrix
                a_list[k-1][m-1, m-1] = i_matrix

    '''
    a_list[0] = a_list[0][0, :] # other rows are used to propagate information of previous tensors, does not make sense to keep them
    a_list[-1] = a_list[-1][:, -1] # other columns are used to propagate information to the next tensors, does not make sense to keep them

    We don't need this bc the implementation uses dummy boundary matrices
    '''
    return a_list