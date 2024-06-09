# -*- coding: utf-8 -*-
# doDMRG_MPO.py
import numpy as np
from numpy import linalg as LA
from ncon import ncon

def doDMRG_MPO_penalty(MPS_L,ML,M,MR,chi, w_penalty, MPS_penalty, numsweeps = 10, dispon = 2, updateon = True, maxit = 2, krydim = 4, normalization = False):
    """
    ------------------------
    by Glen Evenbly (c) for www.tensors.net, (v1.1)
    Modifications I made:
     - Make M be a list, to represent MPOs formed by different bulk matrices
     - Add penalty to find first excited state
    ------------------------
    Implementation of DMRG for a 1D chain with open boundaries, using the two-site update strategy.
    Each update is accomplished using a custom implementation of the Lanczos iteration to find (an approximation to) the
    ground state of the superblock Hamiltonian.

    Input 'MPS' is containing the MPS tensors whose length is equal to that of the 1D lattice.

    The Hamiltonian is specified by an MPO
    with 'ML' and 'MR' the tensors at the left and right boundaries, and 'M' the bulk MPO tensor.
    Automatically grow the MPS bond dimension to maximum dimension 'chi'.

    Outputs 'MPS' and 'MPS_B' are arrays of the MPS tensors in left and right orthogonal form respectively,
    while 'sWeight' is an array of the Schmidt coefficients across different lattice positions.

    'Ekeep' is a vector describing the energy at each update step.

    Optional arguments:
    `numsweeps::Integer=10`: number of DMRG sweeps
    `dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
    `updateon::Bool=true`: enable or disable tensor updates
    `maxit::Integer=2`: number of iterations of Lanczos method for each diagonalization
    `krydim::Integer=4`: maximum dimension of Krylov space in superblock diagonalization
    """

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    chid = M[0].shape[2] #local dimension
    Nsites = len(MPS_L)
    L = [0 for x in range(Nsites)]
    L[0] = ML
    R = [0 for x in range(Nsites)]
    R[Nsites-1] = MR

    # Dummy penalty MPS boundary matrices
    L_p = [0 for x in range(Nsites)]
    L_p[0] = np.zeros((1, 1))
    L_p[0][0, 0] = 1
    R_p = [0 for x in range(Nsites)]
    R_p[Nsites-1] = np.zeros((1, 1))
    R_p[Nsites-1][0, 0] = 1
  

    for p in range(Nsites-1):
        chil = MPS_L[p].shape[0]
        chir = MPS_L[p].shape[2]

        utemp, stemp, vhtemp = LA.svd(MPS_L[p].reshape(chil*chid,chir), full_matrices=False)

        MPS_L[p] = utemp.reshape(chil,chid,chir)
        MPS_L[p+1] = ncon([np.diag(stemp) @ vhtemp,MPS_L[p+1]], [[-1,1],[1,-2,-3]])/LA.norm(stemp)
        L[p+1] = ncon([L[p],M[p],MPS_L[p],np.conj(MPS_L[p])],[[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]])
        L_p[p+1] = ncon([L_p[p],MPS_L[p],np.conj(MPS_penalty[p])],[[1,4],[4,3,-3],[1,3,-2]])
    
    chil = MPS_L[Nsites-1].shape[0]
    chir = MPS_L[Nsites-1].shape[2]

    utemp, stemp, vhtemp = LA.svd(MPS_L[Nsites-1].reshape(chil*chid,chir), full_matrices=False)

    MPS_L[Nsites-1] = utemp.reshape(chil,chid,chir)
    sWeight = [0 for x in range(Nsites+1)]
    sWeight[Nsites] = (np.diag(stemp) @ vhtemp) / LA.norm(stemp)
    
    Ekeep = np.array([])
    MPS_R = [0 for x in range(Nsites)]

    for k in range(1,numsweeps+2):
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1:
            updateon = False
            dispon = 0
        
        ###### Optimization sweep: right-to-left
        for p in range(Nsites-2,-1,-1):
        
            ##### two-site update
            chil = MPS_L[p].shape[0]
            chir = MPS_L[p+1].shape[2]
            psiGround = ncon([MPS_L[p],MPS_L[p+1],sWeight[p+2]],[[-1,-2,1],[1,-3,2],[2,-4]]).reshape(chil*chid*chid*chir) # vector | psi >
            if updateon: # disables updates for the final sweep
                local_MPS_penalty = ncon([MPS_penalty[p],MPS_penalty[p+1]],[[-1,-2,1],[1,-3,-4]]).reshape(chil*chid*chid*chir) # orthogonality constraint
                psiGround, Entemp = eigLanczos(psiGround,doApplyMPO,(L[p],M[p],M[p+1],R[p+1],w_penalty,local_MPS_penalty,L_p[p],R_p[p+1],normalization), maxit = maxit, krydim = krydim)
                Ekeep = np.append(Ekeep,Entemp)
            
            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil*chid,chid*chir), full_matrices=False)
            chitemp = min(len(stemp),chi)
            MPS_L[p] = utemp[:,range(chitemp)].reshape(chil,chid,chitemp)
            sWeight[p+1] = np.diag(stemp[range(chitemp)]/LA.norm(stemp[range(chitemp)]))
            MPS_R[p+1] = vhtemp[range(chitemp),:].reshape(chitemp,chid,chir)
            
            ##### new block Hamiltonian
            R[p] = ncon([M[p+1],R[p+1],MPS_R[p+1],np.conj(MPS_R[p+1])],[[-1,2,3,5],[2,1,4],[-3,5,4],[-2,3,1]])
            R_p[p] = ncon([R_p[p+1],MPS_R[p+1],np.conj(MPS_penalty[p+1])],[[1,4],[-3,3,4],[-2,3,1]])
            
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
        
        ###### left boundary tensor
        chil = MPS_L[0].shape[0]; chir = MPS_L[0].shape[2]
        Atemp = ncon([MPS_L[0],sWeight[1]],[[-1,-2,1],[1,-3]]).reshape(chil,chid*chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        MPS_R[0] = vhtemp.reshape(chil,chid,chir)
        sWeight[0] = utemp @ (np.diag(stemp)/LA.norm(stemp))
        
        ###### Optimization sweep: left-to-right
        for p in range(Nsites-1):
        
            ##### two-site update
            chil = MPS_R[p].shape[0]
            chir = MPS_R[p+1].shape[2]
            psiGround = ncon([sWeight[p],MPS_R[p],MPS_R[p+1]],[[-1,1],[1,-2,2],[2,-3,-4]]).reshape(chil*chid*chid*chir)
            if updateon: # disables updates for the final sweep
                local_MPS_penalty = ncon([MPS_penalty[p],MPS_penalty[p+1]],[[-1,-2,2],[2,-3,-4]]).reshape(chil*chid*chid*chir)
                psiGround, Entemp = eigLanczos(psiGround,doApplyMPO,(L[p],M[p],M[p+1],R[p+1],w_penalty,local_MPS_penalty,L_p[p],R_p[p+1],normalization), maxit = maxit, krydim = krydim)
                Ekeep = np.append(Ekeep,Entemp)
            
            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil*chid,chid*chir), full_matrices=False)
            chitemp = min(len(stemp),chi)
            MPS_L[p] = utemp[:,range(chitemp)].reshape(chil,chid,chitemp)
            sWeight[p+1] = np.diag(stemp[range(chitemp)]/LA.norm(stemp[range(chitemp)]))
            MPS_R[p+1] = vhtemp[range(chitemp),:].reshape(chitemp,chid,chir)
            
            ##### new block Hamiltonian
            L[p+1] = ncon([L[p],M[p],MPS_L[p],np.conj(MPS_L[p])],[[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]])
            L_p[p+1] = ncon([L_p[p],MPS_L[p],np.conj(MPS_penalty[p])],[[1,4],[4,3,-3],[1,3,-2]])

            ##### display energy
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
                
        ###### right boundary tensor
        chil = MPS_R[Nsites-1].shape[0]
        chir = MPS_R[Nsites-1].shape[2]
        Atemp = ncon([MPS_R[Nsites-1],sWeight[Nsites-1]],[[1,-2,-3],[-1,1]]).reshape(chil*chid,chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        MPS_L[Nsites-1] = utemp.reshape(chil,chid,chir)
        sWeight[Nsites] = (stemp/LA.norm(stemp))*vhtemp
        
        if dispon == 1:
            print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))
            
    return Ekeep, MPS_L, sWeight, MPS_R


#-------------------------------------------------------------------------
def doApplyMPO(psi,L,M1,M2,R, w_penalty, psi_penalty,L_p,R_p,normalization):
    """ function for applying MPO to state """    

    psi_reshaped = psi.reshape(L.shape[2],M1.shape[3],M2.shape[3],R.shape[2])
    psi_penalty_reshaped = psi_penalty.reshape(L_p.shape[0],M1.shape[3],M2.shape[3],R_p.shape[0])

    # compute H|psi>
    result = ncon([psi_reshaped,L,M1,M2,R],
                  [[1,3,5,7],[2,-1,1],[2,4,-2,3],[4,6,-3,5],[6,-4,7]]).reshape(L.shape[2]*M1.shape[3]*M2.shape[3]*R.shape[2])
        
    # add penalty term w|psi_penalty><psi_penalty|psi>
    ortho_term = ncon([L_p,np.conj(psi_penalty_reshaped),R_p],
                      [[1,-1],[1,-2,-3,2],[2,-4]]).reshape(L.shape[2]*M1.shape[3]*M2.shape[3]*R.shape[2])
    
    if normalization:
        overlap = np.dot(ortho_term, psi) / (np.linalg.norm(ortho_term)*np.linalg.norm(psi))
    else:
        overlap = np.dot(ortho_term, psi)

    result += w_penalty*overlap*np.conj(ortho_term)
    
    return result

#-------------------------------------------------------------------------
def eigLanczos(psivec,linFunct,functArgs, maxit = 2, krydim = 4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    
    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec)) # initial guess for the eigenvector in case it is null
    
    psi = np.zeros([len(psivec),krydim+1])
    A = np.zeros([krydim,krydim]) # stores the projected operator
    dval = 0
    
    for ik in range(maxit):
        
        # psi is the normalized psivec
        psi[:,0] = psivec/max(LA.norm(psivec),1e-16)

        # compute the updated ip-th basis vector in psi
        for ip in range(1,krydim+1):
                
            # apply the MPO to the state  ( \psi' = H \psi)
            psi[:,ip] = linFunct(psi[:,ip-1],*functArgs)
            
            for ig in range(ip): # compute the projected operator
                A[ip-1,ig] = np.dot(psi[:,ip],psi[:,ig])
                A[ig,ip-1] = np.conj(A[ip-1,ig])
            
            for ig in range(ip): # ensure orthogonality and normalize
                psi[:,ip] = psi[:,ip] - np.dot(psi[:,ig],psi[:,ip])*psi[:,ig]
                psi[:,ip] = psi[:,ip] / max(LA.norm(psi[:,ip]),1e-16)
                    
        [dtemp,utemp] = LA.eigh(A) # Computes eigenvalues and eigenvectors 
                                   # dtemp contains the eigenvalues in ascending order

        # update the estimate of the eigenvector
        psivec = psi[:,range(0,krydim)] @ utemp[:,0] # utemp[:,0] is eigenvector of the smallest eigenvalue
        
    # store the updated eigenvector and eigenvalue
    psivec = psivec/LA.norm(psivec)
    dval = dtemp[0] # smallest eigenvalue
    
    return psivec, dval