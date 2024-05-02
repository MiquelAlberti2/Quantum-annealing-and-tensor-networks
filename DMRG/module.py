# -*- coding: utf-8 -*-
# doDMRG_MPO.py
import numpy as np
from numpy import linalg as LA
from ncon import ncon

def doDMRG_MPO(MPS,ML,M,MR,chi, numsweeps = 10, dispon = 2, updateon = True, maxit = 2, krydim = 4):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
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
    Nsites = len(MPS)
    L = [0 for x in range(Nsites)]
    L[0] = ML
    R = [0 for x in range(Nsites)]
    R[Nsites-1] = MR

    for p in range(Nsites-1):
        chil = MPS[p].shape[0]
        chir = MPS[p].shape[2]

        utemp, stemp, vhtemp = LA.svd(MPS[p].reshape(chil*chid,chir), full_matrices=False)

        MPS[p] = utemp.reshape(chil,chid,chir)
        MPS[p+1] = ncon([np.diag(stemp) @ vhtemp,MPS[p+1]], [[-1,1],[1,-2,-3]])/LA.norm(stemp)
        L[p+1] = ncon([L[p],M[p],MPS[p],np.conj(MPS[p])],[[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]])
    
    chil = MPS[Nsites-1].shape[0]
    chir = MPS[Nsites-1].shape[2]

    utemp, stemp, vhtemp = LA.svd(MPS[Nsites-1].reshape(chil*chid,chir), full_matrices=False)

    MPS[Nsites-1] = utemp.reshape(chil,chid,chir)
    sWeight = [0 for x in range(Nsites+1)]
    sWeight[Nsites] = (np.diag(stemp) @ vhtemp) / LA.norm(stemp)
    
    Ekeep = np.array([])
    MPS_B = [0 for x in range(Nsites)]

    for k in range(1,numsweeps+2):
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1:
            updateon = False
            dispon = 0
        
        ###### Optimization sweep: right-to-left
        for p in range(Nsites-2,-1,-1):
        
            ##### two-site update
            chil = MPS[p].shape[0]
            chir = MPS[p+1].shape[2]
            psiGround = ncon([MPS[p],MPS[p+1],sWeight[p+2]],[[-1,-2,1],[1,-3,2],[2,-4]]).reshape(chil*chid*chid*chir)
            if updateon: # disables updates for the final sweep
                psiGround, Entemp = eigLanczos(psiGround,doApplyMPO,(L[p],M[p],M[p+1],R[p+1]), maxit = maxit, krydim = krydim)
                Ekeep = np.append(Ekeep,Entemp)
            
            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil*chid,chid*chir), full_matrices=False)
            chitemp = min(len(stemp),chi)
            MPS[p] = utemp[:,range(chitemp)].reshape(chil,chid,chitemp)
            sWeight[p+1] = np.diag(stemp[range(chitemp)]/LA.norm(stemp[range(chitemp)]))
            MPS_B[p+1] = vhtemp[range(chitemp),:].reshape(chitemp,chid,chir)
            
            ##### new block Hamiltonian
            R[p] = ncon([M[p+1],R[p+1],MPS_B[p+1],np.conj(MPS_B[p+1])],[[-1,2,3,5],[2,1,4],[-3,5,4],[-2,3,1]])
            
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
        
        ###### left boundary tensor
        chil = MPS[0].shape[0]; chir = MPS[0].shape[2]
        Atemp = ncon([MPS[0],sWeight[1]],[[-1,-2,1],[1,-3]]).reshape(chil,chid*chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        MPS_B[0] = vhtemp.reshape(chil,chid,chir)
        sWeight[0] = utemp @ (np.diag(stemp)/LA.norm(stemp))
        
        ###### Optimization sweep: left-to-right
        for p in range(Nsites-1):
        
            ##### two-site update
            chil = MPS_B[p].shape[0]
            chir = MPS_B[p+1].shape[2]
            psiGround = ncon([sWeight[p],MPS_B[p],MPS_B[p+1]],[[-1,1],[1,-2,2],[2,-3,-4]]).reshape(chil*chid*chid*chir)
            if updateon: # disables updates for the final sweep
                psiGround, Entemp = eigLanczos(psiGround,doApplyMPO,(L[p],M[p],M[p+1],R[p+1]), maxit = maxit, krydim = krydim)
                Ekeep = np.append(Ekeep,Entemp)
            
            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil*chid,chid*chir), full_matrices=False)
            chitemp = min(len(stemp),chi)
            MPS[p] = utemp[:,range(chitemp)].reshape(chil,chid,chitemp)
            sWeight[p+1] = np.diag(stemp[range(chitemp)]/LA.norm(stemp[range(chitemp)]))
            MPS_B[p+1] = vhtemp[range(chitemp),:].reshape(chitemp,chid,chir)
            
            ##### new block Hamiltonian
            L[p+1] = ncon([L[p],M[p],MPS[p],np.conj(MPS[p])],[[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]])
        
            ##### display energy
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
                
        ###### right boundary tensor
        chil = MPS_B[Nsites-1].shape[0]
        chir = MPS_B[Nsites-1].shape[2]
        Atemp = ncon([MPS_B[Nsites-1],sWeight[Nsites-1]],[[1,-2,-3],[-1,1]]).reshape(chil*chid,chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        MPS[Nsites-1] = utemp.reshape(chil,chid,chir)
        sWeight[Nsites] = (stemp/LA.norm(stemp))*vhtemp
        
        if dispon == 1:
            print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))
            
    return Ekeep, MPS, sWeight, MPS_B


#-------------------------------------------------------------------------
def doApplyMPO(psi,L,M1,M2,R):
    """ function for applying MPO to state """
    
    return ncon([psi.reshape(L.shape[2],M1.shape[3],M2.shape[3],R.shape[2]),L,M1,M2,R],
                       [[1,3,5,7],[2,-1,1],[2,4,-2,3],[4,6,-3,5],[6,-4,7]]).reshape( 
                               L.shape[2]*M1.shape[3]*M2.shape[3]*R.shape[2])

#-------------------------------------------------------------------------
def eigLanczos(psivec,linFunct,functArgs, maxit = 2, krydim = 4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    
    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))
    
    psi = np.zeros([len(psivec),krydim+1])
    A = np.zeros([krydim,krydim])
    dval = 0
    
    for ik in range(maxit):
        
        psi[:,0] = psivec/max(LA.norm(psivec),1e-16)
        for ip in range(1,krydim+1):
                
            psi[:,ip] = linFunct(psi[:,ip-1],*functArgs)
            
            for ig in range(ip):
                A[ip-1,ig] = np.dot(psi[:,ip],psi[:,ig])
                A[ig,ip-1] = np.conj(A[ip-1,ig])
            
            for ig in range(ip):
                psi[:,ip] = psi[:,ip] - np.dot(psi[:,ig],psi[:,ip])*psi[:,ig]
                psi[:,ip] = psi[:,ip] / max(LA.norm(psi[:,ip]),1e-16)
                    
        [dtemp,utemp] = LA.eigh(A)
        psivec = psi[:,range(0,krydim)] @ utemp[:,0]
        
    psivec = psivec/LA.norm(psivec)
    dval = dtemp[0]
    
    return psivec, dval