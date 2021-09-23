from __future__ import print_function
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import scipy.sparse
import scipy.sparse.linalg
import math
import sys

class Variational_Ansatz:
    """
    Assumes that we have some operator, which will be applied in a specific way G*psi or exp{iG}*psi to create a trial state
    """
    def __init__(self,_H, _G, _ref, _params):
        """
        _H      : sparse matrix
        _G_ops  : list of sparse matrices - each corresponds to a variational parameter
        _ref    : reference state vector
        _params : initialized list of parameters
        """

        self.H = _H
        self.G = _G
        self.ref = cp.deepcopy(_ref)
        self.curr_params = _params 
        self.n_params = len(self.curr_params)
        self.hilb_dim = self.H.shape[0] 
        
        self.iter = 0
        self.energy_per_iteration = []
        self.psi_norm = 1.0
        self.n_procs = 1

    def energy(self,params):
        print(" VIRTUAL Class: please override")
        exit()
    
    def gradient(self,params):
        print(" VIRTUAL Class: please override")
        exit()
    
    def prepare_state(self,params):
        print(" VIRTUAL Class: please override")
        exit()

        
    def callback(self,x):
        try:
            err = np.sqrt(np.vdot(self.der, self.der))
            print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(self.iter,
                self.curr_energy.real, err, np.max(np.abs(self.der))))
        except:
            print(" Iter:%4i Current Energy = %20.16f Psi Norm Error %10.1e" %(self.iter,
                self.curr_energy.real, 1-self.psi_norm))
        self.iter += 1
        self.energy_per_iteration.append(self.curr_energy)
        sys.stdout.flush()




class tUCCSD(Variational_Ansatz):
    
    def energy(self,params):
        new_state = self.prepare_state(params)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state:
        exp{A1}exp{A2}exp{A3}...exp{An}|ref>
        """
        new_state = self.ref * 1.0
        for k in reversed(range(0, len(parameters))):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*self.G[k]), new_state)
        return new_state
    
    
    def gradient(self,parameters):
        """ 
        """
        grad = []
        new_ket = self.prepare_state(parameters)
        new_bra = new_ket.transpose().conj()
        
        hbra = new_bra.dot(self.H)
        term = 0
        ket = cp.deepcopy(new_ket)
        grad = self.Recurse(parameters, grad, hbra, ket, term)
        self.der = grad
        return np.asarray(grad)

    def Recurse(self, parameters, grad, hbra, ket, term):
        if term == 0:
            hbra = hbra
            ket = ket
        else:
            hbra = (scipy.sparse.linalg.expm_multiply(-self.G[term-1]*parameters[term-1], hbra.transpose().conj())).transpose().conj()
            ket = scipy.sparse.linalg.expm_multiply(-self.G[term-1]*parameters[term-1], ket)
        grad.append((2*hbra.dot(self.G[term]).dot(ket).toarray()[0][0].real))
        if term<len(parameters)-1:
            term += 1
            self.Recurse(parameters, grad, hbra, ket, term)
        return np.asarray(grad)





class UCC(Variational_Ansatz):
    
    def energy(self,params):
        new_state = self.prepare_state(params)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state:
        exp{A1+A2+A3...+An}|ref>
        """
        generator = scipy.sparse.csc_matrix((self.hilb_dim, self.hilb_dim), dtype = complex)
        new_state = self.ref * 1.0
        for mat_op in range(0,len(self.G)):
            generator = generator+parameters[mat_op]*self.G[mat_op]
        new_state = scipy.sparse.linalg.expm_multiply(generator, self.ref)
        return new_state
    
    

