import scripy
import openfermion
import networkx as nx 
import os
import numpy as np 
import copy
import random
import sys
import pickle
from openfermionprojectq import uccsd_trotter_engine, TimeEvolution
from projectq.backends import CommandPrinter

import operator_pools
from tVQE import *

from openfermion import *

def qaoa(n,
         g,
         adapt_thresh = 1e-5,
         theta_thresh = 1e-12,
         layer = 1,
         pool = operator_pools.qaoa()
        ):
   
    G = g,
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] *1j

    w, v = scripy.sparse.linalg.eigs(hamiltonian)
    GS = scripy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    # Initial States
    reference_ket = scripy.sparse.csc_matrix(
        np.full((2**n,1), 1/np.sqrt(2**n))
    )
    reference_bra = reference_ket.transpose().conj()

    # Thetas
    parameters = []

    print(" Start QAOA Algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  QAOA: ", p+1)
        print(" --------------------------------------------------------------------------")

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        ansatz_ops.insert(0, pool.mixer_ops[0])
        ansatz_mat.insert(0, pool.mixer_mat[0])

        parameters.insert(0, 1)
        parameters.insert(0, 1)

        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)
        
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print("Finished: %20.12f" % trial_model.curr_energy)
        print("Maxcut objective:", trial_model.curr_energy + pool.shift)
        print("Error:", GS_energy.real - trial_model.curr_energy)

def adapt_qaoa(n,
               g,
               adapt_thresh = 1e-5,
               theta_thresh = 1e-12,
               layer = 1,
               pool = operator_pools.qaoa(),
              ):
    G = g
    pool.init(n,G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] * 1j
     
    w, v = scipy.sparse.linalg.eigs(hamiltonian) 
    GS = scipy.sparse.csc_matrix(v[:w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:' GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    # Initial States
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    reference_bra = reference_ket.transpose().conj()

    # Thetas
    parameters = []

    print(" Start QAOA algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                           ADAPT-QAOA: ", p+1)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
     
        #Fix the cost operators
        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        parameters.insert(0, 1)

        #Adapt choose the mixer 
        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)

        sig = hamiltonian.dot(curr_state)

        print(" Check each new operator for coupling")
        group = []
        print(" Measure commutators:")

        Sign = []

        for op_trial in range(pool.n_ops):

            opA = pool.spmat_ops[op_trial]
            com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert (com.shape == (1,1))
            com = com[0,0]
            assert (np.isclose(com.imag, 0))
            com = com.real

            opstring = ""
            for t in pool.pool_ops[op_trial].terms:
                opstring += str(t)
                break

            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" % (op_trial, opstring, com))
            
            curr_norm += com * com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial     
            
        curr_norm = np.sqrt(curr_norm)
        
        min_options = {'gtol': theta_thresh, 'disp': False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" % curr_norm)
        print(" Max  of <[A,H]> = %12.8f" % max_of_com)

        new_op = pool.pool_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n] * pool.pool_ops[group[n]]
            new_mat += Sign[n] * pool.spmat_ops[group[n]]

        print(" Add operator %4i" % next_index)    

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " % n)

        parameters.insert(0, 0)
        ansatz_ops.insert(0, new_op)
        ansatz_mat.insert(0, new_mat)

        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)
        
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', GS_energy.real - trial_model.curr_energy)

    print(" Number of operators in ansatz: ", len(ansatz_ops))
    print(" *Finished: %20.12f" % trial_model.curr_energy)
    print(' Error:', GS_energy.real - trial_model.curr_energy)
    print(" -----------Final ansatz----------- ")
    print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
    new_state = reference_ket
    E_step = []
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * ansatz_mat[k]), new_state)
        E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0, 0].real)
        print(len(parameters))
        print(k)
        print('Energy step', float(E_step[len(parameters) - k - 1]))
        print("")
    for si in range(len(ansatz_ops)):
        s = ansatz_ops[si]
        opstring = ""
        for t in s.terms:
            opstring += str(t)
            break
        print(" %4s %20f %10s" % (s, parameters[si], si))
        print(" ")

        compiler_engine = uccsd_trotter_engine(compiler_backend=CommandPrinter())
        wavefunction = compiler_engine.allocate_qureg(n)

        H = 1j * s  # Qubits -pool

        # Trotter step parameters.
        time = parameters[si]

        evolution_operator = TimeEvolution(time, H)

        evolution_operator | wavefunction

        compiler_engine.flush()
        



