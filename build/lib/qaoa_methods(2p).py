import scipy
import openfermion
import networkx as nx
import os
import numpy as np
import copy
import random
import sys
import pickle
from scipy.linalg import norm

import operator_pools
from tVQE import *

from openfermion import *


def qaoa(n,
         g,
         para,
         f,
         f_r,
         adapt_thresh=1e-5,
         theta_thresh=1e-12,
         layer = 1,
         pool=operator_pools.qaoa(),
         ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    # print('w', w)
    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+> states: -->
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
        print("                                  QAOA: ", p+1)
        print(" --------------------------------------------------------------------------")

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        ansatz_ops.insert(0, pool.mixer_ops[0])
        ansatz_mat.insert(0, pool.mixer_mat[0])

        #parameters.insert(0, random.uniform(0, 0.1))
        #parameters.insert(0, random.uniform(0, 0.1))
        #parameters.insert(0, random.uniform(-np.pi, np.pi))
        #parameters.insert(0, random.uniform(-np.pi, np.pi))
        parameters.insert(0, para)
        parameters.insert(0, para)

        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        
    #     print(" Finished: %20.12f" % trial_model.curr_energy)
    #     print(' maxcut objective:', trial_model.curr_energy + pool.shift)
    #     print(' Error:', GS_energy.real - trial_model.curr_energy)
        f.write("%d   %20.12f\n" % (p, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
    # print(" Number of operators in ansatz: ", len(ansatz_ops))
    # print(" *Finished: %20.12f" % trial_model.curr_energy)
    # print('Error:', GS_energy.real - trial_model.curr_energy)
    # print('Initia:', para)
    f_r.write("%f   %20.12f\n" % (para, (GS_energy.real - trial_model.curr_energy)/GS_energy.real))

def adapt_qaoa(n,
         g,
         para,
         f,
         f_mix,
         adapt_thresh=1e-5,
         theta_thresh=1e-12,
         layer = 1,
         pool=operator_pools.qaoa(),
         ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] * 1j
    H = pool.cost_ops[0] * 1j
    #pickle.dump(H, open('./hamiltonian.p', 'wb'))

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+++++...> states:
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )

    #Start from random states: -->
    # ket_0 = np.array([[ 1.+0.j], [ 0.+0.j]])
    # ket_1 = np.array([[ 0.+0.j], [ 1.+0.j]])
    # theta_0 = np.pi*np.random.random(n)
    # phi_0 = 2*np.pi*np.random.random(n)
    # state_r = np.zeros((n,2,1),dtype=np.complex_)
    # for i in range(0,n):
    #     state_r[i] = np.cos(0.5*theta_0[i])*ket_0 + np.exp(1.0j*phi_0[i])*np.sin(0.5*theta_0[i])*ket_1
    # temp = {} # Dynamic array
    # for i in range(0, n):
    #     temp[i] = np.zeros((2**(i+1), 1))
    # temp[0] = state_r[0]
    # #temp[1] = np.kron(state[0], state[1])
    # for i in range(1, n):
    #     temp[i] = np.kron(temp[i-1], state_r[i])
    # ini_r = temp[n-1]    

    # reference_ket = scipy.sparse.csc_matrix(ini_r)
    
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

        #x0 = {}
        #x0[0] = [0.001]
        #x0[1] = [0.001, -np.pi/4]
        #x0[2] = [0.001, -np.pi/4, 2.0]
        #x0[3] = [0.001, -np.pi/4, 2.1, -2.5]

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        parameters.insert(0, para)
        #parameters.insert(0, random.uniform(-np.pi, np.pi))

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
            assert (com.shape == (1, 1))
            com = com[0, 0]
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

        parameters.insert(0, 0.01)
        #parameters.insert(0, para)
        #parameters.insert(0, random.uniform(-np.pi, np.pi))
        ansatz_ops.insert(0, new_op)
        ansatz_mat.insert(0, new_mat)

        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        # print(" Finished: %20.12f" % trial_model.curr_energy)
        # print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        # print(' Error:', GS_energy.real - trial_model.curr_energy)
        f.write("%d   %20.12f\n" % (p, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))

    print(" Number of operators in ansatz: ", len(ansatz_ops))
    print(" *Finished: %20.12f" % trial_model.curr_energy)
    print(' Error:', (GS_energy.real - trial_model.curr_energy) / (GS_energy.real))
    #f_r.write("%f   %20.12f\n" % (para, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))

    print(" -----------Final ansatz----------- ")
    print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
    #pickle.dump(ansatz_ops, open('./ansatz.p', 'wb'))
    #pickle.dump(parameters, open('./paremeter.p', 'wb'))
    new_state = reference_ket
    E_step = []
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * ansatz_mat[k]), new_state)
        E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0, 0].real)
        print(len(parameters))
        print(k)
        print('Energy step', float(E_step[len(parameters) - k - 1]))
        print("")

    print("-------- Ansatz Operators-------")    
    for si in range(len(ansatz_ops)):
        s = ansatz_ops[si]
        opstring = ""
        for t in s.terms:
            opstring += str(t)
            break
        print("terms and parameters")    
        print(" %4s %20f %10s" % (s, parameters[si], si))
        print(" ")
        if (si % 2) == 0:
            f_mix.write("%d   %4s \n\n" % (si, s))


