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

#from openfermion import *

def qaoa(n,
         g,
         para,
         f,
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
        parameters.insert(0, para)

        ansatz_ops.insert(0, pool.mixer_ops[0])
        ansatz_mat.insert(0, pool.mixer_mat[0])

        #parameters.insert(0, random.uniform(0, 0.1))
        #parameters.insert(0, random.uniform(0, 0.1))
        #parameters.insert(0, random.uniform(-np.pi, np.pi))
        #parameters.insert(0, random.uniform(-np.pi, np.pi))
        parameters.insert(0, 0)
        

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
    #f_r.write("%f   %20.12f\n" % (para, (GS_energy.real - trial_model.curr_energy)/GS_energy.real))

def adapt_qaoa_np_min(n,
         g,
         para,
         f,
         f_r,
         f_mix,
         adapt_thresh=1e-5,
         theta_thresh=1e-12,
                layer = 1,
         adapt_maxiter = 50,
         pool=operator_pools.qaoa(),
         adapt_conver = "norm"
                ):
    # {{{
    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0]*1j
    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:, w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    reference_ket = scipy.sparse.csc_matrix(
        np.full((2 ** n, 1), 1 / np.sqrt(2 ** n))
    )
    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas
    parameters = []
    print(" Start ADAPT algorithm")
    curr_state = 1.0 * reference_ket
    ansatz_ops = []
    ansatz_mat = []
    Sign = []
    min_options = {'gtol': theta_thresh, 'disp': False}
    print(" Now start to grow the ansatz")
    for p in range(0, layer):
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-QAOA iteration: ", p)
        print(" --------------------------------------------------------------------------")

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])
        parameters.insert(0, para)

        next_index = None
        next_deriv = 0

        for op_trial in range(pool.n_ops):

            ansatz_ops_trial = []
            ansatz_mat_trial = []
            parameters_trial = []

            ansatz_ops_trial.insert(0, pool.cost_ops[0])
            ansatz_mat_trial.insert(0, pool.cost_mat[0])
            parameters_trial.insert(0, para)
            
            trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, curr_state, parameters_trial)
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial,
                                               method='Nelder-Mead')

            parameters_trial = list(opt_result['x'])

            ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
            ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
            parameters_trial.insert(0, para)

            dE = abs(E-trial_model.curr_energy)
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))

            if abs(dE) > abs(next_deriv) + 1e-9:
                next_deriv = dE
                next_index = op_trial
                parameters_cand = parameters_trial

        max_of_dE = next_deriv
        print(" Max of dE = %12.8f" % max_of_dE)

        new_op = pool.pool_ops[next_index]
        new_mat = pool.spmat_ops[next_index]
        print(" Add operator %4i" % next_index)

        parameters[0] = para
        parameters.insert(0, parameters_cand[0])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)


        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' Error:', GS_energy.real - trial_model.curr_energy)
        f.write("%d   %20.12f\n" % (p, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")

    print(" Number of operators in ansatz: ", len(ansatz_ops))
    print(" *Finished: %20.12f" % trial_model.curr_energy)
    print(' Error:', (GS_energy.real - trial_model.curr_energy) / (GS_energy.real))
    f_r.write("%f   %20.12f\n" % (para, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))

    # print(final_state)
    print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
    for si in range(len(ansatz_ops)):
        s = ansatz_ops[si]
        print(" %4s %20f %10s" % (s, parameters[si], si))
        print(" ")
        if (si % 2) == 0:
            f_mix.write("%4s %20f \n\n" % (s, parameters[si]))

            


