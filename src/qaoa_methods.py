import scipy
import openfermion
import networkx as nx
import os
import numpy as np
import copy
import random
import sys
import pickle
import math
from scipy.linalg import norm

import operator_pools
from tVQE import *

from openfermion import *

def run(n,
         g,
         f,
         f_mix,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa(),
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+> states: -->
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )

    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)

            sig = hamiltonian.dot(curr_state)

            next_deriv = 0

            for op_trial in range(pool.n_ops):

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))
    
                if abs(com) > abs(next_deriv) + 1e-9:
                    next_deriv = com
                    next_index = op_trial

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)

            parameters.insert(0, 0)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=trial_model.callback)
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')

        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        ## Calculate the entanglement entropy
        #entropy = trial_model.entanglement_entropy(parameters)

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', abs(GS_energy.real - trial_model.curr_energy))
        #print('Entanglement_entropy:', entropy)
        f.write("%d   %20.12f\n" % (p, (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))

    for ind in range(len(ansatz_ops)):
        print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
        print("")
        if (ind % 2) == 0:
            f_mix.write("%4s %20f %s \n\n" % (ansatz_ops[ind], parameters[ind], ind))

