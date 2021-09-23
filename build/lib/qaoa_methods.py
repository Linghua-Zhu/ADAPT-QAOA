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
         f_overlap,
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

    ## Calculate the overlap
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

    print('hamiltonian:',hamiltonian)
    print('deg_manifold_length:', len(deg_manifold))


    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('GS_1', np.reshape(deg_manifold[0], (-1, 2**n)))
    print('GS_2', np.reshape(deg_manifold[1], (-1, 2**n)))
    # print("GS: ", GS.transpose())
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

        if selection == 'min_1p':
            
            next_deriv = 0

            for op_trial in range(pool.n_ops):

                trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
                curr_state = trial_model.prepare_state(parameters)

                ansatz_ops_trial = []
                ansatz_mat_trial = []
                parameters_trial = []
    
                ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
                ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                parameters_trial.insert(0, init_para)
                trial_model_1 = tUCCSD(hamiltonian, ansatz_mat_trial, curr_state, parameters_trial)

                if opt_method == 'NM':    
                    opt_result = scipy.optimize.minimize(trial_model_1.energy, parameters_trial,
                                                         method='Nelder-Mead')
                if opt_method == 'BFGS':
                	opt_result = scipy.optimize.minimize(trial_model_1.energy, parameters_trial,  
                                           options = min_options, method = 'BFGS')
                if opt_method == 'Cobyla':
                    opt_result = scipy.optimize.minimize(trial_model_1.energy, parameters_trial, 
                                           method = 'Cobyla')
    
                parameters_trial = list(opt_result['x'])
    
                dE = E-trial_model_1.curr_energy
    
                # if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))
    
                if dE > abs(next_deriv) + 1e-9:
                    next_deriv = dE
                    next_index = op_trial
                    parameters_cand = parameters_trial.copy()
    
            max_of_dE = next_deriv
            print(" Max  of dE = %12.8f" % max_of_dE)

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            
            parameters.insert(0, parameters_cand[0])
            print(parameters)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)
        if selection == 'min_2p':

            next_deriv = 0

            for op_trial in range(pool.n_ops):

                ansatz_ops_trial = []
                ansatz_mat_trial = []
                parameters_trial = []
    
                ansatz_ops_trial.insert(0, pool.cost_ops[0])
                ansatz_mat_trial.insert(0, pool.cost_mat[0])
                parameters_trial.insert(0, init_para)
    
                ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
                ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                parameters_trial.insert(0, init_para)
                trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, curr_state, parameters_trial)
    
                if opt_method == 'NM':    
                    opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial,
                                               method='Nelder-Mead')
                if opt_method == 'BFGS':
                	opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient, 
                                           options = min_options, method = 'BFGS', callback=trial_model.callback)
                if opt_method == 'Cobyla':
                    opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, 
                                           method = 'Cobyla')
    
                parameters_trial = list(opt_result['x'])
    
                dE = abs(E-trial_model.curr_energy)
    
                # if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))
    
                if dE > abs(next_deriv) + 1e-9:
                    next_deriv = dE
                    next_index = op_trial
                    parameters_cand = parameters_trial.copy()
    
            max_of_dE = next_deriv
            print(" Max  of dE = %12.8f" % max_of_dE)

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            
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

            parameters[0] = parameters_cand[1]
            parameters.insert(0, parameters_cand[0])

        if selection == 'min_np':

            next_deriv = 0

            for op_trial in range(pool.n_ops):

                ansatz_ops_trial = ansatz_ops.copy()
                ansatz_mat_trial = ansatz_mat.copy()
                parameters_trial = parameters.copy()
    
                ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
                ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                parameters_trial.insert(0, init_para)
                trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, reference_ket, parameters_trial)
    
                if opt_method == 'NM':    
                    opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial,
                                               method='Nelder-Mead')
                if opt_method == 'BFGS':
                	opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient, 
                                           options = min_options, method = 'BFGS', callback=trial_model.callback)
                if opt_method == 'Cobyla':
                    opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, 
                                           method = 'Cobyla')
    
                parameters_trial = list(opt_result['x'])
    
                dE = abs(E-trial_model.curr_energy)
    
                # if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))
    
                if dE > abs(next_deriv) + 1e-9:
                    next_deriv = dE
                    next_index = op_trial
                    parameters_cand = parameters_trial.copy()
    
            max_of_dE = next_deriv
            print(" Max  of dE = %12.8f" % max_of_dE)

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            
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

            parameters = parameters_cand.copy()

        if selection == 'hybrid':

            if p == 0:
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

            if p != 0:
                next_deriv = 0
    
                for op_trial in range(pool.n_ops):
    
                    ansatz_ops_trial = ansatz_ops.copy()
                    ansatz_mat_trial = ansatz_mat.copy()
                    parameters_trial = parameters.copy()
        
                    ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
                    ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                    parameters_trial.insert(0, init_para)
                    trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, reference_ket, parameters_trial)
        
                    if opt_method == 'NM':    
                        opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial,
                                                   method='Nelder-Mead')
                    if opt_method == 'BFGS':
                    	opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient, 
                                               options = min_options, method = 'BFGS', callback=trial_model.callback)
                    if opt_method == 'Cobyla':
                        opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, 
                                               method = 'Cobyla')
        
                    parameters_trial = list(opt_result['x'])
        
                    dE = abs(E-trial_model.curr_energy)
        
                    # if abs(com) > adapt_thresh:
                    print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))
        
                    if dE > abs(next_deriv) + 1e-9:
                        next_deriv = dE
                        next_index = op_trial
                        parameters_cand = parameters_trial.copy()
        
                max_of_dE = next_deriv
                print(" Max  of dE = %12.8f" % max_of_dE)
    
                new_op = pool.pool_ops[next_index]
                new_mat = pool.spmat_ops[next_index]
        
                print(" Add operator %4i" % next_index)
                
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
    
                parameters = parameters_cand.copy()

        # trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        # if opt_method == 'NM':    
        #     opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
        #                                                  method='Nelder-Mead', callback=trial_model.callback)
        # if opt_method == 'BFGS':
        #     opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
        #                                                     options = min_options, method = 'BFGS', callback=trial_model.callback)
        # if opt_method == 'Cobyla':
        #     opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
        #                                           method = 'Cobyla')

        # parameters = list(opt_result['x'])
        # fin_energy = trial_model.curr_energy.copy()

        if selection == 'other_np':

            if p == 0:
                next_index = 11
    
                new_op = pool.pool_ops[next_index]
                new_mat = pool.spmat_ops[next_index]
        
                print(" Add operator %4i" % next_index)
    
                parameters.insert(0, 0)
                ansatz_ops.insert(0, new_op)
                ansatz_mat.insert(0, new_mat)

            if p != 0:
                next_deriv = 0
    
                for op_trial in range(pool.n_ops):
    
                    ansatz_ops_trial = ansatz_ops.copy()
                    ansatz_mat_trial = ansatz_mat.copy()
                    parameters_trial = parameters.copy()
        
                    ansatz_ops_trial.insert(0, pool.pool_ops[op_trial])
                    ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                    parameters_trial.insert(0, init_para)
                    trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, reference_ket, parameters_trial)
        
                    if opt_method == 'NM':    
                        opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial,
                                                   method='Nelder-Mead')
                    if opt_method == 'BFGS':
                        opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient, 
                                               options = min_options, method = 'BFGS', callback=trial_model.callback)
                    if opt_method == 'Cobyla':
                        opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, 
                                               method = 'Cobyla')
        
                    parameters_trial = list(opt_result['x'])
        
                    dE = abs(E-trial_model.curr_energy)
        
                    # if abs(com) > adapt_thresh:
                    print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], dE))
        
                    if dE > abs(next_deriv) + 1e-9:
                        next_deriv = dE
                        next_index = op_trial
                        parameters_cand = parameters_trial.copy()
        
                max_of_dE = next_deriv
                print(" Max  of dE = %12.8f" % max_of_dE)
    
                new_op = pool.pool_ops[next_index]
                new_mat = pool.spmat_ops[next_index]
        
                print(" Add operator %4i" % next_index)
                
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
    
                parameters = parameters_cand.copy()

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

        # Consider degenerate calculate overlap
        curr_state = trial_model.prepare_state(parameters)

        print("GS:", GS.transpose().conj())
        print("Current State: ", curr_state)
 
        # overlap = GS.transpose().conj().dot(curr_state)[0, 0]
        # overlap = overlap.real
        # overlap = overlap * overlap
        # print("The Overlap:", overlap)

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

