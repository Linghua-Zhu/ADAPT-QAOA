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
from  openfermionprojectq  import  uccsd_trotter_engine, TimeEvolution
from  projectq.backends  import CommandPrinter

import operator_pools
from tVQE import *

from openfermion import *


def qaoa(n,
         g,
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

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    # print('w', w)
    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+> states: -->
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )

    #Start from |1> states: -->
    #ini_0=np.full((2**n - 1, 1), 0)
    #ini_1=np.append(ini_0,[[1]],axis = 0)
    #reference_ket = scipy.sparse.csc_matrix(ini_1)
    #ket_1 = np.array([[ 0.+0.j], [ 1.+0.j]])
    #state_1 = np.zeros((n,2,1),dtype=np.complex_)
    #for i in range(0,n):
    #    state_1[i] = ket_1
    #temp = {} # Dynamic array
    #for i in range(0, n):
    #    temp[i] = np.zeros((2**(i+1), 1))
    #temp[0] = state_1[0]
    ##temp[1] = np.kron(state[0], state[1])
    #for i in range(1, n):
    #    temp[i] = np.kron(temp[i-1], state_1[i])
    #ini_1 = temp[n-1]    
    #reference_ket = scipy.sparse.csc_matrix(ini_1)

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
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', GS_energy.real - trial_model.curr_energy)


def q_adapt_vqe_p1(n,
         g,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
                layer = 1,
         adapt_maxiter = 100,
         pool=operator_pools.qaoa(),
         adapt_conver = "norm"
                ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0]*1j
    print(hamiltonian)

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:, w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print(w)
    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    reference_ket = scipy.sparse.csc_matrix(
        np.full((2 ** n, 1), 1 / np.sqrt(2 ** n))
    )
    reference_bra = reference_ket.transpose().conj()

    # Thetas
    parameters = []

    print(" Start ADAPT algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []
    Sign = []

    ansatz_ops.insert(0, pool.cost_ops[0])
    ansatz_mat.insert(0, pool.cost_mat[0])

    parameters.insert(0, 1)

    min_options = {'gtol': theta_thresh, 'disp': False}

    trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

    opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                         options=min_options, method='Nelder-Mead', callback=trial_model.callback)

    # print(ansatz_ops)

    parameters = list(opt_result['x'])
    curr_state = trial_model.prepare_state(parameters)

    print(" Now start to grow the ansatz")
    for n_iter in range(0, adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        group = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

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

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" % curr_norm)
        print(" Max  of <[A,H]> = %12.8f" % max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            # pickle.dump(ansatz_ops, open('./h4_ansatz.p', 'wb'))
            print(" Ansatz Growth Converged!")
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

            break

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

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        # print(ansatz_ops)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0, 0]
        overlap = overlap.real
        overlap = overlap * overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' Error:', GS_energy.real - trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4s %20f %10s" % (s, parameters[si], si))
            print(" ")

def q_adapt_vqe(n,
         g,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
                layer = 1,
         adapt_maxiter = 100,
         pool=operator_pools.qaoa(),
         adapt_conver = "norm"
                ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0]*1j

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:, w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    reference_ket = scipy.sparse.csc_matrix(
        np.full((2 ** n, 1), 1 / np.sqrt(2 ** n))
    )
    reference_bra = reference_ket.transpose().conj()

    # Thetas
    parameters = []

    print(" Start ADAPT algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []
    Sign = []

    print(" Now start to grow the ansatz")
    for n_iter in range(0, adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        group = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

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

            if abs(com) > abs(next_deriv) + 1e-15:
                next_deriv = com
                next_index = op_trial

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp': False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" % curr_norm)
        print(" Max  of <[A,H]> = %12.8f" % max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            # pickle.dump(ansatz_ops, open('./h4_ansatz.p', 'wb'))
            print(" Ansatz Growth Converged!")
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

            break

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

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        # print(ansatz_ops)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0, 0]
        overlap = overlap.real
        overlap = overlap * overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' Error:', GS_energy.real - trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4s %20f %10s" % (s, parameters[si], si))
            print(" ")


def q_adapt_vqe_min(n,
         g,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
                layer = 1,
         adapt_maxiter = 100,
         pool=operator_pools.qaoa(),
         adapt_conver = "norm"
                ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0]*1j

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:, w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real)

    reference_ket = scipy.sparse.csc_matrix(
        np.full((2 ** n, 1), 1 / np.sqrt(2 ** n))
    )
    reference_bra = reference_ket.transpose().conj()

    E = reference_bra.dot(hamiltonian.dot(reference_ket)).real
    E = E[0,0]

    # Thetas
    parameters = []

    print(" Start ADAPT algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []
    Sign = []

    min_options = {'gtol': theta_thresh, 'disp': False}

    print(" Now start to grow the ansatz")
    for n_iter in range(0, adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        group = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

        for op_trial in range(pool.n_ops):

            ansatz_ops_trial = []
            ansatz_mat_trial = []
            parameters_trial = []

            ansatz_ops_trial.append(pool.pool_ops[op_trial])
            ansatz_mat_trial.append(pool.spmat_ops[op_trial])
            parameters_trial.insert(0,1)
            trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, curr_state, parameters_trial)

            opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient,
                                                 options=min_options, method='Nelder-Mead',
                                                 callback=trial_model.callback)

            dE = abs(E-trial_model.curr_energy)

            opstring = ""
            for t in pool.pool_ops[op_trial].terms:
                opstring += str(t)
                break

            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" % (op_trial, opstring, dE))

            curr_norm += dE * dE

            if abs(dE) > abs(next_deriv) + 1e-9:
                next_deriv = dE
                next_index = op_trial

        curr_norm = np.sqrt(curr_norm)

        max_of_dE = next_deriv
        print(" Norm of <[A,H]> = %12.8f" % curr_norm)
        print(" Max  of <[A,H]> = %12.8f" % max_of_dE)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            # pickle.dump(ansatz_ops, open('./h4_ansatz.p', 'wb'))
            print(" Ansatz Growth Converged!")
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

            break

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

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        # print(ansatz_ops)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0, 0]
        overlap = overlap.real
        overlap = overlap * overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' Error:', GS_energy.real - trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" % ("Term", "Coeff", "#"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4s %20f %10s" % (s, parameters[si], si))
            print(" ")


def adapt_qaoa(n,
         g,
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

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+> states: -->
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )

    #Start from |1> states: -->
    # ini_0=np.full((2**n - 1, 1), 0)
    # ini_1=np.append(ini_0,[[1]],axis = 0)
    # reference_ket = scipy.sparse.csc_matrix(ini_1)

 
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

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        parameters.insert(0, 1)

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