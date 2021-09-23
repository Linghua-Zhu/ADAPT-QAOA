import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys

import argparse
import hdvv
import qubit

from lib import Hamiltonian
from lib import ci_string 

import operator_pools

from qubit import *
from tVQE import *

from openfermion import *

from scipy.sparse.csgraph import dijkstra

#import igraph

#   Setup input arguments
parser = argparse.ArgumentParser(description='Run VQE',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-s','--seed', type=int, default=1, help='Should I use a random seed?', required=False)
parser.add_argument('-r','--bond_length', type=float, default=None, help='Input parameter for bond distance scan', required=False)
parser.add_argument('-t','--thresh', type=float, default=1e-3, help='How tightly to converge?', required=False)
parser.add_argument('--max_iter', type=int, default=200, help='How many iterations (operators) to converge?', required=False)
parser.add_argument('-a','--ansatz', type=str, default='pqrs', help="Which Ansatz to use",
        choices=["ijab","ijab_spinfree", "pqrs", "pqrs_spinfree", "hamiltonian"], required=False)
parser.add_argument('--sort', type=str, default=None, help="Which Ansatz ordering to use",
        choices=["AH","AH_reversed"], required=False)
parser.add_argument('--filter', type=str, default="None", help="Filter out t amplitudes based on a criterion",
        choices=["AH","None"], required=False)
parser.add_argument('-g', '--grow', type=str, default="AH", help="How to grow the ansatz",
        choices=["AH","opt1","random",'lexical','newton'], required=False)
parser.add_argument('-c', '--convergence', type=str, default="norm", help="How to decide with the ansatz is converged",
        choices=["max","rms","norm"], required=False)
parser.add_argument('--uccsd', action='store_true', help="Do un-trotterized version?", required=False)
parser.add_argument('--spin_adapt', action='store_true', help="Spin adapt excitation operators?", required=False)
args = vars(parser.parse_args())


#JW transform Hamiltonian computed classically with OFPsi4


global global_der 
global global_energy  
global global_iter  
global_der = np.array([])
global_energy = 0.0 
global_iter = 0 


basis = 'sto-3g'
multiplicity = 1
geometry = [('H', (0,0,1.5)),('H', (0, 0, 3)), ('H', (0,0,4.5)), ('H', (0, 0, 6))]
r1 = 1.5
geometry = [('H', (0,0,1*r1)), ('H', (0,0,2*r1)), ('H', (0,0,3*r1)), ('H', (0,0,4*r1)), ('H', (0,0,5*r1)), ('H', (0,0,6*r1))]
geometry = [('H', (0,0,1*r1)), ('H', (0,0,2*r1)), ('H', (0,0,3*r1)), ('H', (0,r1,1*r1)), ('H', (0,r1,2*r1)), ('H',(0,r1,3*r1))]

r = args['bond_length']
#octahedral
geometry = [('H', (0,0,+r)), ('H', (0,0,-r)), ('H', (0,+r1,0)), ('H', (0,-r,0)), ('H', (+r,0,0)), ('H', (-r,0,0))]


geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r)), ('H', (0,0,7*r)), ('H', (0,0,8*r))]




geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r))]

rlih = 2.39 * args['bond_length']
geometry = [('Li', (0,0,0)), ('H',(0,0,rlih))]

geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

rbeh2 = 1.342 * args['bond_length']
geometry = [('H', (0,0,-rbeh2)), ('Be', (0,0,0)), ('H', (0,0,rbeh2))]


molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2=0, run_cisd=0, run_ccsd = 0, run_fci=1, delete_input=0)
#molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2=1, run_cisd=1, run_ccsd = 1, run_fci=1, delete_input=0)
n_spinorbitals = int(molecule.n_orbitals*2)
print('HF energy      %20.16f au' %(molecule.hf_energy))
#print('MP2 energy     %20.16f au' %(molecule.mp2_energy))
#print('CISD energy    %20.16f au' %(molecule.cisd_energy))
#print('CCSD energy    %20.16f au' %(molecule.ccsd_energy))
print('FCI energy     %20.16f au' %(molecule.fci_energy))


#Build p-h reference and map it to JW transform
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
reference_bra = reference_ket.transpose().conj()
#JW transform Hamiltonian computed classically with OFPsi4
hamiltonian_op = molecule.get_molecular_hamiltonian()
hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
print(reference_bra.dot(hamiltonian.dot(reference_ket)))

#print(" Reference energy: %12.8f" %reference_bra.dot(hamiltonian).dot(reference_ket)[0,0].real)

#Thetas
parameters = []

#Second_quantized operations (not Jordan-Wignered)
SQ_CC_ops = []

alpha_orbs = list(range(int(n_spinorbitals/2)))
beta_orbs = list(range(int(n_spinorbitals/2)))
alpha_orbs = [ alpha_orbs[i]*2 for i in alpha_orbs]
beta_orbs = [ beta_orbs[i]*2+1 for i in beta_orbs]
print("Alpha Orbs:", alpha_orbs)
print("Beta Orbs :", beta_orbs)


n_alpha = molecule.get_n_alpha_electrons()
n_beta = molecule.get_n_beta_electrons()

alpha_occ = alpha_orbs[0:n_alpha]
alpha_vir = alpha_orbs[n_alpha::]
beta_occ = beta_orbs[0:n_beta]
beta_vir = beta_orbs[n_beta::]


print(alpha_occ, alpha_vir)
print(beta_occ, beta_vir)


#pool = operator_pools.singlet_SD(2,2)
#pool = operator_pools.singlet_SD()
pool = operator_pools.singlet_GSD()
pool.init(molecule)

'''
Count t2 second-quantized operations, add a parameter for each one, and add each one to the list
'''
#ansatz_type = "pqrs"
#ansatz_type = "ijab"
#ansatz_type = "pqrs_spinfree"
#ansatz_type = "ijab_spinfree"

ansatz_type = args['ansatz']

n_spat_orbs = int(n_spinorbitals/2)

if ansatz_type == "ijab":
    for i in alpha_occ:
        for j in alpha_occ:
            for a in alpha_vir:
                for b in alpha_vir:
                    if i>=j:
                        continue
                    if a>=b:
                        continue
                    #print(" Term %4i %4i %4i %4i" %(i,a,b,j), " V= %12.8f %12.8f" %(hamiltonian_op.two_body_tensor[i,a,b,j], hamiltonian_op.two_body_tensor[j,a,b,i]))
                    two_elec = openfermion.FermionOperator(((a,1),(i,0),(b,1),(j,0)))-openfermion.FermionOperator(((j,1),(b,0),(i,1),(a,0)))
                    if args['spin_adapt']:
                        two_elec += openfermion.FermionOperator(((a+1,1),(i+1,0),(b+1,1),(j+1,0)))-openfermion.FermionOperator(((j+1,1),(b+1,0),(i+1,1),(a+1,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
    for i in beta_occ:
        for j in beta_occ:
            for a in beta_vir:
                for b in beta_vir:
                    if args['spin_adapt']:
                        continue
                    if i>=j:
                        continue
                    if a>=b:
                        continue
                    #print(" Term %4i %4i %4i %4i" %(i,a,b,j), " V= %12.8f %12.8f" %(hamiltonian_op.two_body_tensor[i,a,b,j], hamiltonian_op.two_body_tensor[j,a,b,i]))
                    two_elec = openfermion.FermionOperator(((a,1),(i,0),(b,1),(j,0)))-openfermion.FermionOperator(((j,1),(b,0),(i,1),(a,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
    for i in alpha_occ:
        for j in beta_occ:
            for a in alpha_vir:
                for b in beta_vir:
                    #print(" Term %4i %4i %4i %4i" %(i,a,b,j), " V= %12.8f %12.8f" %(hamiltonian_op.two_body_tensor[i,a,b,j], hamiltonian_op.two_body_tensor[j,a,b,i]))
                    two_elec = openfermion.FermionOperator(((a,1),(i,0),(b,1),(j,0)))-openfermion.FermionOperator(((j,1),(b,0),(i,1),(a,0)))
                    if args['spin_adapt']:
                        if i>j:
                            continue
                        two_elec += openfermion.FermionOperator(((b-1,1),(j-1,0),(a+1,1),(i+1,0)))-openfermion.FermionOperator(((i+1,1),(a+1,0),(j-1,1),(b-1,0)))

                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)

elif ansatz_type == "pqrs":
    #aa
    pq = 0
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p>q:
                continue
            rs = 0
            for r in alpha_orbs:
                for s in alpha_orbs:
                    if r>s:
                        continue
                    if pq<rs:
                        continue
                    #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                        #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                        #continue
                    two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                    if args['spin_adapt']:
                        two_elec += openfermion.FermionOperator(((r+1,1),(p+1,0),(s+1,1),(q+1,0)))-openfermion.FermionOperator(((q+1,1),(s+1,0),(p+1,1),(r+1,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
                    rs += 1
            pq += 1

    #bb
    pq = 0
    for p in beta_orbs:
        for q in beta_orbs:
            if p>q:
                continue
            rs = 0
            for r in beta_orbs:
                for s in beta_orbs:
                    if args['spin_adapt']:
                        continue
                    if r>s:
                        continue
                    if pq<rs:
                        continue
                    #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                        #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                        #continue
                    two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
                    rs += 1
            pq += 1

    #ab
    pq = 0
    for p in alpha_orbs:
        for q in beta_orbs:
            rs = 0
            for r in alpha_orbs:
                for s in beta_orbs:
                    if pq<rs:
                        continue
                    #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                        #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                        #continue
                    two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                    if args['spin_adapt']:
                        if p>q:
                            continue
                        two_elec += openfermion.FermionOperator(((s-1,1),(q-1,0),(r+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(r+1,0),(q-1,1),(s-1,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
                    rs += 1
            pq += 1

elif ansatz_type == "ijab_spinfree":
    for p in range(0, molecule.n_electrons):
        for q in range(p+1, molecule.n_electrons):
            for a in range(molecule.n_electrons, n_spinorbitals):
                for b in range(a+1, n_spinorbitals):
                    if abs(hamiltonian_op.two_body_tensor[p,a,b,q]) < 1e-8:
                        #print(" Dropping term %4i %4i %4i %4i" %(p,a,b,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,a,b,q])
                        continue
                    two_elec = openfermion.FermionOperator(((a,1),(p,0),(b,1),(q,0)))-openfermion.FermionOperator(((q,1),(b,0),(p,1),(a,0)))
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)

elif ansatz_type == "pqrs_spinfree":
    n = n_spinorbitals
    pq = 0 
    for p in range(0, n):
        for q in range(p+1, n):
            rs = 0
            for r in range(p, n):
                for s in range(r+1, n):
                    #if pq<rs:
                    #    continue
                    two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                    if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                        print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                        continue
                    print(two_elec)
                    parameters.append(0)
                    SQ_CC_ops.append(two_elec)
                    rs += 1
            pq += 1

elif ansatz_type == "hamiltonian":
    for h in hamiltonian_op:
        val = hamiltonian_op[h]
        if abs(val) < 1e-3:
            continue

        if len(h) == 2:
            print(h)
            p = h[0][0]
            q = h[1][0]
            op = openfermion.FermionOperator(((p,1),(q,0)))-openfermion.FermionOperator(((q,1),(p,0)))
            parameters.append(0)
            SQ_CC_ops.append(op)
        if len(h) == 4:
            print(h,hamiltonian_op[h])
            p = h[0][0]
            q = h[1][0]
            r = h[2][0]
            s = h[3][0]
            op = openfermion.FermionOperator(((p,1),(q,1),(r,0),(s,0)))-openfermion.FermionOperator(((s,1),(r,1),(q,0),(p,0)))
            #op = openfermion.FermionOperator(((p,1),(r,0),(q,1),(s,0)))-openfermion.FermionOperator(((s,1),(q,0),(r,1),(p,0)))
            parameters.append(0)
            SQ_CC_ops.append(op)

do_shuffle = 0 
if do_shuffle: 
    order = list(range(len(SQ_CC_ops)))
    random.seed(args['seed'])
    random.shuffle(order)

    print(" Order: ", order)
    SQ_CC_ops = [ SQ_CC_ops[i] for i in order]

'''
Count t1 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
singles = []


if ansatz_type == "ijab":
    #aa
    for p in alpha_occ:
        for q in alpha_vir:
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            if args['spin_adapt']:
                one_elec += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
            parameters.append(0)
            singles.append(one_elec)
    #bb
    for p in beta_occ:
        for q in beta_vir:
            if args['spin_adapt']:
                continue
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            parameters.append(0)
            singles.append(one_elec)

elif ansatz_type == "pqrs":
    #aa
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p>=q:
                continue
            #if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
            #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
            #    continue
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            if args['spin_adapt']:
                one_elec += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
            parameters.append(0)
            singles.append(one_elec)
    #bb
    for p in beta_orbs:
        for q in beta_orbs:
            if args['spin_adapt']:
                continue
            if p>q:
                continue
            #if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
            #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
            #    continue
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            parameters.append(0)
            singles.append(one_elec)

elif ansatz_type == "ijab_spinfree":
    for p in range(0, molecule.n_electrons):
        for q in range(molecule.n_electrons, n_spinorbitals):
            if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
                #print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
                continue
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            parameters.append(0)
            singles.append(one_elec)

elif ansatz_type == "pqrs_spinfree":
    n = n_spinorbitals
    for p in range(0, n):
        for q in range(p+1, n):
            if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
                print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
                continue
                pass
            one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
            parameters.append(0)
            singles.append(one_elec)



if do_shuffle: 
    order = list(range(len(singles)))
    random.shuffle(order)
    print(" Order: ", order)
    singles = [ singles[i] for i in order]
SQ_CC_ops.extend(singles)

SQ_CC_ops = cp.deepcopy(pool.fermi_ops)
parameters = [0]*len(SQ_CC_ops) 


#for op in SQ_CC_ops:
#    print(op)


print(" Number of parameters: ", len(parameters))
#Jordan_Wigners into the Pauli Matrices, then computes their products as sparse matrices.
JW_CC_ops = []
for classical_op in SQ_CC_ops:
    JW_CC_ops.append(openfermion.transforms.get_sparse_operator(classical_op, n_qubits = molecule.n_qubits))
op_pool = cp.deepcopy(JW_CC_ops)
n_op_pool = len(op_pool)


#[fci_e, fci_v] = scipy.sparse.linalg.eigs(hamiltonian,1)
#print(" FCI energy: %12.8f" %fci_e[0])

precompute_AH_com = False
if precompute_AH_com:
    print(" Build commutators <[A,H]>:")
    AHcom = []
    ABHcom = np.zeros((len(JW_CC_ops), len(JW_CC_ops)))
    for opAi in range(len(JW_CC_ops)):
        opA = JW_CC_ops[opAi]
        AHc = hamiltonian.dot(opA) - opA.dot(hamiltonian)
        AHcom.append(AHc)
    #    #v = fci_v[:,0]
    #    #print(v.shape, AHc.shape)
    #    #print(type(v), type(AHc))
    #    #fci_com =  v.T.conj().dot(AHc.dot(v)) 
    #    ref_com =  reference_ket.T.conj().dot(AHc.dot(reference_ket)) 
    #    assert(ref_com.shape == (1,1))
    #    #fci_com = fci_com.real
    #    ref_com = ref_com[0,0].real
    #    #print("%12.8f, %12.8f" %(ref_com, fci_com))
    #    #print("%12.8f" %(ref_com))
    #    AHcom.append(ref_com*ref_com.conj())
    #    for opBi in range(len(JW_CC_ops)):
    #        opB = JW_CC_ops[opBi]
    #        #ABHc = AHc.dot(opB)-opB.dot(AHc)
    #        ABHc = opA.dot(opB)-opB.dot(opA)
    #        ABHc = ABHc.dot(hamiltonian)-hamiltonian.dot(ABHc)
    #        ref_com =  reference_ket.T.conj().dot(ABHc.dot(reference_ket)) 
    #        assert(ref_com.shape == (1,1))
    #        ref_com = ref_com[0,0].real
    #        print("     %12.8f" %(ref_com))
    #        ABHcom[opAi,opBi] = ref_com*ref_com
    ##        ABHcom[opBi,opAi] = ref_com*ref_com

N = len(JW_CC_ops)


## Make graph
#g = igraph.Graph()
#N = len(JW_CC_ops)
#g.add_vertices(N)
#weights = []
#labels = []
#for i in range(0,N):
#    labels.append(i+1)
#    for j in range(i+1,N):
#        print(i,j)
#        g.add_edge(i,j)
#        weights.append(ABHcom[i,j])



    
#if args['sort'] == "AH":
#    new_order = np.argsort(AHcom)
#    JW_CC_ops = [ JW_CC_ops[i] for i in new_order]
#    AHcom = [ AHcom[i] for i in new_order]
#elif args['sort'] == "AH_reversed":
#    new_order = np.argsort(AHcom)[::-1]
#    JW_CC_ops = [ JW_CC_ops[i] for i in new_order]
#    AHcom = [ AHcom[i] for i in new_order]
#
#if args['filter'] == "AH":
#    print("Filter out t amplitudes based on size of <[A,H]>")
#    new_ops = []
#    new_par = []
#    new_AH  = []
#    for c in range(len(AHcom)):
#        if abs(AHcom[c]) > 1e-8:
#            new_ops.append(JW_CC_ops[c])
#            new_par.append(parameters[c])
#            new_AH.append(AHcom[c])
#    parameters = new_par
#    JW_CC_ops = new_ops
#    AHcom = new_AH
#    print('New number of parameters: ', len(JW_CC_ops))
##distances, predecessors = dijkstra(ABHcom, return_predecessors=True)
##print(distances)


parameters_save = cp.deepcopy(parameters)
JW_CC_ops_save = cp.deepcopy(JW_CC_ops)
SQ_CC_ops_save = cp.deepcopy(SQ_CC_ops)
#JW_CC_ops = JW_CC_ops[0:1]
#parameters = parameters[0:1]
#
#
#model = tUCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)
#print(" Start optimization. Starting energy: %12.8f" %model.energy(parameters))
#opt_result = scipy.optimize.minimize(model.energy, parameters, jac=model.gradient, options = {'gtol': 1e-6, 'disp':
#    True}, method = 'BFGS', callback=model.callback)
#print(" Finished: %20.12f" % model.curr_energy)
#parameters = list(opt_result['x'])
#for p in parameters:
#    print(p)



if args['uccsd'] == True:
    uccsd = UCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)
    opt_result = scipy.optimize.minimize(uccsd.energy, parameters, options = {'gtol': 1e-6, 'disp':True}, method =
            'BFGS', callback=uccsd.callback)
    print(" Finished: %20.12f" % uccsd.curr_energy)
    parameters = opt_result['x']
    for p in parameters:
        print(p)
    exit()

if args['grow'] == "AH":
    op_indices = []
    parameters = []
    JW_CC_ops = []
    SQ_CC_ops = []
    curr_state = 1.0*reference_ket
    print(" Now start to grow the ansatz")
    max_iter = args['max_iter']
    for n_op in range(0,max_iter):
        print("\n\n\n Check each new operator for coupling")
        next_com = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
        for op_trial in range(len(op_pool)):
            
            opA = op_pool[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            opstring = ""
            for t in SQ_CC_ops_save[op_trial].terms:
                opstring += str(t)
                break
            if abs(com) > args['thresh']:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )
            next_com.append(abs(com))
            #print(" %i %20s %12.8f" %(op_trial, SQ_CC_ops[op_trial], com) )
        
        
        min_options = {'gtol': 1e-7, 'disp':False}
     
        sqrd = []
        for i in next_com:
            sqrd.append(i*i)
        norm_of_com = np.linalg.norm(np.array(next_com))
        rms_of_com = np.sqrt(np.mean(next_com))
        com = abs(com)
        max_of_com = max(next_com)
        print(" Norm of <[A,H]> = %12.8f" %norm_of_com)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)
        print(" RMS  of <[A,H]> = %12.8f" %rms_of_com)

        converged = False
        if args['convergence'] == "max":
            if max_of_com < args['thresh']:
                converged = True
        elif args['convergence'] == "norm":
            if norm_of_com < args['thresh']:
                converged = True
        elif args['convergence'] == "rms":
            if rms_of_com < args['thresh']:
                converged = True
        else:
            print(" Convergence criterion not specified")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(SQ_CC_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(SQ_CC_ops)):
                s = SQ_CC_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        next_index = next_com.index(max(next_com))
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        JW_CC_ops.insert(0,JW_CC_ops_save[next_index])
        SQ_CC_ops.insert(0,SQ_CC_ops_save[next_index])
        
        trial_model = tUCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(SQ_CC_ops)):
            s = SQ_CC_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

        


if args['grow'] == "newton":
    op_indices = []
    parameters = []
    JW_CC_ops = []
    SQ_CC_ops = []
    curr_state = 1.0*reference_ket
    print(" Now start to grow the ansatz")
    max_iter = args['max_iter']
    for n_op in range(0,max_iter):
        print("\n\n\n Check each new operator for coupling")
        next_deriv = []
        next_echange = []
        #next_deriv2 = []
        print(" Measure derivatives:")
        Hpsi = hamiltonian.dot(curr_state)
        Ecurr = curr_state.transpose().conj().dot(Hpsi)[0,0].real
        H = hamiltonian
        for op_trial in range(len(op_pool)):
           
            opA = op_pool[op_trial]
            Apsi = opA.dot(curr_state)

            # f'[x]  = 2R <HA>
            # f''[x] = 2R <HAA> - 2<AHA>
            
            fd1 = 2*(Hpsi.transpose().conj().dot(Apsi)).real
            fd2 = 2*(Hpsi.transpose().conj().dot(opA.dot(Apsi))).real + 2*Apsi.conj().transpose().dot(H.dot(Apsi))
            assert(fd1.shape == (1,1))
            assert(fd2.shape == (1,1))
            fd1 = fd1[0,0]
            fd2 = fd2[0,0]
            assert(np.isclose(fd1.imag,0))
            assert(np.isclose(fd2.imag,0))
            fd1 = fd1.real
            fd2 = fd2.real
  
            echange = 0
            if fd1 > 1e-8:
                if fd2 > 1e-12 or fd2 < -1e-12:
                    echange = -.5*fd1*fd1/fd2
            next_echange.append(echange)
            opstring = ""
            for t in SQ_CC_ops_save[op_trial].terms:
                opstring += str(t)
                break
            if abs(fd1) > args['thresh']:
                print(" %4i %40s 1st Deriv %12.8f 2nd Deriv %12.8f Predicted Energy change %12.8f" %(op_trial, opstring,
                    fd1, fd2, echange) )
            next_deriv.append(abs(fd1))
            #next_deriv2.append(abs(fd2))
        
        
        min_options = {'gtol': 1e-6, 'disp':False}
     
        sqrd = []
        for i in next_deriv:
            sqrd.append(i*i)
        norm_of_deriv = np.linalg.norm(np.array(next_deriv))
        rms_of_deriv = np.sqrt(np.mean(next_deriv))
        fd1 = abs(fd1)
        max_of_deriv = max(next_deriv)
        print(" Norm of <[A,H]> = %12.8f" %norm_of_deriv)
        print(" Max  of <[A,H]> = %12.8f" %max_of_deriv)
        print(" RMS  of <[A,H]> = %12.8f" %rms_of_deriv)
        print(" Estimate for final energy = %12.8f" %(Ecurr + sum(next_echange)))

        converged = False
        if args['convergence'] == "max":
            if max_of_deriv < args['thresh']:
                converged = True
        elif args['convergence'] == "norm":
            if norm_of_deriv < args['thresh']:
                converged = True
        elif args['convergence'] == "rms":
            if rms_of_deriv < args['thresh']:
                converged = True
        else:
            print(" Convergence criterion not specified")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(SQ_CC_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(SQ_CC_ops)):
                s = SQ_CC_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        next_index = next_echange.index(min(next_echange))
        #next_index = next_deriv.index(max(next_deriv))
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        JW_CC_ops.insert(0,JW_CC_ops_save[next_index])
        SQ_CC_ops.insert(0,SQ_CC_ops_save[next_index])
       
        trial_model = tUCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)

        ## double check derivatives with numerical
        #params_p = cp.deepcopy(parameters)
        #params_m = cp.deepcopy(parameters)
        #stepsize = 1e-5
        #params_p[0] += stepsize
        #params_m[0] -= stepsize
        #ep = trial_model.energy(params_p) 
        #e0 = trial_model.energy(parameters)
        #em = trial_model.energy(params_m)
        #d1 = (ep-em)/(2*stepsize)
        #d2 = (ep-2*e0+em)/(stepsize*stepsize)
        #print( ep, em)
        #print( "Compare 1st derivatives: %12.8f %12.8f" %(next_deriv[next_index], d1))
        #print( "Compare 2nd derivatives: %12.8f %12.8f" %(next_deriv2[next_index], d2))

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(SQ_CC_ops)):
            s = SQ_CC_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

        


if args['grow'] == "random":
    op_indices = []
    parameters = []
    JW_CC_ops = []
    SQ_CC_ops = []
    curr_state = 1.0*reference_ket
    print(" Now start to grow the ansatz")
    max_iter = args['max_iter']
    for n_op in range(0,max_iter):
        print("\n\n\n Check each new operator for coupling")
        next_index = random.randrange(len(op_pool))

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        JW_CC_ops.insert(0,JW_CC_ops_save[next_index])
        SQ_CC_ops.insert(0,SQ_CC_ops_save[next_index])
        
        trial_model = tUCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)
        

        min_options = {'gtol': 1e-6, 'disp':False}
        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(SQ_CC_ops)):
            s = SQ_CC_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

if args['grow'] == "lexical":
    op_indices = []
    parameters = []
    JW_CC_ops = []
    SQ_CC_ops = []
    curr_state = 1.0*reference_ket
    print(" Now start to grow the ansatz")
    max_iter = args['max_iter']
    for n_op in range(0,max_iter):
        print("\n\n\n Check each new operator for coupling")

        next_index = n_op % len(op_pool) 
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        JW_CC_ops.insert(0,JW_CC_ops_save[next_index])
        SQ_CC_ops.insert(0,SQ_CC_ops_save[next_index])
        
        trial_model = tUCCSD(hamiltonian,JW_CC_ops, reference_ket, parameters)
        

        min_options = {'gtol': 1e-6, 'disp':False}
        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(SQ_CC_ops)):
            s = SQ_CC_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

        



if args['grow'] == "opt1":
    op_indices = []
    parameters = []
    JW_CC_ops = []
    SQ_CC_ops = []
    print(" Now start to grow the ansatz")
    for n_op in range(0,50):
        print("\n\n\n Check each new operator for coupling")
        next_couplings = []
        next_params = []
        next_jw_ops = []
        for op_trial in range(len(JW_CC_ops_save)):
            
            print(" Trial Operator: %5i Number of Operators: %i" %( op_trial, len(parameters)+1))
            trial_params = cp.deepcopy(parameters)
            trial_jw_ops = cp.deepcopy(JW_CC_ops)
           
            trial_params.append(0)
            trial_jw_ops.append(JW_CC_ops_save[op_trial])
            
            trial_model = tUCCSD(hamiltonian,trial_jw_ops, reference_ket, trial_params)
            opt_result = scipy.optimize.minimize(trial_model.energy, trial_params, jac=trial_model.gradient, 
                    options = {'gtol': 1e-6, 'disp':True}, method = 'BFGS', callback=trial_model.callback)
            print(" Finished: %20.12f" % trial_model.curr_energy)
            next_couplings.append(trial_model.curr_energy)
            next_params.append(list(opt_result['x']))
            next_jw_ops.append(trial_jw_ops)
    
        # Sort couplings ascending
        sorted_order = np.argsort(next_couplings)
        
        update_index = sorted_order[0]
        op_indices.append(update_index)
        SQ_CC_ops.insert(0,SQ_CC_ops_save[update_index])
        JW_CC_ops = cp.deepcopy(next_jw_ops[update_index])
        parameters = cp.deepcopy(next_params[update_index])
        print(" Best Energy = %12.8f " %next_couplings[update_index])
        print(op_indices)
        
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(SQ_CC_ops)):
            s = SQ_CC_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )





