import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 

#Manually initialize state
basis = 'sto-3g'
multiplicity = 1
geometry = [('H', (0,0,1.5)),('H', (0, 0, 3)), ('H', (0,0,4.5)), ('H', (0, 0, 6))]
geometry = [('H', (0,0,1.5)),('H', (0, 0, 3)), ('H', (0,0,4.5)), ('H', (0, 0, 6)), ('H', (0, 0, 7.5)), ('H', (0, 0, 9))]
r1 = 1.5
geometry = [('H', (0,0,1*r1)), ('H', (0,0,2*r1)), ('H', (0,0,3*r1)), ('H', (0,0,4*r1)), ('H', (0,0,5*r1)), ('H', (0,0,6*r1)), ('H', (0,0,7*r1)), ('H', (0,0,8*r1))]
molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_mp2=1, run_cisd=1, run_ccsd = 1, run_fci=1, delete_input=0)
n_spinorbitals = int(molecule.n_orbitals*2)
print('HF energy      %20.16f au' %(molecule.hf_energy))
print('MP2 energy     %20.16f au' %(molecule.mp2_energy))
print('CISD energy    %20.16f au' %(molecule.cisd_energy))
print('CCSD energy    %20.16f au' %(molecule.ccsd_energy))
print('FCI energy     %20.16f au' %(molecule.fci_energy))

global global_der 
global global_energy  
global global_iter  
global_der = np.array([])
global_energy = 0.0 
global_iter = 0 

#Build p-h reference and map it to JW transform
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
reference_bra = reference_ket.transpose().conj()

#JW transform Hamiltonian computed classically with OFPsi4
hamiltonian = openfermion.transforms.get_sparse_operator(molecule.get_molecular_hamiltonian())

#Thetas
parameters = []

#Second_quantized operations (not Jordan-Wignered)
SQ_CC_ops = []


'''
Count t2 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
for p in range(0, molecule.n_electrons):
    for q in range(p+1, molecule.n_electrons):
        for a in range(molecule.n_electrons, n_spinorbitals):
            for b in range(a+1, n_spinorbitals):
                 two_elec = openfermion.FermionOperator(((a,1),(p,0),(b,1),(q,0)))-openfermion.FermionOperator(((q,1),(b,0),(p,1),(a,0)))
                 parameters.append(0)
                 SQ_CC_ops.append(two_elec)

'''
Count t1 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
for p in range(0, molecule.n_electrons):
    for q in range(molecule.n_electrons, n_spinorbitals):
        one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
        parameters.append(0)
        SQ_CC_ops.append(one_elec)

print(" Number of parameters: ", len(parameters))
#Jordan_Wigners into the Pauli Matrices, then computes their products as sparse matrices.
JW_CC_ops = []
for classical_op in SQ_CC_ops:
    JW_CC_ops.append(openfermion.transforms.get_sparse_operator(classical_op, n_qubits = molecule.n_qubits))
'''
SPE based on a traditional, untrotterized ansatz
v'=exp(a+b+...+n)v
'''
def SPE(parameters):
    generator = scipy.sparse.csc_matrix((2**(molecule.n_qubits), 2**(molecule.n_qubits)), dtype = complex)
    for mat_op in range(0,len(JW_CC_ops)):
        generator = generator+parameters[mat_op]*JW_CC_ops[mat_op]
    new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    return energy.toarray()[0][0].real

'''
SPE based on full, 1st-order Trotter decomposition
v'=exp(a)exp(b)...exp(n)v
'''
def Trotter_SPE(parameters):
    global global_energy  
    new_state = reference_ket
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    global_energy = energy.toarray()[0][0].real
    assert(global_energy.imag <  1e-14)
    global_energy = global_energy.real
    return global_energy 


                 
                
#Numerical trotterized gradient
def Numerical_Trot_Grad(parameters):
    global global_der
    step_size = 1e-6
    grad = []
    for k in reversed(range(0, len(parameters))):
        para = copy.copy(parameters)
        para[k]+=step_size
        diff = Trotter_SPE(para)
        para[k]-=2*step_size
        diff -= Trotter_SPE(para)
        grad.append(diff/(step_size*2))
    global_der = np.asarray(grad)
    return np.asarray(grad)

def Five_Point_Grad(parameters):
    grad = []
    for k in reversed(range(0, len(parameters))):
        forw = copy.copy(parameters)
        forw2 = copy.copy(parameters)
        reve = copy.copy(parameters)
        reve2 = copy.copy(parameters)
        forw[k]+=1e-7
        forw2[k]+=2e-7
        reve[k]-=1e-7
        reve2[k]-=2e-7
        f2 = Trotter_SPE(forw2)
        f1 = Trotter_SPE(forw)
        r1 = Trotter_SPE(reve)
        r2 = Trotter_SPE(reve2)
        diff = (-f2+8*f1-8*r1+r2)/(1.2e-6)
        grad.append(diff)
    return np.asarray(grad)

#Analytical trotter gradient
def Trotter_Gradient(parameters):
    global global_der 
    grad = []
    new_state = copy.copy(reference_ket)
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    hbra = new_bra.dot(hamiltonian)
    #H_l = hamiltonian
    #H_r = hamiltonian
    term = 0
    ket = copy.copy(new_state)
    grad = Recurse(parameters, grad, hbra, ket, term)
    global_der = grad
    return np.asarray(grad)

#Recursive component of analytical trotter gradient
def Recurse(parameters, grad, hbra, ket, term):
    if term == 0:
        hbra = hbra
        ket = ket
    else:
        hbra = (scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], hbra.transpose().conj())).transpose().conj()
        ket = scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], ket)
    grad.append((2*hbra.dot(JW_CC_ops[term]).dot(ket).toarray()[0][0].real))
    #grad.insert(0,(2*hbra.dot(JW_CC_ops[term]).dot(ket).toarray()[0][0].real))
    if term<len(parameters)-1:
        term += 1
        Recurse(parameters, grad, hbra, ket, term)
    return np.asarray(grad)

def callback(parameters):
    global global_der 
    global global_energy  
    global global_iter 
    err = np.sqrt(np.vdot(global_der, global_der))
    print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(global_iter,
        global_energy, err, np.max(np.abs(global_der))))
    global_iter += 1


#for p in range(len(parameters)):
#    parameters[p] = (random.random()-.5)*.001

#der_num = Numerical_Trot_Grad(parameters)
#der_ana = Trotter_Gradient(parameters)
#print(" Numerical: ")
#print(der_num)
#print("\n Analytical: ")
#print(der_ana)
#print("\n Error: ")
#print(np.linalg.norm(der_num-der_ana))

scipy.optimize.minimize(Trotter_SPE, parameters, jac=Trotter_Gradient, options = {'gtol': 1e-7, 'disp': True}, method = 'BFGS', callback=callback)
