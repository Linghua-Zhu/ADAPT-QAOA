import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import scipy.sparse
import scipy.sparse.linalg


class Qubit:
    def __init__(self,ind):
        self.index      = ind
        self.hilb_dim   = 2
        assert(self.hilb_dim == 2)
        self.ops = {}
        self.build_operators()


    def __str__(self):
        return " Q%-4i Size of Hilbert Space=%-12i\n" %(self.index,self.hilb_dim)
    
    def dim(self):
        return self.hilb_dim 
     
    def build_operators(self):
        x = np.array([[0,1.],[1.,0]])
        y = np.array([[0,(0-1j)],[(0+1j),0]])
        z = np.array([[1,0],[0,-1]])
        I = np.array([[1,0],[0,1]])
        sp = (x + y*(0+1j))/2.0
        sm = (x - y*(0+1j))/2.0
        n = sp.dot(sm)
      
        self.I = QubitOperator("I",I)
        self.N = QubitOperator("N",n)
        self.X = QubitOperator("X",x)
        self.Y = QubitOperator("Y",y)
        self.Z = QubitOperator("Z",z)
        self.P = QubitOperator("+",sp)
        self.M = QubitOperator("-",sm)
        self.ops["I"] = QubitOperator("I",I)
        self.ops["X"] = QubitOperator("X",x)
        self.ops["Y"] = QubitOperator("Y",y)
        self.ops["Z"] = QubitOperator("Z",z)
        self.ops["P"] = QubitOperator("+",sp)
        self.ops["M"] = QubitOperator("-",sm)
        self.ops["N"] = QubitOperator("N",n)



class QubitOperator:
    """
    Pauli operator which acts on a qubit
    """
    def __init__(self,_name, _mat):
        """
        Construct a new 'QubitOperator' object.

        :param _name: The string used to denote the Pauli operator, i.e. {X,Y,Z,I,+,-} 
        :param _mat: 2x2 Numpy NDArray 
        :return: returns nothing
        """
        self.name = cp.deepcopy(_name)
        self.mat  = 1.0*_mat    

    def __str__(self):
        return self.name 



class QubitLattice:
    def __init__(self,_n):
        self.n_qubits = cp.deepcopy(_n)
        self.qubits = []
        self.hilb_dim = 1

        for s in range(self.n_qubits):
            
            self.qubits.append(Qubit(s))
            self.hilb_dim *= 2 

    def __str__(self):
        return " QubitLattice: #qubits=%-4i Size of Hilbert Space=%12i" %(self.n_qubits,self.hilb_dim)



class OperatorString:
    def __init__(self,l):
        self.lattice    = cp.deepcopy(l)
        self.ops        = []   # list of QubitOperators initialized to I
        self.n_qubits   = self.lattice.n_qubits
        self.coeff      = 1.0
        for qi in range(self.n_qubits):
            self.ops.append(self.lattice.qubits[qi].I)
        self.affects = set()
        self.mat_formed = False 

    def update_operator(self,qi,op):
        """
        qi is the qubit index
        op is a string either "X,Y,Z,I,P,M" 
        """
        self.ops[qi] = cp.deepcopy(self.lattice.qubits[qi].ops[op])
        if op != "I":
            self.affects.add(qi)
        elif op == "I":
            self.affects.discard(qi)

    def __str__(self):
        string = ""
        for qi in range(self.n_qubits):
            string += str(self.ops[qi])
            if qi < self.n_qubits-1:
                string += "*"
        if np.isclose(self.coeff.imag,0):
            string += "%20.16f"%self.coeff.real
        else:
            string += "%20.16f %20.16fj"%(self.coeff.real,self.coeff.imag)
        return string
    
    def form_matrix(self):

        if self.n_qubits < 1:
                return
        if self.mat_formed == True:
            return
        self.mat = self.ops[0].mat
        for qi in range(1,self.n_qubits):
            self.mat = np.kron(self.mat,self.ops[qi].mat)
        self.mat.shape = (self.lattice.hilb_dim, self.lattice.hilb_dim) 
        assert(np.isreal(self.mat.all))
        self.mat = self.mat.real
        self.mat_formed == True

    def clear_matrix(self):

        if self.n_qubits < 1:
                return
        self.mat = np.array(())        

    def apply_to_state(self,state):
        """
        Apply an operator to a state:
        params: State instance
        returns: new State instance, new = self | State
        """
        new = state.copy()
        for qi in range(self.n_qubits):
            if self.ops[qi].name == "I":
                continue

            elif self.ops[qi].name == "Z":
                idx = [slice(None)]*(qi)
                idx.extend([slice(1,None,None), Ellipsis])
                new.v[tuple(idx)] *= -1

            else:
                A = self.ops[qi].mat 
                new.v = new.v.swapaxes(qi,0)
                new.v = np.tensordot(A, new.v, axes=(1,0)) 
                new.v = new.v.swapaxes(qi,0)
        new.v *= self.coeff

        return new
    
    def exp_val(self,state):
        state.v.shape = self.lattice.hilb_dim
        ev = np.vdot(state.v, np.dot(self.mat,state.v))
        state.v.shape = state.shape
        return ev



class State:
    def __init__(self,l):
        self.lattice = l
        self.shape = []
        for qi in range(self.lattice.n_qubits):
            self.shape.append(2)
        self.v = np.zeros(self.shape, dtype='complex128')

    def __len__(self):
        return self.lattice.hilb_dim
    
    def __iadd__(self,other):
        self.v += other.v
        return self
    
    def dot(self,other):
        return np.vdot(self.v,other.v) 
    
    def assign(self,vin):
        try:
            self.v = cp.deepcopy(vin)
            self.v.shape = cp.deepcopy(self.shape)
        except:
            print(" Problem with the shapes")
            print(self.v.shape, vin.shape)
            print(len(vin),len(self.v))
            raise
    def copy(self):
        new = State(self.lattice)
        new.assign(self.v)
        return new
    
    def fold(self):
        """
        Index state as a tensor
        """
        self.v.shape = self.shape
    
    def unfold(self):
        """
        Index state as a vector 
        """
        self.v = self.v.reshape(self.lattice.hilb_dim)
    def normalize(self):
        """
        Normalize state
        """
        self.v = self.v / self.dot(self)


    def set_to_hf(self,na,nb):
        """
        Set state to Hartree-Fock state
        params: na = number of alpha electrons
        params: nb = number of beta electrons
        returns: 
        """
        self.v = np.zeros(self.lattice.hilb_dim, dtype='complex128')
        self.v[0] = 1
        #self.v[self.lattice.hilb_dim-1] = 1
        self.fold()

        # Form HF operator
        for i in range(na):
            x = OperatorString(self.lattice)
            x.update_operator(2*i,"X")
            self.v = x.apply_to_state(self).v
        for i in range(nb):
            x = OperatorString(self.lattice)
            x.update_operator(2*i+1,"X")
            self.v = x.apply_to_state(self).v
   
        #self.v += .001*(np.random.rand(self.v.shape[0])-.5)/self.v.shape[0]
        self.v = self.v / np.linalg.norm(self.v) # just make sure norm is tight

    
    def set_to_hf_not_spin_staggered(self,na,nb):
        """
        Set state to Hartree-Fock state
        params: na = number of alpha electrons
        params: nb = number of beta electrons
        returns: 
        """
        self.v = np.zeros(self.lattice.hilb_dim, dtype='complex128')
        self.v[0] = 1
        #self.v[self.lattice.hilb_dim-1] = 1
        va = np.array([1.0,0.0])
        vb = np.array([0.0,1.0])
        v = vb*1
        # Form HF operator
        for i in range(na+nb+1,self.lattice.n_qubits):
            v = np.kron(v,vb)
        for i in range(na+nb):
            v = np.kron(v,va)

        self.v = v*1.0
    
    def set_to_neel(self,na,nb):
        """
        Set state to Hartree-Fock state
        params: na = number of alpha electrons
        params: nb = number of beta electrons
        returns: 
        """
        self.v = np.zeros(self.lattice.hilb_dim, dtype='complex128')
        self.v[self.lattice.hilb_dim-1] = 1
        self.fold()

        # Form HF operator
        for i in range(na):
            x = OperatorString(self.lattice)
            x.update_operator(2*i,"X")
            self.v = x.apply_to_state(self).v
        for i in range(nb):
            x = OperatorString(self.lattice)
            x.update_operator(2*na + 2*i+1,"X")
            self.v = x.apply_to_state(self).v
    
        self.v = self.v / np.linalg.norm(self.v) # just make sure norm is tight

    
    def apply_operator_sum(self, op):
        """
        Apply a sum of operator strings to state
        params: op a list of OperatorString objects
        returns: a new state
        """
        new = State(self.lattice)
        for t in op:
            new += t.apply_to_state(self)
        return new


    def exp_val(self, op_sum):
        """
        Compute expectation value 
        params: op_sum = list of OperatorString objects
        returns: double
        """
        ev = self.dot(self.apply_operator_sum(op_sum))
        if np.isclose(ev.imag, 0):
            return ev.real
        else:
            return ev
