import openfermion
import numpy as np
import copy as cp

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n = 0
        self.G = 0
        self.w = 0

    def init(self, n, G):

        self.n = n
        self.G = G
        self.w = np.zeros([self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                temp = self.G.get_edge_data(i,j,default=0)
                if temp != 0:
                    self.w[i,j] = temp['weight']
        print(self.w)
        self.generate_SQ_Operators()

    def generate_SparseMatrix(self):
        self.cost_mat = []
        self.mixer_mat = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.cost_ops:
            self.cost_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n))
        for op in self.mixer_ops:
            self.mixer_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n))
        # print(self.cost_mat[0]) 

        self.spmat_ops = []
        for op in self.pool_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits=self.n))
        return

class qaoa(OperatorPool):
    def generate_SQ_Operators(self):

        A = QubitOperator('Z0 Z1', 0)
        B = QubitOperator('X0', 0)
        X = QubitOperator('X0', 0)
        C = QubitOperator('X0', 0)
        D = QubitOperator('Z0 Y1', 0)
        #E = QubitOperator('Z0 Y1 Z2', 0)

        
        self.pool_ops = []

        self.cost_ops = []
        self.shift = 0
        for i in range(0,self.n):
        #     #for j in range(i+1,self.n):
            for j in range(i):
                if self.w[i, j] != 0:
                    A += QubitOperator('Z%d Z%d' % (i, j), -0.5j*self.w[i, j])
                    self.shift -= 0.5*self.w[i, j]
        self.cost_ops.append(A)


        self.mixer_ops = []
        for i in range(0, self.n):
            B += QubitOperator('X%d' % i, 1j) 
        self.pool_ops.append(B)
        self.mixer_ops.append(B)

        # for i in range(0, self.n):
        #     C += QubitOperator('Y%d' % i, 1j) 
        # self.pool_ops.append(C)    

        # for i in range(0, self.n):
        #     X = QubitOperator('X%d' % i, 1j)
        #     self.pool_ops.append(X)
            
        for i in range(0,self.n):
            for j in range(i+1,self.n):
                D = QubitOperator('Z%d Y%d' % (i, j) , 1j)
                self.pool_ops.append(D)
                #D = QubitOperator('X%d Y%d' % (i, j), 1j)
                #self.pool_ops.append(D)
                #D = QubitOperator('Z%d Z%d' % (i, j), 1j)
                #self.pool_ops.append(D)
                D = QubitOperator('X%d X%d' % (i, j), 1j)
                self.pool_ops.append(D)
                #D = QubitOperator('X%d Z%d' % (i, j), 1j)
                #self.pool_ops.append(D)
                D = QubitOperator('Y%d Y%d' % (i, j), 1j)
                self.pool_ops.append(D)


        self.n_ops = len(self.pool_ops)

        return






