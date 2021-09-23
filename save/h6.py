import vqe_methods 
import operator_pools

r = 1.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r))]


vqe_methods.adapt_vqe(geometry,pool = operator_pools.singlet_SD())
