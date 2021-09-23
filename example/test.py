import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 4
p = 6
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.6), (0, 3, 0.7), (0, 2, 0.5), (1, 3, 0.2)]
g.add_weighted_edges_from(elist)

filename = 'error' + '.txt'
filename_1 = 'mix_operator' + '.txt'
f = open(filename, "a")
f_mix = open(filename_1, "a")
qaoa_methods.run(n, 
	             g, 
	             f,
	             f_mix,
	             adapt_thresh=1e-10, 
	             theta_thresh=1e-7,
	             layer=p, 
	             pool=operator_pools.qaoa(), 
	             init_para=0.01, 
	             structure = 'qaoa', 
	             selection = 'grad', 
	             rand_ham = 'False', 
	             opt_method = 'NM',
	             landscape = False,
	             landscape_after = False,
	             resolution = 100)
f.close()
f_mix.close()