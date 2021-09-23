# ADAPT-QAOA

The fixed form of the original QAOA ansatz is not optimal, thus we developed an iterative version of QAOA, which is problem-tailored. 
--> We kept the structure and the cost Hamiltonian of QAOA but built up the ansatz iteratively, one layer at a time. 
--> The mixer operator, which unlike the original QAOA here could be an entangling gate, is dictated by the problem Hamiltonian. 
--> We showed that our algorithm, ADAPT-QAOA, performed better than the original QAOA (we tested on Maxcut problem).

Read More about the ADAPT-QAOA work: https://arxiv.org/abs/2005.10258

# Quantum Entanglement in ADAPT-QAOA

What make the advantage for quantum computing over classical computation? 
Coherence? Entanglement?

If yes, what is the role for entanglement? 
Let's explore more...

# Set up code
The code modified from ADAPT-VQE's code, method detailed in Nature Communications, 10, 3007, (2019)
Install method is the same as ADAPT-VQE

Before install the code, create a virtual environment is recommended.
