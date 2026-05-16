# January 03, 2026

## Objectives
- Understand the Variational Quantum Eigensolver (VQE).
- Minimize the expectation value of a simple Hamiltonian.

## Progress
- Implemented `vqe_simple.py`.
- Defined a 2-qubit Hamiltonian $H = 2.0 Z_0 + 0.5 X_1$.
- Used `qml.GradientDescentOptimizer` to find the ground state.
- Observed convergence of energy values over 100 steps.
