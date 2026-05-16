import pennylane as qml
from pennylane import numpy as np

# January 03, 2026: Variational Quantum Eigensolver (VQE)
# Finding the ground state energy of a simple Hamiltonian

def main():
    dev = qml.device("default.qubit", wires=2)

    # Define a simple Hamiltonian: H = 2.0*Z0 + 0.5*X1
    coeffs = [2.0, 0.5]
    obs = [qml.PauliZ(0), qml.PauliX(1)]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(hamiltonian)

    # Initial parameters
    params = np.array([0.1, 0.1], requires_grad=True)

    # Optimization using Gradient Descent
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    steps = 100

    print("Starting VQE optimization...")
    for i in range(steps):
        params = opt.step(circuit, params)
        if (i + 1) % 20 == 0:
            energy = circuit(params)
            print(f"Step {i+1:3d}: Energy = {energy:.6f}")

    print(f"\nOptimized Parameters: {params}")
    print(f"Ground State Energy: {circuit(params):.6f}")
    # Theoretically, for this H, it should be -2.0 - 0.5 = -2.5 (if we can rotate fully)
    # But RY is restricted, let's see.

if __name__ == "__main__":
    main()
