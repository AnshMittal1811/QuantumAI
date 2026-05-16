import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# January 07, 2026: Barren Plateaus Investigation
# Demonstrating that gradient variance vanishes exponentially with qubit count/depth.

def main():
    def calculate_grad_variance(n_qubits, depth, n_samples=100):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(weights):
            # Random initialization
            qml.StrongEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        grads = []
        for _ in range(n_samples):
            # Random weights
            weights = np.random.uniform(0, 2 * np.pi, size=(depth, n_qubits, 3), requires_grad=True)
            # Gradient with respect to the first parameter
            grad_fn = qml.grad(circuit)
            g = grad_fn(weights)[0, 0, 0]
            grads.append(g)
        
        return np.var(grads)

    qubit_counts = [2, 4, 6]
    depth = 5
    variances = []

    print(f"Calculating gradient variances (depth={depth})...")
    for n in qubit_counts:
        var = calculate_grad_variance(n, depth)
        variances.append(var)
        print(f"Qubits: {n}, Variance: {var:.8f}")

    # Theoretical expectation: Variance decreases as 1/2^n
    # Note: On a local simulator, this is just a demonstration.
    print("\nObservation: As qubit count increases, gradients become smaller and harder to train (Barren Plateaus).")

if __name__ == "__main__":
    main()
