import pennylane as qml
from pennylane import numpy as np

# January 01, 2026: Hello Quantum World
def main():
    # Define a device (CPU-based lightning simulator)
    dev = qml.device("lightning.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi, wires=0)
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = np.array(0.54, requires_grad=True)
    theta = np.array(0.12, requires_grad=True)

    result = circuit(phi, theta)
    print(f"QNode result (Expectation value of PauliZ): {result}")

    # Compute gradient using autograd
    d_circuit = qml.grad(circuit)
    print(f"Gradient with respect to (phi, theta): {d_circuit(phi, theta)}")

if __name__ == "__main__":
    main()
