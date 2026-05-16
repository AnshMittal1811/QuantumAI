import pennylane as qml
from pennylane import numpy as np

# January 04, 2026: Quantum Feature Maps
# Embedding classical data into a quantum state

def main():
    dev = qml.device("default.qubit", wires=2)

    # Angle Encoding: maps x to RX(x)
    @qml.qnode(dev)
    def angle_encoding_circuit(x):
        qml.AngleEmbedding(x, wires=[0, 1], rotation='X')
        return qml.state()

    # IQP (Instantaneous Quantum Polynomial) Embedding
    # A more complex feature map often used for Quantum Support Vector Machines
    @qml.qnode(dev)
    def iqp_encoding_circuit(x):
        qml.IQPEmbedding(x, wires=[0, 1], n_repeats=1)
        return qml.probs(wires=[0, 1])

    x = np.array([0.5, 1.2])
    
    state = angle_encoding_circuit(x)
    print(f"Angle Encoding State Vector: {state}")
    
    probs = iqp_encoding_circuit(x)
    print(f"IQP Encoding Probs: {probs}")

    # Theoretical Note: The overlap between two such circuits defines the Quantum Kernel
    # K(x, y) = |<psi(x)|psi(y)>|^2
    
if __name__ == "__main__":
    main()
