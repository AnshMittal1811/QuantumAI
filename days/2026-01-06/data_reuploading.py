import pennylane as qml
from pennylane import numpy as np

# January 06, 2026: Data Re-uploading Classifier
# A single qubit can be a universal function approximator by re-uploading data.

def main():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(weights, x):
        # We repeat the data loading and rotation layers
        for w in weights:
            qml.RZ(x, wires=0)
            qml.RY(w, wires=0)
        return qml.expval(qml.PauliZ(0))

    # Binary classification task: x < 0 -> -1, x > 0 -> 1
    def loss(weights, x, y):
        predictions = [circuit(weights, xi) for xi in x]
        return np.mean((np.array(predictions) - y)**2)

    # Training data
    x = np.linspace(-1, 1, 10)
    y = np.where(x < 0, -1, 1)

    # Weights for 3 layers of re-uploading
    weights = np.array([0.1, 0.1, 0.1], requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    print("Training Data Re-uploading Classifier...")
    for i in range(50):
        weights = opt.step(lambda w: loss(w, x, y), weights)
        if (i+1) % 10 == 0:
            print(f"Step {i+1}: Loss = {loss(weights, x, y):.6f}")

    # Test
    test_x = np.array([-0.5, 0.5])
    print(f"\nPredictions for {test_x}: {[circuit(weights, tx) for tx in test_x]}")

if __name__ == "__main__":
    main()
