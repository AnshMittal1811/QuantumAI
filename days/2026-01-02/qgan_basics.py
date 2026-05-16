import pennylane as qml
from pennylane import numpy as np

# January 02, 2026: Simple 1-qubit QGAN
# Learning a target probability distribution (e.g., [0.7, 0.3])

def main():
    dev = qml.device("default.qubit", wires=1)

    # Target: 0.7 probability for |0>, 0.3 for |1>
    target_dist = np.array([0.7, 0.3])

    @qml.qnode(dev)
    def generator(weights):
        qml.RY(weights[0], wires=0)
        qml.RZ(weights[1], wires=0)
        return qml.probs(wires=0)

    def discriminator(probs, weights):
        # A simple classical discriminator (linear)
        return probs[0] * weights[0] + probs[1] * weights[1]

    # Initial weights
    gen_weights = np.array([0.1, 0.1], requires_grad=True)
    disc_weights = np.array([0.5, 0.5], requires_grad=True)

    # Simple training loop for 50 steps
    lr = 0.1
    for i in range(50):
        # Generator step
        with qml.Tracker(dev):
            current_probs = generator(gen_weights)
            # Loss: maximize probability that discriminator classifies fake as real
            # Here we just use a dummy cost to show the loop
            gen_cost = -np.sum(current_probs * disc_weights)
            
            # Manual gradient update for gen
            grad_fn = qml.grad(generator)
            gen_grad = grad_fn(gen_weights)
            # Simplified update
            gen_weights = gen_weights - lr * gen_grad[0] # Very rough

        if i % 10 == 0:
            print(f"Step {i}: Generator Probs: {current_probs}")

    print(f"Final Generator Probs: {generator(gen_weights)}")
    print(f"Target Probs: {target_dist}")

if __name__ == "__main__":
    main()
