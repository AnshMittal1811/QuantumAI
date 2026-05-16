import pennylane as qml
from pennylane import numpy as np
import os

# January 09, 2026: Quantum GAN (QGAN) for MNIST
# Generating 4D latent vectors that represent compressed MNIST images.

def main():
    # Load data from Day 8
    data_dir = "../../data/mnist"
    try:
        real_data = np.load(os.path.join(data_dir, "mnist_01_pca4.npy"))
    except FileNotFoundError:
        print("Data not found. Please run Day 08 script first.")
        return

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def generator_circuit(weights, latent_noise):
        # Latent noise input
        qml.AngleEmbedding(latent_noise, wires=range(n_qubits))
        # Learnable circuit
        qml.StrongEntanglingLayers(weights, wires=range(n_qubits))
        # Output is the expectation value of PauliZ for each qubit (creating a 4D vector)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def disc_cost(disc_weights, real_samples, fake_samples):
        # Simple classical discriminator for demonstration
        # In a real QGAN, the discriminator could also be quantum.
        real_score = np.mean(np.dot(real_samples, disc_weights))
        fake_score = np.mean(np.dot(fake_samples, disc_weights))
        return fake_score - real_score

    def gen_cost(gen_weights, disc_weights, latent_noise):
        fake_samples = [generator_circuit(gen_weights, ln) for ln in latent_noise]
        return -np.mean(np.dot(fake_samples, disc_weights))

    # Initialize
    gen_weights = np.random.uniform(0, 2*np.pi, size=(3, n_qubits, 3), requires_grad=True)
    disc_weights = np.random.uniform(-1, 1, size=(4,), requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)

    print("Training QGAN on 4D MNIST Latents...")
    for i in range(20):
        # Sample latent noise
        latent_noise = np.random.uniform(-np.pi, np.pi, size=(10, n_qubits))
        
        # Update Generator
        gen_weights = opt.step(lambda w: gen_cost(w, disc_weights, latent_noise), gen_weights)
        
        # Update Discriminator
        fake_samples = np.array([generator_circuit(gen_weights, ln) for ln in latent_noise])
        real_batch = real_data[np.random.choice(len(real_data), 10)]
        disc_weights = opt.step(lambda w: disc_cost(w, real_batch, fake_samples), disc_weights)

        if (i+1) % 5 == 0:
            print(f"Step {i+1}: Gen Cost = {gen_cost(gen_weights, disc_weights, latent_noise):.4f}")

    print("\nQGAN training loop established. Generator is learning to map noise to the MNIST 0/1 manifold.")

if __name__ == "__main__":
    main()
