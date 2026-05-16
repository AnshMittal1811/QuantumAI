import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as np

# January 05, 2026: Hybrid Quantum-Classical Neural Network
# Using PennyLane's TorchLayer to integrate a QNode into a PyTorch model

def main():
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # weight_shapes maps the names of the QNode's arguments to their shapes
    weight_shapes = {"weights": (3, n_qubits)} # 3 layers

    # Convert the QNode to a PyTorch Layer
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # Define a simple hybrid model
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.clayer_in = nn.Linear(4, n_qubits) # Reduce input dim to qubit count
            self.qlayer = qlayer
            self.clayer_out = nn.Linear(n_qubits, 1) # Predict a single scalar

        def forward(self, x):
            x = torch.sigmoid(self.clayer_in(x))
            x = self.qlayer(x)
            x = self.clayer_out(x)
            return x

    model = HybridModel()
    
    # Test a forward pass
    dummy_input = torch.randn(1, 4)
    output = model(dummy_input)
    
    print(f"Input: {dummy_input}")
    print(f"Hybrid Model Output: {output}")
    print("\nModel Parameters (including Quantum Weights):")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

if __name__ == "__main__":
    main()
