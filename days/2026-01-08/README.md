# January 08, 2026

## Objectives
- Prepare MNIST dataset for Quantum Machine Learning.
- Perform dimensionality reduction using PCA.

## Progress
- Developed `mnist_prep.py`.
- Downloaded and filtered MNIST (digits 0 and 1).
- Reduced 784-dimensional images to 4-dimensional latent vectors using PCA.
- Normalized the data to the range $[-\pi, \pi]$ for efficient Quantum Angle Encoding.
- Saved processed data to the `data/` directory for subsequent generative tasks.
