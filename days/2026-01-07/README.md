# January 07, 2026

## Objectives
- Investigate the phenomenon of Barren Plateaus.
- Analyze how gradient variance scales with qubit count and circuit depth.

## Progress
- Developed `barren_plateaus.py`.
- Quantified the vanishing gradient issue by sampling gradients from random unitary circuits.
- Confirmed that variance decreases as $1/2^n$ (where $n$ is the qubit count), making optimization extremely difficult for deep, wide circuits without proper initialization.
