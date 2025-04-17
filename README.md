# Digital Twin Sensor Steering (DT-SS)

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

![DCD overview diagram](/docs/images/Sensor-Configuration-Evolution.svg)

This repository contains the official implementation of the paper:

**"Adaptive Sensor Steering Strategy Using Deep Reinforcement Learning for Dynamic Data Acquisition in Digital Twins"**  
by *Collins O. Ogbodo, Timothy J. Rogers, Mattia Dal Borgo, and David J. Wagg*  
Preprint available on [arXiv]([https://arxiv.org/abs/placeholder](https://arxiv.org/abs/2504.10248))

---

## 🚀 Key Features

- **Digital Twin Integration** — Synchronized physical-virtual interaction with near real-time feedback.
- **MDP-based Formulation** — Sensor movement modeled as a sequential decision process.
- **Rainbow DQN Agent** — Uses distributional RL for risk-aware decision-making.
- **Information-Theoretic Reward** — Maximizes determinant of the FIM for informative sensor placement.
- **Spatial Correlation-Aware** — Rewards spatially well-distributed sensor configurations.
- **Case Studies** — Applied to a cantilever plate under damage severity and localization scenarios.
---

## Setup
To install the necessary dependencies, run the following commands:
```
conda create --name DT_SS python=3.8
conda activate DT_SS
pip install -r requirements.txt
```
---
## Citation
@article{ogbodo2025adaptive,
  title={Adaptive Sensor Steering Strategy Using Deep Reinforcement Learning for Dynamic Data Acquisition in Digital Twins},
  author={Ogbodo, Collins O and Rogers, Timothy J and Borgo, Mattia Dal and Wagg, David J},
  journal={arXiv preprint arXiv:2504.10248},
  year={2025}
}


