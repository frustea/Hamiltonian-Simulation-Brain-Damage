# Efficient Hamiltonian Simulation Through Optimal Laser Damage

## Introduction

This repository is to demonstrate the method used in the paper: [arXiv link]

## Abstract
Operating a quantum simulator entails an essential phase: the engineering of the desired Hamiltonian via available experimental controls. In this study, we focus on facilitating arbitrary interaction patterns by fine-tuning the amplitudes of driving lasers in a trapped ion quantum simulator. Construction of sophisticated interaction patterns often requires additional laser tones, subsequently complicating the experimental setup. To address this challenge, we propose adopting  an algorithm widely used  in  machine learning  called the optimal brain damage method. Employing this strategy, we aim to identify laser amplitudes that reliably produce the desired Hamiltonian while minimizing the need for supplementary laser tones. In this method, we leverage the Hessian of the cost function, which measures the deviation between the experimental Hamiltonian and the theoretical target, to remove redundant control elements. As a case study, we conduct numerical simulations of two distinct Hamiltonians. The outcomes  demonstrate a significant parameter reduction on the order of $50\%$ while ensuring  acceptable reconstruction error. Furthermore, when the experiment is constrained by a given parameter budget, the errors are markedly smaller, spanning up to several orders of magnitude less compared to analogous scenarios that employ an equivalent number of laser tones without the benefit of Hessian information.



## Prerequisites

You need to have installed the library Tensorflow before running this project. The whole code is written in Python.

## Method
The physical simulation of phonon vibrations in the trapped ions is available in
```bash
# utility file 
trapped_ions.py
```
To see the model implementation and Optimal Brain Damage algorithm, 
```bash
/src/optimal_bain_damage.py
```

## Getting Started

1. Clone the repository: `git clone https://github.com/frustea/Hamiltonian-Simulation-Brain-Damage.git`
2. Install dependencies: `pip install -r requirements.txt`
3. `cd src' 
4. Run a simulation: `python optimal_brain_damge.py`
## Contributing

Contributions to extend the model and analyses are welcome! Please open an issue or PR.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
 
