# Efficient Hamiltonian Simulation Through Optimal Laser Damage

## Introduction

This repository is to demonstrate the method used in the paper: 

## Abstract
Operating a quantum simulator entails an essential phase: the engineering of the desired Hamiltonian via available experimental controls. In this study, we focus on facilitating arbitrary interaction patterns by fine-tuning the amplitudes of driving lasers in a trapped ion quantum simulator. Construction of sophisticated interaction patterns often requires additional laser tones, subsequently complicating the experimental setup. To address this challenge, we propose adopting  an algorithm widely used  in  machine learning  called the optimal brain damage method. Employing this strategy, we aim to identify laser amplitudes that reliably produce the desired Hamiltonian while minimizing the need for supplementary laser tones. In this method, we leverage the Hessian of the cost function, which measures the deviation between the experimental Hamiltonian and the theoretical target, to remove redundant control elements. As a case study, we conduct numerical simulations of two distinct Hamiltonians. The outcomes  demonstrate a significant parameter reduction on the order of $50\%$ while ensuring  acceptable reconstruction error. Furthermore, when the experiment is constrained by a given parameter budget, the errors are markedly smaller, spanning up to several orders of magnitude less compared to analogous scenarios that employ an equivalent number of laser tones without the benefit of Hessian information.



## Prerequisites

You need to have installed the library Tensorflow before running this project. The whole code is wirtein in python.

## Method
The physical simulation of phono vibrations in the trapped ions are available in
```bash
# utility file 
trapped_ions.py
```
To see the model implementation and Optimal Brain Damage algorithm, 
```bash
/src/optimal_bain_damage.py
```
contain the whole pipeline for the Schwinger model. For the Harper-Hofstadter model please check:
```bash
/utility/Harper_Hofstadter_Himltonian.py
````
 
 
