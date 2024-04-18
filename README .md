<p align="center">
<img src="figures/Q&ML.png" alt="Q&ML Logo" width="600">
</p>

<h2><p align="center">A PyThon Library for Quantum Computation and Machine Learning</p></h2>
<h3><p align="center">Updated, Scalable, Easy Implement, Easy Reading and Comprehension</p></h3>


<p align="center">
    <a href="https://github.com/QUANTUM-AND-ML/QaML/blob/main/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/github/license/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
    </a>
    <a href="https://www.python.org/downloads/release/python-3813/">
        <img alt="Version" src="https://img.shields.io/badge/Python-3.8-orange">
    </a>
    <a href="https://github.com/search?q=repo%3AQUANTUM-AND-ML%2FQaML++language%3APython&type=code">
        <img alt="Language" src="https://img.shields.io/github/languages/top/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
    </a>
   <a href="https://github.com/QUANTUM-AND-ML/QaML/activity">
        <img alt="Activity" src="https://img.shields.io/github/last-commit/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
    </a>
       <a href="https://www.nsfc.gov.cn/english/site_1/index.html">
        <img alt="Fund" src="https://img.shields.io/badge/supported%20by-NSFC-green">
    </a>
    <a href="https://twitter.com/FindOne0258">
        <img alt="twitter" src="https://img.shields.io/badge/twitter-chat-2eb67d.svg?logo=twitter">
    </a>


</p>
<br />



## Quantum & Machine Learning
Relevant scripts and data for the paper entitled "Output Estimation of Quantum Circuits based on Graph Neural Networks"

## Table of contents
* [**Main work**](#Main-work)
* [**Results display**](#Results-display)
* [**Python scripts**](#Python-scripts)
* [**Dependencies**](#Dependencies)

## Main work
In this paper, motivated by the natural graph representation of quantum circuits, we propose a **Graph Neural Networks** (*GNNs*) based scheme to **predict output expectation values of quantum circuits under noisy and noiseless situations**. We first generate two large datasets which are classically simulated quantum circuits with analytical expectation values and random quantum circuits with noisy expectation values obtained on noisy simulators. Then, we transform each circuit in above datasets into the corresponding graph with gates and circuit properties as node features, where noise properties are embed as node features for noisy expectation values estimation. Next, graph neural network estimator is trained to predict single-qubit and two-qubits noisy and noiseless expectation values. Evaluated on 100 quantum circuits, the graph neural network estimator can achieve more than **0.90 $R^2$ scores**, up to **0.998** and **0.991 $R^2$ scores** under noiseless and noisy situations. Notably, our GNNs estimator is designed to be scalable, where the GNNs estimator trained using small-scale quantum circuits with few qubits and low depth of quantum circuits can effectively predict larger-scale quantum circuits.

<p align="center">
<img src="figures/Figure_1.png" alt="Figure 1" width="800">
</p>

**Figure 1.** The framework for expectation value prediction. **a**) Generate random quantum circuits. **b**) Transform the random quantum circuits into graph structures. **c**) Incorporate noise information into the graph nodes of the quantum circuits, train the graph neural network, and predict the expectation values of quantum circuits.

## Results display
**Table 1.** Noiseless expectation values of random circuits with different number of qubits and depth of circuits are predicted. The GNNs estimator is trained using dataset consisting of 10000 classically simulated quantum circuits and epoch is set as 50. In the table, “*N*” represents the number of qubits, and “*P*” represents the circuit depth.
<p align="center">
<img src="figures/Table_1.png" alt="Table 1" width="700">
</p>

**Table 2.** Noisy expectation values of random circuits with different number of qubits and depth of circuits are predicted. The GNNs estimator is trained using dataset consisting of 10000 classically simulated quantum circuits and epoch is set as 50. In the table, “*N*” represents the number of qubits, and “*P*” represents the circuit depth.
<p align="center">
<img src="figures/Table_2.png" alt="Table 1" width="700">
</p>

<p align="center">
<img src="figures/Figure_2.png" alt="Figure 2" width="600">
</p>

**Figure 2.** The scalable performance of the GNNs estimator. The GNNs estimator is trained using random circuit datasets with $\tilde{N}$ or $\tilde{N} _{withnoise} =$ 3, 5 and 7 qubits under noisy and noiseless situations. The GNNs estimator after training is used to predict the expectation values of random circuits with $N =$ 7, 11 and 16 qubits.

<p align="center">
<img src="figures/Figure_3.png" alt="Figure 3" width="550">
</p>

**Figure 3.** The scalable performance of the GNNs estimator. The comparison of the scalable performance between the GNNs estimator and the CNNs estimator. The quantum circuits with 3, 4, and 5 qubits are used as the training set to predict the expectation values of quantum circuits with 7, 11, and 16 qubits.

## Python scripts
Here is the **brief introduction** to each python file for better understanding and usage:

* "main.py" primarily includes dataset splitting and the **training** and **testing** of the neural network.
* "Dataset.py" mainly includes the generation and saving of a large amount of datasets.
* "CNN_generate_datas.py" mainly implements the generation of random quantum circuits used for training **Convolutional Neural Networks** (*CNNs*), as mentioned in the paper "[**Supervised learning of random quantum circuits via scalable neural networks**](https://iopscience.iop.org/article/10.1088/2058-9565/acc4e2)."
* "GNN_generate_datas.py" mainly implements the generation of random quantum circuits used for training **Graph Neural Networks** (*GNNs*) to compute the ground-state energy of hydrogen molecules.
* "circuit_to_graph.py" primarily defines a class that allows **the conversion of quantum circuits into graphs**.
* "expectation_and_ground_state_energy_calculation.py" mainly defines functions for calculating the expectation values of quantum circuits and the ground state energy of the hydrogen molecule.
* "CNNs.py" and "GNNs.py" respectively contain the architectures of the constructed **Convolutional Neural Networks** (*CNNs*) and **Graph Neural Networks** (*GNNs*).
* "simplification.py" mainly contains some rules for **simplifying quantum circuits**.

## Dependencies
- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- Qiskit >= 0.36.1
- PyTorch >= 1.8.0
- Parallel computing may require NVIDIA GPUs acceleration
