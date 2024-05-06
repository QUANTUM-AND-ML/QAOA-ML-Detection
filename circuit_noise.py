from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
import numpy as np
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pylab
import matplotlib.colors as mcolors
import copy
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

# IBM Perth. The dataset is saved in a folder named data_ibm_perth.
T1 = [197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96]
T2 = [97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83]
P01 = [0.0031, 0.00222, 0.00282, 0.0018, 0.0023, 0.00264, 0.00072]
P10 = [0.00226, 0.00236, 0.0023, 0.0013, 0.00164, 0.00208, 0.00054]
gateErrors = [['rz', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['rx', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['h', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['cx', 0.00735, 0.00769667, 0.00694, 0.00835, 0.01169, 0.0098133, 0.00984]]


# Add errors to noise model
noise_model = noise.NoiseModel()
for i in range(num_samples):
    noise_model.add_quantum_error(depolarizing_error(gateErrors[0][i + 1], 1), ['rx', 'h', 'i', 'rz'], [i])
    for k in range(num_samples):
        noise_model.add_quantum_error(
            depolarizing_error((gateErrors[3][i + 1] + gateErrors[3][k + 1]) / 2, 2),
            ['cx','rzz'], [i, k])

# 加入T1和T2
# Instruction times (in nanoseconds)
time_reset = 1000  # 1 microsecond
time_measure = 1000  # 1 microsecond
time_rx = 100  # virtual gate
time_rz = 100  # virtual gate
time_h = 50  # (single X90 pulse)
time_i = 50  # (two X90 pulses)
time_cx = 300

# QuantumError objects
errors_reset = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_reset)
                for t1, t2 in zip(T1, T2)]
errors_measure = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_measure)
                  for t1, t2 in zip(T1, T2)]
errors_rx = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_rx)
            for t1, t2 in zip(T1, T2)]
errors_rz = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_rz)
              for t1, t2 in zip(T1, T2)]
errors_h = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_h)
            for t1, t2 in zip(T1, T2)]
errors_i = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_i)
            for t1, t2 in zip(T1, T2)]
errors_cx = [[thermal_relaxation_error(t1a * 1000, t2a * 1000, time_cx).expand(
    thermal_relaxation_error(t1b * 1000, t2b * 1000, time_cx))
    for t1a, t2a in zip(T1, T2)]
    for t1b, t2b in zip(T1, T2)]

# Add errors to noise model
for j in range(num_samples):
    noise_model.add_quantum_error(errors_reset[j], "reset", [j])
    noise_model.add_quantum_error(errors_measure[j], "measure", [j])
    noise_model.add_quantum_error(errors_rx[j], "rx", [j])
    noise_model.add_quantum_error(errors_rz[j], "rz", [j])
    noise_model.add_quantum_error(errors_h[j], "h", [j])
    noise_model.add_quantum_error(errors_i[j], "i", [j])
    for k in range(num_samples):
        noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

# Add readout error
# Measurement miss-assignement probabilities
for i in range(num_samples):
    noise_model.add_readout_error(ReadoutError([[1 - P10[i], P10[i]], [P01[i], 1 - P01[i]]]), [i])

print(noise_model)
