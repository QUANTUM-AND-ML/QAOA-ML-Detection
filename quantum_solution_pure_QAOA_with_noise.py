from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
import numpy as np
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pylab
import matplotlib.colors as mcolors
import copy
import networkx as nx
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit_aer import AerSimulator, noise
from qiskit.utils import QuantumInstance
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.algorithms.minimum_eigensolvers import QAOA, VQE
from scipy.optimize import minimize
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import CircuitStateFn, PauliExpectation
from qiskit.algorithms.gradients import FiniteDiffEstimatorGradient
from main import h, signal_power, snr_dB, num_samples, generate_bpsk_signal,generate_awgn_noise

# IBM Perth. The dataset is saved in a folder named data_ibm_perth.
T1 = [197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96]
T2 = [97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83]
P01 = [0.031, 0.0222, 0.0282, 0.018, 0.023, 0.0264, 0.0072]
P10 = [0.0226, 0.0236, 0.023, 0.013, 0.0164, 0.0208, 0.0054]
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

# 添加读出错误
# Measurement miss-assignement probabilities
for i in range(num_samples):
    noise_model.add_readout_error(ReadoutError([[1 - P10[i], P10[i]], [P01[i], 1 - P01[i]]]), [i])

print(noise_model)





# 存储训练中间的期望值
counts = []
values = []


# 存储训练中间的期望值
counts1 = []
values1 = []

def store_intermediate_result1(eval_count, parameters, mean, std):
    counts1.append(eval_count)
    values1.append(mean)

def pure_qaoa_circuit_for_COBYLA(num_qubits, depth_p, params, matrix_A, vector_b):

    gamma_params = params[:depth_p]
    beta_params =  params[depth_p:]

    pure_qaoa_for_COBYLA = QuantumCircuit(num_qubits)

    # 给每个比特添加H门
    for qubit in range(num_qubits):
        pure_qaoa_for_COBYLA.h(qubit)
    pure_qaoa_for_COBYLA.barrier()
    for p in range(depth_p):
        # 遍历所有比特的组合，为每对比特添加一个RZZ门
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pure_qaoa_for_COBYLA.rzz(matrix_A[i][j] * gamma_params[p], i, j)
        pure_qaoa_for_COBYLA.barrier()
        for i in range(num_qubits):
            pure_qaoa_for_COBYLA.rz(-1 * vector_b[i] * gamma_params[p], i)

        for i in range(num_qubits):
            pure_qaoa_for_COBYLA.rx(beta_params[p], i)
        pure_qaoa_for_COBYLA.barrier()
    return pure_qaoa_for_COBYLA

def pure_qaoa_circuit_for_Bayesian(num_qubits, depth_p, params, matrix_A, vector_b):

    gamma_params = params[:depth_p]
    beta_params =  params[depth_p:]

    pure_qaoa_for_Bayesian = QuantumCircuit(num_qubits)

    # 给每个比特添加H门
    for qubit in range(num_qubits):
        pure_qaoa_for_Bayesian.h(qubit)
    pure_qaoa_for_Bayesian.barrier()
    for p in range(depth_p):
        # 遍历所有比特的组合，为每对比特添加一个RZZ门
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pure_qaoa_for_Bayesian.rzz(matrix_A[i][j] * gamma_params[p], i, j)
        pure_qaoa_for_Bayesian.barrier()
        for i in range(num_qubits):
            pure_qaoa_for_Bayesian.rz(-1 * vector_b[i] * gamma_params[p], i)

        for i in range(num_qubits):
            pure_qaoa_for_Bayesian.rx(beta_params[p], i)
        pure_qaoa_for_Bayesian.barrier()
    return pure_qaoa_for_Bayesian


qubits = num_samples

depth_p = 3

# Define A = h^H * h
A = np.dot(np.conjugate(h).T, h)  / 100
# 将对角线元素置为零
np.fill_diagonal(A, 0)

# 生成BPSK信号并加入噪声传输
noise = generate_awgn_noise(signal_power, snr_dB, num_samples)
transmitter_symbols, transmitter_signals = generate_bpsk_signal(num_samples)
received_signals = np.dot(h, transmitter_signals) + noise

# 收到信号的 b = y.T * H
b = np.dot(received_signals.T, h) # Local magnetic fields
b = b.flatten()

#根据信息生成含参数QAOA线路
pure_qaoa_for_COBYLA = pure_qaoa_circuit_for_COBYLA(qubits, depth_p,[np.pi,np.pi,np.pi,np.pi,np.pi,np.pi ], A, b)
pure_qaoa_for_COBYLA.draw(output='mpl')
plt.show()

# 定义QAOA线路解决问题的的哈密顿量Hf
# 初始化 Pauli 字符串列表
pauli_strings = []
paulis = []
coeffs = []
# 添加局部磁场项
for i in range(qubits):
    pauli_strings.append((("I" * i) + "Z" + ("I" * (qubits - i - 1)), -1 * b[( qubits - 1 )-i]))
    paulis.append(("I" * i) + "Z" + ("I" * (qubits - i - 1)))
    coeffs.append(-1 * b[( qubits - 1 )- i])
# 添加相互作用项
for i in range(qubits):
    for j in range(i + 1, qubits):
        pauli_strings.append((("I" * i) + "Z" + ("I" * (j - i - 1)) + "Z" + ("I" * (qubits - j - 1)), A[( qubits - 1 )- i][( qubits - 1 ) - j]))
        paulis.append(("I" * i) + "Z" + ("I" * (j - i - 1)) + "Z" + ("I" * (qubits - j - 1)))
        coeffs.append(A[( qubits - 1 ) - i][( qubits - 1 ) - j])
#print(pauli_strings)
print(paulis)
print(coeffs)
hamiltonian = SparsePauliOp.from_list(pauli_strings)
print("Ising 模型的哈密顿量 Hf：", hamiltonian)

def objective_function_for_COBYLA(params):

    params = params
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    # Simulation of the quantum circuit
    sim_noise = AerSimulator(noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_CNNs_noise = transpile(pure_qaoa_for_Bayesian, sim_noise)

    result = sim_noise.run(circ_CNNs_noise, shots=2048).result()
    '''
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # 编译电路以适应选择的后端
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # 运行电路并获取结果
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    '''
    # 获取测量结果
    counts_circuit = result.get_counts()

    # 初始化期望值
    expectation = 0

    # 遍历哈密顿量的每一项
    for pauli, coeff in zip(paulis, coeffs):
        # 初始化当前哈密顿量项的期望值
        term_expectation = 0

        # 遍历采样中的每一个基态
        for basis_state, count in counts_circuit.items():
            # 计算当前基态的期望值，并加权求和
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # 如果哈密顿量项为单位矩阵，期望值即为基态出现的次数
            else:
                basis_state_contrib = 1  # 基态的贡献初始为1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1  # 如果哈密顿量项不为单位矩阵，则根据基态的相应比特取值计算贡献
                term_expectation += basis_state_contrib * count  # 将当前基态的贡献乘以基态出现的次数并累加到当前哈密顿量项的期望值上

        # 将当前哈密顿量项的期望值乘以其系数，并累加到总期望值上
        expectation += coeff * term_expectation

    # 对期望值进行归一化
    total_counts = sum(counts_circuit.values())
    expectation /= total_counts
    values.append(expectation)

    return expectation

'''
optimizer = COBYLA(maxiter=100)

estimator = Estimator()  # 由于 VQE 使用的是波函数估计，因此评估器为 None 即可
'''

initial_params = np.random.uniform(0, 2*np.pi, 2 * depth_p)

# 指定最大迭代步数
options = {'maxiter': 100}

# 调用 minimize 函数
results = minimize(objective_function_for_COBYLA, initial_params, method='COBYLA', options=options)
'''
#训练参数
vqe = VQE(ansatz = pure_qaoa_for_COBYLA, optimizer = optimizer, estimator = estimator, callback=store_intermediate_result, initial_point = initial_params)
results = vqe.compute_minimum_eigenvalue(hamiltonian)
'''

print(results)

counts = list(range(len(values)))
# 提取最优参数
optimal_params = results.x

pure_qaoa_for_COBYLA = pure_qaoa_circuit_for_COBYLA(qubits, depth_p, optimal_params, A, b)
pure_qaoa_for_COBYLA.measure_all()
# Simulation of the quantum circuit
sim_noise = AerSimulator(noise_model=noise_model)

# Transpile circuit for noisy basis gates
circ_CNNs_noise = transpile(pure_qaoa_for_COBYLA, sim_noise)

result = sim_noise.run(circ_CNNs_noise, shots=1024).result()
'''
# 选择模拟器或实际量子计算机
backend = Aer.get_backend('qasm_simulator')

# 编译电路以适应选择的后端
compiled_circuit = transpile(pure_qaoa_for_COBYLA, backend)

# 运行电路并获取结果
job = backend.run(compiled_circuit, shots=2048)
result = job.result()
'''
# 获取测量结果
counts_circuit = result.get_counts()
#print(counts_circuit)

max_value = max(counts_circuit.values())

max_keys = [key for key, value in counts_circuit.items() if value == max_value]

#print(max_keys)

bit_string = max_keys[0]

# 将比特串转换为整数列表
decoded_bits = [int(bit) for bit in bit_string]
decoded_bits = decoded_bits[::-1]
decoded_bits = [1 - bit for bit in decoded_bits]

#print('transmitter_symbol:', transmitter_symbols)
#print('b', b)
#print('decoded_bits:', decoded_bits)
# 比较两个行向量，返回一个布尔数组，表示对应位置上元素是否相同
element_wise_comparison = transmitter_symbols == decoded_bits

wrong_number_of_bits = 0

# 统计布尔数组中为 False（不相同）的元素数量
num_different_elements = np.sum(~element_wise_comparison)

wrong_number_of_bits = num_different_elements + wrong_number_of_bits

# 获取所有可能的比特串
all_bitstrings = [format(i, f'0{num_samples}b') for i in range(2**num_samples)]

# 构建一个字典，将未出现的比特串的计数设置为0
full_result_counts = {bitstring: counts_circuit.get(bitstring, 0) for bitstring in all_bitstrings}

# 反转字符串序列和0与1换位
reversed_counts = {}
for key, value in full_result_counts.items():
    reversed_key = ''.join(['1' if char == '0' else '0' for char in key[::-1]])  # 反转键并进行0和1的取反操作
    reversed_counts[reversed_key] = value

# 使用 plot_histogram 函数，并传递 full_result_counts 和 bar_labels 参数
plot_histogram(reversed_counts, bar_labels=True, figsize=(20, 16))
plt.show()

pylab.rcParams["figure.figsize"] = (12, 8)
pylab.plot(counts, values, label= 'COBYLA')
pylab.xlabel("Iterations")
pylab.ylabel("Expected values")
pylab.title("Expected value changes with iteration in noisy circuits")
pylab.legend(loc="upper right")
plt.show()

print('transmitter_symbols', transmitter_symbols)
print('decoded_bits', decoded_bits)

# 贝叶斯优化(Bayesian Optimization)QAOA的线路参数
'''
  有关贝叶斯优化(Bayesian Optimization)的相关代码
'''

def objective_function_for_Bayesian(params_0, params_1, params_2, params_3, params_4, params_5):

    params = [params_0, params_1,params_2, params_3, params_4, params_5]
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # 编译电路以适应选择的后端
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # 运行电路并获取结果
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()

    # 获取测量结果
    counts_circuit = result.get_counts()

    # 初始化期望值
    expectation = 0

    # 遍历哈密顿量的每一项
    for pauli, coeff in zip(paulis, coeffs):
        # 初始化当前哈密顿量项的期望值
        term_expectation = 0

        # 遍历采样中的每一个基态
        for basis_state, count in counts_circuit.items():
            # 计算当前基态的期望值，并加权求和
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # 如果哈密顿量项为单位矩阵，期望值即为基态出现的次数
            else:
                basis_state_contrib = 1  # 基态的贡献初始为1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1 # 如果哈密顿量项不为单位矩阵，则根据基态的相应比特取值计算贡献
                term_expectation += basis_state_contrib * count  # 将当前基态的贡献乘以基态出现的次数并累加到当前哈密顿量项的期望值上

        # 将当前哈密顿量项的期望值乘以其系数，并累加到总期望值上
        expectation += coeff * term_expectation

    # 对期望值进行归一化
    total_counts = sum(counts_circuit.values())
    expectation /= total_counts

    return -1 * expectation

#param_space = {'params': ParameterVector('θ', 2*depth_p)}

# 定义每个参数的取值范围字典
param_bounds = {f'params_{i}': (0, 2*np.pi) for i in range(2*depth_p)}

# 创建BayesianOptimization对象并优化参数
optimizer_Bayesian = BayesianOptimization(objective_function_for_Bayesian, param_bounds)
optimizer_Bayesian.maximize(init_points=1, n_iter=50)

# 获取最优参数和目标函数值
best_params = optimizer_Bayesian.max['params']
print("最优参数:", best_params)

# 提取所有值并转换为列表
values_list = list(best_params.values())
print(values_list)

best_pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, values_list, A, b)
best_pure_qaoa_for_Bayesian.measure_all()

# Simulation of the quantum circuit
sim_noise = AerSimulator(noise_model=noise_model)

# Transpile circuit for noisy basis gates
circ_CNNs_noise = transpile(best_pure_qaoa_for_Bayesian, sim_noise)

result = sim_noise.run(circ_CNNs_noise, shots=1024).result()
'''
backend = Aer.get_backend('qasm_simulator')
backend.shots = 2048

# 编译电路以适应选择的后端
compiled_circuit = transpile(best_pure_qaoa_for_Bayesian, backend)

# 运行电路并获取结果
job = backend.run(compiled_circuit, shots=2048)
result = job.result()
'''
# 获取测量结果
counts_circuit = result.get_counts()
print(counts_circuit)
# 获取所有可能的比特串
all_bitstrings = [format(i, f'0{qubits}b') for i in range(2**qubits)]

# 构建一个字典，将未出现的比特串的计数设置为0
full_result_counts = {bitstring: counts_circuit.get(bitstring, 0) for bitstring in all_bitstrings}

# 反转字符串序列和0与1换位
reversed_counts = {}
for key, value in full_result_counts.items():
    reversed_key = ''.join(['1' if char == '0' else '0' for char in key[::-1]])  # 反转键并进行0和1的取反操作
    reversed_counts[reversed_key] = value

# 使用 plot_histogram 函数，并传递 full_result_counts 和 bar_labels 参数
plot_histogram(reversed_counts, bar_labels=True, figsize=(20, 16))
plt.show()

# 创建画布和子图
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制每次迭代的值和迭代次数的图表
ax.plot(range(1, len(optimizer_Bayesian.res)+1), [res['target'] for res in optimizer_Bayesian.res], label ='Bayesian')
pylab.xlabel("Iterations")
pylab.ylabel("Expected values")
pylab.title("Expected value changes with iteration in noisy circuits")
ax.legend(loc="upper right")

# 显示图表
plt.show()

# 贝叶斯和COBYLA联合优化(Bayesian Optimization)QAOA的线路参数
'''
    贝叶斯用来寻找初值，COBYLA用来训练
'''

def objective_function_for_COBYLA1(params):

    params = params
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    # Simulation of the quantum circuit
    sim_noise = AerSimulator(noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_CNNs_noise = transpile(pure_qaoa_for_Bayesian, sim_noise)

    result = sim_noise.run(circ_CNNs_noise, shots=2048).result()
    '''
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # 编译电路以适应选择的后端
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # 运行电路并获取结果
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    '''
    # 获取测量结果
    counts_circuit = result.get_counts()

    # 初始化期望值
    expectation = 0

    # 遍历哈密顿量的每一项
    for pauli, coeff in zip(paulis, coeffs):
        # 初始化当前哈密顿量项的期望值
        term_expectation = 0

        # 遍历采样中的每一个基态
        for basis_state, count in counts_circuit.items():
            # 计算当前基态的期望值，并加权求和
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # 如果哈密顿量项为单位矩阵，期望值即为基态出现的次数
            else:
                basis_state_contrib = 1  # 基态的贡献初始为1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1  # 如果哈密顿量项不为单位矩阵，则根据基态的相应比特取值计算贡献
                term_expectation += basis_state_contrib * count  # 将当前基态的贡献乘以基态出现的次数并累加到当前哈密顿量项的期望值上

        # 将当前哈密顿量项的期望值乘以其系数，并累加到总期望值上
        expectation += coeff * term_expectation

    # 对期望值进行归一化
    total_counts = sum(counts_circuit.values())
    expectation /= total_counts
    values1.append(expectation)

    return expectation

# 定义每个参数的取值范围字典
param_bounds = {f'params_{i}': (0, np.pi) for i in range(2*depth_p)}

# 创建BayesianOptimization对象并优化参数
optimizer_Bayesian1 = BayesianOptimization(objective_function_for_Bayesian, param_bounds)
optimizer_Bayesian1.maximize(init_points=1, n_iter=8)

# 获取最优参数和目标函数值
best_params = optimizer_Bayesian1.max['params']
print("最优参数:", best_params)

# 提取所有值并转换为列表
values_list = list(best_params.values())
'''
# 计算列表的中间索引
mid_index = len(values_list) // 2

initial_params = values_list[mid_index:] + values_list[:mid_index]
print('initial_params', initial_params)
'''
pure_qaoa_for_COBYLA = pure_qaoa_circuit_for_COBYLA(qubits, depth_p, values_list, A, b)

# 指定最大迭代步数
options = {'maxiter': 100}

# 调用 minimize 函数
results = minimize(objective_function_for_COBYLA1, values_list, method='COBYLA', options=options)
'''
#训练参数
vqe = VQE(ansatz = pure_qaoa_for_COBYLA, optimizer = optimizer, estimator = estimator, callback=store_intermediate_result, initial_point = initial_params)
results = vqe.compute_minimum_eigenvalue(hamiltonian)
'''

print(results)

counts1 = list(range(len(values1)))
# 提取最优参数
optimal_params = results.x

pure_qaoa_for_COBYLA = pure_qaoa_circuit_for_COBYLA(qubits, depth_p, optimal_params, A, b)
pure_qaoa_for_COBYLA.measure_all()
# Simulation of the quantum circuit
sim_noise = AerSimulator(noise_model=noise_model)

# Transpile circuit for noisy basis gates
circ_CNNs_noise = transpile(pure_qaoa_for_COBYLA, sim_noise)

result = sim_noise.run(circ_CNNs_noise, shots=1024).result()
# 获取测量结果
counts_circuit = result.get_counts()
#print(counts_circuit)

max_value = max(counts_circuit.values())

max_keys = [key for key, value in counts_circuit.items() if value == max_value]

#print(max_keys)

bit_string = max_keys[0]

# 将比特串转换为整数列表
decoded_bits = [int(bit) for bit in bit_string]
decoded_bits = decoded_bits[::-1]
decoded_bits = [1 - bit for bit in decoded_bits]
#print('transmitter_symbol:', transmitter_symbols)
#print('b', b)
#print('decoded_bits:', decoded_bits)
# 比较两个行向量，返回一个布尔数组，表示对应位置上元素是否相同
element_wise_comparison = transmitter_symbols == decoded_bits

wrong_number_of_bits = 0

# 统计布尔数组中为 False（不相同）的元素数量
num_different_elements = np.sum(~element_wise_comparison)

wrong_number_of_bits = num_different_elements + wrong_number_of_bits

# 获取所有可能的比特串
all_bitstrings = [format(i, f'0{num_samples}b') for i in range(2**num_samples)]

# 构建一个字典，将未出现的比特串的计数设置为0
full_result_counts = {bitstring: counts_circuit.get(bitstring, 0) for bitstring in all_bitstrings}

# 反转字符串序列和0与1换位
reversed_counts = {}
for key, value in full_result_counts.items():
    reversed_key = ''.join(['1' if char == '0' else '0' for char in key[::-1]])  # 反转键并进行0和1的取反操作
    reversed_counts[reversed_key] = value/1024

# 自定义 RGB 颜色
custom_color = mcolors.to_hex((0.2, 0.4, 0.6))  # 例如：浅蓝色
# 使用 plot_histogram 函数，并传递 full_result_counts 和 bar_labels 参数
plot_histogram(reversed_counts, bar_labels=True, figsize=(20, 16), color=custom_color)
# 设置纵轴标题
plt.ylabel("probability",fontsize=25)
# 设置纵轴刻度标签的字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

pylab.rcParams["figure.figsize"] = (12, 8)
pylab.plot(counts, values, label='COBYLA', color='blue', linestyle='-')
pylab.plot(counts1, values1, label='Bayesian and COBYLA', color='black', linestyle='--')
pylab.xlabel("Iterations")
pylab.ylabel("Expected values")
pylab.title("Expected value changes with iteration in noisy circuits")
pylab.legend(loc="upper right")
plt.show()

