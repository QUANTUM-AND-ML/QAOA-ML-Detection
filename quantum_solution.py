from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
import numpy as np
from qiskit.circuit import Gate, Instruction, Parameter
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pylab
import copy
import networkx as nx
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit.algorithms.minimum_eigensolvers import QAOA, VQE
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.gradients import FiniteDiffEstimatorGradient
from main import h, signal_power, snr_dB, num_samples, generate_bpsk_signal,generate_awgn_noise


def create_graph(edges_list):
    # 创建图
    G = nx.Graph()

    # 添加连接边到图中
    G.add_edges_from(edges_list)

    return G

def dfs(graph, node, visited, current_path, max_path):
    visited[node] = True
    current_path.append(node)

    for neighbor in graph.neighbors(node):
        if not visited[neighbor]:
            dfs(graph, neighbor, visited, current_path, max_path)

    # Check if the current path is longer than the maximum path
    if len(current_path) > len(max_path):
        max_path[:] = current_path[:]

    # Backtrack
    visited[node] = False
    current_path.pop()

def maximum_parallel_operation(num_samples, qubit_connection_list, equivalent_edge_list):
    new_edge_connection_list = []
    new_equivalent_edge_list = []

    circuit_matrix = np.zeros((num_samples, len(qubit_connection_list)))
    for edges_tuple, count in zip(qubit_connection_list, range(len(qubit_connection_list))):
        circuit_matrix[edges_tuple[0], count] = count + 1
        circuit_matrix[edges_tuple[1], count] = count + 1
    #print(circuit_matrix)
    for edges_tuple1, column in zip(qubit_connection_list, range(len(qubit_connection_list))):
        for i in range(column):
            if circuit_matrix[edges_tuple1[0], i] == 0 and circuit_matrix[edges_tuple1[1], i] == 0:
                #print(column, edges_tuple1)
                #print(i)
                circuit_matrix[edges_tuple1[0], i] = column + 1
                circuit_matrix[edges_tuple1[1], i] = column + 1
                circuit_matrix[edges_tuple1[0], column] = 0
                circuit_matrix[edges_tuple1[1], column] = 0
                #print(circuit_matrix)
                break
    order = []
    for j in range(len(qubit_connection_list)):
        for i in range(num_samples):
            if circuit_matrix[[i], [j]] != 0 and int(circuit_matrix[[i], [j]] - 1) not in order:
                order.append(int(circuit_matrix[[i], [j]]) - 1)
    #print(order)
    #print(circuit_matrix)
    #print(edge_connection_list)
    #print(type(edge_connection_list))
    for num in order:
        new_edge_connection_list.append(qubit_connection_list[num])
        new_equivalent_edge_list.append(equivalent_edge_list[num])

    #print("edge_connection_list", qubit_connection_list)
    #print("new_edge_connection_list", new_edge_connection_list)

    return new_edge_connection_list, new_equivalent_edge_list

# 输出最大路径的列表
def find_max_path(graph):
    num_nodes = graph.number_of_nodes()
    visited = {node: False for node in graph.nodes}
    max_path = []

    for node in graph.nodes:
        if not visited[node]:
            current_path = []
            dfs(graph, node, visited, current_path, max_path)

    return max_path


def find_middle_points(lst):
    length = len(lst)

    # 判断列表长度是否为偶数
    if length % 2 == 0:
        # 如果是偶数，返回中间两个点的值
        middle_left = length // 2 - 1
        middle_right = length // 2
        return middle_left, middle_right
    else:
        # 如果是奇数，返回中间点的值和后面的一个值
        middle = length // 2
        return middle, middle + 1 if middle + 1 < length else None

# 创建QAOA对称性质的电路
def qaoa_circuit(num_qubits, depth_p, node_dict, equivalent_node_number, edge_connection_list, equivalent_edge_list, type_edge_number, G):

    #node_dict, equivalent_node_number, edge_connection_list, equivalent_edge_list, type_edge_number, G = graph(vertices_number, edges_number)
    max_path = find_max_path(G)

    #print('max_path', max_path)
    if len(max_path) == num_qubits:
        print('Right')
    else:
        print('Wrong')

    edge_connection_list = list(edge_connection_list)

    #print(edge_connection_list)
    #print(equivalent_edge_list)
    edge_connection_list, equivalent_edge_list  = maximum_parallel_operation(num_qubits, edge_connection_list, equivalent_edge_list)
    #print(edge_connection_list)
    #print(equivalent_edge_list)

    # QAOA的深度
    p = depth_p

    # 定义参数
    gamma_params = [Parameter(f'gamma_{i}') for i in range(equivalent_node_number *  p )]
    beta_params = [Parameter(f'bata_{i}') for i in range(type_edge_number * p )]
    theta_params = [Parameter(f'theta_{i}') for i in range(type_edge_number * p)]

    qaoa_symmetry = QuantumCircuit(num_qubits)

    for step in range(0, p):
        if step == 0:
            middle1, middle2 = find_middle_points(max_path)
            for i in range(middle1, -1, -1):
                if max_path[i] <= max_path[i + 1]:
                    qaoa_symmetry.rx(beta_params[equivalent_edge_list[edge_connection_list.index((max_path[i], max_path[ i + 1]))]], max_path[i])
                    qaoa_symmetry.cx(max_path[i], max_path[i+1])
                else:
                    qaoa_symmetry.rx(beta_params[equivalent_edge_list[edge_connection_list.index((max_path[i+1], max_path[i]))]], max_path[i])
                    qaoa_symmetry.cx(max_path[i], max_path[i + 1])
            for i in range(middle2, len(max_path) - 1):
                if max_path[i] <= max_path[i + 1]:
                    qaoa_symmetry.rx(beta_params[equivalent_edge_list[edge_connection_list.index((max_path[i], max_path[i + 1]))]], max_path[i+1])
                    qaoa_symmetry.cx(max_path[i+1], max_path[i])
                else:
                    qaoa_symmetry.rx(beta_params[equivalent_edge_list[edge_connection_list.index((max_path[i+1], max_path[i ]))]], max_path[i + 1])
                    qaoa_symmetry.cx(max_path[i + 1], max_path[i])

            qaoa_symmetry.barrier()

            for i in range(num_qubits):
                qaoa_symmetry.h(i)
            qaoa_symmetry.barrier()

            new_edge_connection_list = copy.copy(edge_connection_list)
            new_equivalent_edge_list = copy.copy(equivalent_edge_list)

            for i in range(len(max_path) - 1):
                if max_path[i] < max_path[i + 1]:
                    new_equivalent_edge_list.remove(equivalent_edge_list[edge_connection_list.index((max_path[i], max_path[ i + 1]))])
                    new_edge_connection_list.remove((max_path[i], max_path[ i + 1]))
                else:
                    new_equivalent_edge_list.remove(equivalent_edge_list[edge_connection_list.index((max_path[i + 1], max_path[i ]))])
                    new_edge_connection_list.remove((max_path[i + 1], max_path[i]))

            new_edge_connection_list, new_equivalent_edge_list = maximum_parallel_operation(num_qubits, new_edge_connection_list, new_equivalent_edge_list)

            for edges_tuple, count in zip(new_edge_connection_list, range(len(new_edge_connection_list))):
                qaoa_symmetry.rzz(beta_params[new_equivalent_edge_list[count] + (step * type_edge_number) ], edges_tuple[0], edges_tuple[1])
            qaoa_symmetry.barrier()
            #print(edge_connection_list)
            #print(equivalent_edge_list)

        else:
            for edges_tuple, count in zip(edge_connection_list, range(len(edge_connection_list))):
                qaoa_symmetry.rzz(beta_params[equivalent_edge_list[count] + (step * type_edge_number) ], edges_tuple[0], edges_tuple[1])
            qaoa_symmetry.barrier()
        for qubit_number in range(num_qubits):
            qaoa_symmetry.rz(theta_params[node_dict[qubit_number] + (step * equivalent_node_number) ], qubit_number)
            qaoa_symmetry.rx(gamma_params[node_dict[qubit_number] + (step * equivalent_node_number) ], qubit_number)
        qaoa_symmetry.barrier()

    qaoa_symmetry.draw(output='mpl')
    plt.show()
    return qaoa_symmetry, G

'''
def qaoa_circuit(num_qubits, depth_p):
    # 定义参数
    gamma_params = [Parameter(f'gamma_{i}') for i in range(depth_p)]
    beta_params = [Parameter(f'bata_{i}') for i in range(depth_p)]

    qr = QuantumRegister(num_qubits, 'q')
    qaoa = QuantumCircuit(num_qubits)

    # Add initial Hadamard layer
    qaoa.h(range(num_qubits))

    # 添加参数化的旋转门
    # 添加交替的参数化旋转门和 CZZ 门
    for depth in range(depth_p):
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qaoa.rzz(gamma_params[depth], qr[i], qr[j])
        qaoa.barrier(qr[0:num_qubits])
        for i in range(num_qubits):
            qaoa.rz(gamma_params[depth], qr[i])
        for i in range(num_qubits):
            qaoa.rx(beta_params[depth], qr[i])
    return qaoa
'''

# 定义比特数目和深度
num_qubits = 4
depth_p = 1

'''
# 连接比特zz泡利串连接的列表
node_dict = {0: 0, 1: 0, 2: 0, 3: 0}
equivalent_node_number = 1
edge_connection_list = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
equivalent_edge_list = [0, 0, 0, 0, 0, 0]
type_edge_number = 1
'''

# 连接比特zz泡利串连接的列表
node_dict = {0: 0, 1: 1, 2: 2, 3: 3}
equivalent_node_number = 4
edge_connection_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
equivalent_edge_list = [0, 1, 2, 3, 4, 5]
type_edge_number = 6


'''
# 调用函数创建图
G = create_graph(edge_connection_list)

qaoa, G = qaoa_circuit(num_qubits, depth_p, node_dict, equivalent_node_number, edge_connection_list, equivalent_edge_list, type_edge_number, G)
#Qaoa_symmetry.measure_all()
qaoa.draw(output='mpl')
plt.show()
'''

# Define A = h^H * h
A = np.dot(np.conjugate(h).T, h)  / 100
# 将对角线元素置为零
np.fill_diagonal(A, 0)

wrong_number_of_bits = 0
number_of_signals_transmitted = 1

# 设置每隔多少次循环打印一次计数器的值
print_interval = 1

# 计数器初始化为 0
counter = 0

for i in range(number_of_signals_transmitted):
    noise = generate_awgn_noise(signal_power, snr_dB, num_samples)

    transmitter_symbols, transmitter_signals = generate_bpsk_signal(num_samples)

    received_signals = np.dot(h, transmitter_signals) + noise

    # 调用函数创建图
    G = create_graph(edge_connection_list)

    qaoa, G = qaoa_circuit(num_qubits, depth_p, node_dict, equivalent_node_number, edge_connection_list,
                           equivalent_edge_list, type_edge_number, G)

    print(received_signals)
    #print(A)
    # 定义ML检测模型的参数
    #b = np.dot(received_signals.T, h) # Local magnetic fields

    b = received_signals # Local magnetic fields
    #print(b)
    b = b.flatten()
    #print(b)

    counts = []
    values = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    # 初始化 Pauli 字符串列表
    pauli_strings = []

    # 添加局部磁场项
    for i in range(num_qubits):
        pauli_strings.append((("I" * i) + "Z" + ("I" * (num_qubits - i - 1)), b[i]))

    # 添加相互作用项
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            pauli_strings.append((("I" * i) + "Z" + ("I" * (j - i - 1)) + "Z" + ("I" * (num_qubits - j - 1)), A[i][j]))

    # 从 Pauli 字符串列表创建 SparsePauliOp
    '''
    hamiltonian = SparsePauliOp.from_list([
            ("ZIII", -1),
            ("IZII", 1),
            ("IIZI", -1),
            ("IIIZ", 1),
        ])
    '''

    hamiltonian = SparsePauliOp.from_list(pauli_strings)

    print("Ising 模型的哈密顿量 Hf：", hamiltonian)

    optimizer = COBYLA(maxiter=100)
    estimator = Estimator()  # 由于 VQE 使用的是波函数估计，因此评估器为 None 即可

    initial_params = np.random.uniform(0, 2*np.pi, qaoa.num_parameters)

    #训练参数
    vqe = VQE(ansatz = qaoa, optimizer = optimizer, estimator = estimator, callback=store_intermediate_result, initial_point = initial_params)
    results = vqe.compute_minimum_eigenvalue(hamiltonian)
    #print(results)

    # 提取最优参数
    optimal_params = results.optimal_parameters


    # 将最优参数应用到变分形式中
    qaoa.assign_parameters(optimal_params, qaoa)
    qaoa.measure_all()

    # 打印带有最优参数的电路
    #qaoa.draw(output='mpl')
    #plt.show()

    # 选择模拟器或实际量子计算机
    backend = Aer.get_backend('qasm_simulator')

    # 编译电路以适应选择的后端
    compiled_circuit = transpile(qaoa, backend)

    # 运行电路并获取结果
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()

    # 获取测量结果
    counts_circuit = result.get_counts()
    print(counts_circuit)

    max_value = max(counts_circuit.values())

    max_keys = [key for key, value in counts_circuit.items() if value == max_value]

    #print(max_keys)

    bit_string = max_keys[0]

    # 将比特串转换为整数列表
    decoded_bits = [int(bit) for bit in bit_string]

    #print('transmitter_symbol:', transmitter_symbols)
    #print('b', b)
    #print('decoded_bits:', decoded_bits)
    # 比较两个行向量，返回一个布尔数组，表示对应位置上元素是否相同
    element_wise_comparison = transmitter_symbols == decoded_bits

    # 统计布尔数组中为 False（不相同）的元素数量
    num_different_elements = np.sum(~element_wise_comparison)

    wrong_number_of_bits = num_different_elements + wrong_number_of_bits

    # 获取所有可能的比特串
    all_bitstrings = [format(i, f'0{num_samples}b') for i in range(2**num_samples)]

    # 构建一个字典，将未出现的比特串的计数设置为0
    full_result_counts = {bitstring: counts_circuit.get(bitstring, 0) for bitstring in all_bitstrings}

    # 使用 plot_histogram 函数，并传递 full_result_counts 和 bar_labels 参数
    plot_histogram(full_result_counts, bar_labels=True, figsize=(20, 16))
    plt.show()



    pylab.rcParams["figure.figsize"] = (12, 8)
    pylab.plot(counts, values, label=type(optimizer).__name__)
    pylab.xlabel("Eval count")
    pylab.ylabel("Energy")
    pylab.title("Energy convergence using Gradient")
    pylab.legend(loc="upper right")
    plt.show()


    counter += 1
    # 如果计数器能被 print_interval 整除，则打印计数器的值
    if counter % print_interval == 0:
        print("当前循环次数：", counter)

print('误码率BER：', wrong_number_of_bits / ( num_samples * number_of_signals_transmitted))
