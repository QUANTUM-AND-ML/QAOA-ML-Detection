from qiskit import Aer, execute, QuantumCircuit
import numpy as np
from qiskit.circuit import Gate, Instruction, Parameter
import matplotlib.pyplot as plt
import pylab
import copy
import networkx as nx
from qiskit.visualization import plot_histogram


def create_graph(edges_list):
    # Create a chart
    G = nx.Graph()

    # Add connecting edges to the chart
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

# Output a list of maximum paths
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

    # Determine if the length of the list is even
    if length % 2 == 0:
        # If it's even, return the values of the middle two points
        middle_left = length // 2 - 1
        middle_right = length // 2
        return middle_left, middle_right
    else:
        # If it's odd, return the value of the middle point and the value after it
        middle = length // 2
        return middle, middle + 1 if middle + 1 < length else None

# Create a QAOA circuit with symmetric properties
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

    # The depth of the QAOA circuit
    p = depth_p

    # Define parameters
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
