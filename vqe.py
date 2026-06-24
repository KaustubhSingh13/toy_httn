import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

'''
    I have intentionally written this to be as clear and easy to read as possible, 
    possibly at the cost of speed.

    When using the algorithm at scale, this could be streamlined.
'''
def expectation(params, ansatz, operator, 
                estimator = StatevectorEstimator(), shots = None):
    if shots is not None:
        estimator.options.default_shots = shots

    pub = (ansatz, operator, list([params]))
    job = estimator.run([pub])
    result = job.result()
    expect_value = float(result[0].data.evs[0])
    return expect_value

def vqe_cost(params, ansatz, hamiltonian:SparsePauliOp, 
             estimator = StatevectorEstimator(), shots = None):
    return expectation(params, ansatz, hamiltonian, 
                       estimator, shots)

def minimisation(cost, init_params, *cost_args, max_iter = 1000):
    result = minimize(cost, init_params, args = cost_args, 
                      method = 'COBYLA', tol=1e-4, options = {'maxiter':max_iter})
    return result


def run_vqe(q_circuit, init_params, hamiltonian:SparsePauliOp,
            estimator = StatevectorEstimator(), shots = None, max_iter = 1000):

        return minimisation(vqe_cost, init_params, q_circuit, hamiltonian, 
                            estimator, shots, max_iter = max_iter)
