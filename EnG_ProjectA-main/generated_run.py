# Auto-generated from JSON spec (Sprint-1)
import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from sklearn.model_selection import train_test_split
from sklearn import datasets

def load_data():
    algorithm_globals.random_seed = 42
    test_size = 0.2

    if "csv" == "synthetic-line":
        X = 2 * algorithm_globals.random.random([20, 2]) - 1
        y = (np.sum(X, axis=1) >= 0).astype(int) * 2 - 1
        return train_test_split(X, y, test_size=test_size, random_state=42)
    elif "csv" == "iris":
        iris = datasets.load_iris()
        X_all, y_all = iris.data, iris.target
        import numpy as np
        mask = np.isin(y_all, [0, 1])
        X = X_all[mask][:, [0, 1]]
        y_raw = y_all[mask]
        y = (y_raw == max([0, 1])).astype(int) * 2 - 1
        return train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        raise ValueError("Unknown dataset type: csv")

def main():
    Xtr, Xte, ytr, yte = load_data()
    qc = QNNCircuit(num_qubits=4)  # includes ZZFeatureMap + RealAmplitudes
    qnn = EstimatorQNN(circuit=qc, estimator=Estimator())
    clf = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=15))
    clf.fit(Xtr, ytr)
    print("Accuracy:", clf.score(Xte, yte))
    print("Predictions:", clf.predict(Xte))

if __name__ == "__main__":
    main()
