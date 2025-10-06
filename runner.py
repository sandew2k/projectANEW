from typing import Dict, Any
import numpy as np

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import StatevectorEstimator as Estimator

from sklearn.model_selection import train_test_split
from sklearn import datasets


def _build_dataset(ds: Dict[str, Any]):
    """Return X_train, X_test, y_train, y_test according to spec."""
    seed = ds.get("seed", 42)
    algorithm_globals.random_seed = seed
    test_size = ds.get("test_size", 0.2)

    if ds["type"] == "synthetic-line":
        n = ds.get("num_samples", 20)
        d = ds.get("num_features", 2)
        X = 2 * algorithm_globals.random.random([n, d]) - 1
        y = (np.sum(X, axis=1) >= 0).astype(int) * 2 - 1  # -1/+1
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    elif ds["type"] == "iris":
        iris = datasets.load_iris()
        X_all = iris.data
        y_all = iris.target
        # pick 2 features, 2 classes for binary demo
        feat_idx = ds.get("features", [0, 1])
        cls = ds.get("classes", [0, 1])
        mask = np.isin(y_all, cls)
        X = X_all[mask][:, feat_idx]
        y_raw = y_all[mask]
        y = (y_raw == max(cls)).astype(int) * 2 - 1  # map to -1/+1
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    else:
        raise ValueError(f"Unknown dataset type: {ds['type']}")


def run_pipeline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build and run an EstimatorQNN classifier pipeline described by 'spec'.
    Returns metrics: accuracy, (optional) predictions, sizes.
    """
    # 1) Data
    Xtr, Xte, ytr, yte = _build_dataset(spec["dataset"])

    # 2) Encoder + Circuit via QNNCircuit (defaults: ZZFeatureMap + RealAmplitudes)
    num_qubits = spec["circuit"].get("num_qubits", Xtr.shape[1])
    qc = QNNCircuit(num_qubits=num_qubits)

    # 3) QNN
    if spec["qnn"].get("type", "estimator") != "estimator":
        raise ValueError("Sprint-1 supports only qnn.type = 'estimator'")
    qnn = EstimatorQNN(circuit=qc, estimator=Estimator())

    # 4) Optimizer + Classifier
    if spec["optimizer"].get("type", "cobyla") != "cobyla":
        raise ValueError("Sprint-1 supports only optimizer.type = 'cobyla'")
    maxiter = spec["optimizer"].get("maxiter", 60)
    clf = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=maxiter))

    # 5) Train & evaluate
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))

    result = {
        "accuracy": acc,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    if spec.get("outputs", {}).get("return_predictions", True):
        result["predictions"] = clf.predict(Xte).tolist()

    return result
