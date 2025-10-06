from pathlib import Path

TEMPLATE = """\
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
    algorithm_globals.random_seed = {seed}
    test_size = {test_size}

    if "{ds_type}" == "synthetic-line":
        X = 2 * algorithm_globals.random.random([{num_samples}, {num_features}]) - 1
        y = (np.sum(X, axis=1) >= 0).astype(int) * 2 - 1
        return train_test_split(X, y, test_size=test_size, random_state={seed})
    elif "{ds_type}" == "iris":
        iris = datasets.load_iris()
        X_all, y_all = iris.data, iris.target
        import numpy as np
        mask = np.isin(y_all, {classes})
        X = X_all[mask][:, {features}]
        y_raw = y_all[mask]
        y = (y_raw == max({classes})).astype(int) * 2 - 1
        return train_test_split(X, y, test_size=test_size, random_state={seed})
    else:
        raise ValueError("Unknown dataset type: {ds_type}")

def main():
    Xtr, Xte, ytr, yte = load_data()
    qc = QNNCircuit(num_qubits={num_qubits})  # includes ZZFeatureMap + RealAmplitudes
    qnn = EstimatorQNN(circuit=qc, estimator=Estimator())
    clf = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter={maxiter}))
    clf.fit(Xtr, ytr)
    print("Accuracy:", clf.score(Xte, yte))
    print("Predictions:", clf.predict(Xte))

if __name__ == "__main__":
    main()
"""

def write_generated_run(spec: dict, out_path: str = None) -> str:
    # save to project root as generated_run.py
    if out_path is None:
        project_root = Path(__file__).resolve().parents[1]
        out_path = project_root / "generated_run.py"

    ds  = spec["dataset"]
    cir = spec["circuit"]
    opt = spec["optimizer"]

    code = TEMPLATE.format(
        seed=ds.get("seed", 42),
        test_size=ds.get("test_size", 0.2),
        ds_type=ds.get("type", "synthetic-line"),
        num_samples=ds.get("num_samples", 20),
        num_features=ds.get("num_features", 2),
        classes=ds.get("classes", [0, 1]),
        features=ds.get("features", [0, 1]),
        num_qubits=cir.get("num_qubits", 2),
        maxiter=opt.get("maxiter", 60),
    )

    Path(out_path).write_text(code)
    return str(out_path)
