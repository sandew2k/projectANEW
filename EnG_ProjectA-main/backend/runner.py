from typing import Dict, Any
import os
import numpy as np
import pandas as pd

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import Estimator

from sklearn.model_selection import train_test_split
from sklearn import datasets


def _build_dataset(ds: Dict[str, Any]):
    """Return X_train, X_test, y_train, y_test according to spec."""
    seed = ds.get("seed", 42)
    algorithm_globals.random_seed = seed
    test_size = float(ds.get("test_size", 0.2))
    if not (0.05 <= test_size <= 0.5):
        test_size = 0.2  # keep it sane for small demos

    dtype = ds.get("type", "synthetic-line")

    # -------- Synthetic 2D line demo --------
    if dtype == "synthetic-line":
        n = int(ds.get("num_samples", 20))
        d = int(ds.get("num_features", 2))
        X = 2 * algorithm_globals.random.random([n, d]) - 1
        y = (np.sum(X, axis=1) >= 0).astype(int) * 2 - 1  # -1/+1
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    # -------- Iris (binary subset) --------
    elif dtype == "iris":
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

    # -------- CSV uploaded by user --------
    elif dtype == "csv":
        csv_path = ds.get("path")
        if not csv_path:
            raise ValueError("dataset.path is required for CSV")

        # allow relative "uploads/..." (relative to backend folder)
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("CSV appears to be empty")

        # ✅ Limit to 50 rows max to keep the demo fast
        df = df.sample(n=min(len(df), 50), random_state=42)

        label_col = ds.get("label_column")
        feat_cols = ds.get("feature_columns") or []

        # basic validation
        if label_col is None:
            raise ValueError("dataset.label_column is required for CSV")
        for c in [label_col, *feat_cols]:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not in CSV headers: {df.columns.tolist()}")

        X = df[feat_cols].values
        y = df[label_col].values

        # Convert labels to {-1,+1}
        if y.dtype.kind in "biuf":              # numeric labels
            # map 0/1 -> -1/+1 (or <=0 -> -1, >0 -> +1)
            uniq = np.unique(y)
            if set(uniq) <= {0, 1}:
                y = np.where(y == 1, 1, -1)
            else:
                y = np.where(y > 0, 1, -1)
        else:                                   # categorical labels
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(f"CSV label must be binary for this demo; got {classes.tolist()}")
            # map lexicographic max to +1 (purely arbitrary but consistent)
            y = np.where(y == classes.max(), 1, -1)

        return train_test_split(X, y, test_size=test_size, random_state=seed)

    else:
        raise ValueError(f"Unknown dataset type: {dtype}")


def run_pipeline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build and run an EstimatorQNN classifier pipeline described by 'spec'.
    Returns metrics: accuracy, (optional) predictions, sizes, and echo params.
    """
    # 1) Data
    Xtr, Xte, ytr, yte = _build_dataset(spec["dataset"])

    # ---- DEBUG: log shapes so we know we got here ----
    print(f"[QML] Train shapes: X={Xtr.shape}, y={ytr.shape}; Test: X={Xte.shape}, y={yte.shape}", flush=True)

    # 2) Encoder + Circuit via QNNCircuit (defaults: ZZFeatureMap + RealAmplitudes)
    # if num_qubits missing, default to num features from Xtr
    n_features = Xtr.shape[1]
    requested_qubits = int(spec["circuit"].get("num_qubits", n_features))
    # For this version, keep num_qubits = number of features
    num_qubits = n_features

    qc = QNNCircuit(num_qubits=num_qubits)

    # 3) QNN
    if spec["qnn"].get("type", "estimator") != "estimator":
        raise ValueError("Sprint-1 supports only qnn.type = 'estimator'")
    qnn = EstimatorQNN(circuit=qc, estimator=Estimator())

    # 4) Optimizer + Classifier
    if spec["optimizer"].get("type", "cobyla") != "cobyla":
        raise ValueError("Sprint-1 supports only optimizer.type = 'cobyla'")
    requested_maxiter = int(spec["optimizer"].get("maxiter", 60))
    # HARD CAP iterations to keep runtime short
    maxiter = max(1, min(requested_maxiter, 20))

    clf = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=maxiter))

    # 5) Train & evaluate
    print(f"[QML] Fitting... (num_qubits={num_qubits}, maxiter={maxiter})", flush=True)
    clf.fit(Xtr, ytr)
    print("[QML] Fit done. Scoring...", flush=True)

    acc = float(clf.score(Xte, yte))

    result = {
        "accuracy": acc,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        # helpful echoes for the UI/report:
        "seed": int(spec["dataset"].get("seed", 42)),
        "num_qubits": num_qubits,
        "maxiter": maxiter,
    }

    if spec.get("outputs", {}).get("return_predictions", True):
        preds = clf.predict(Xte).tolist()
        result["predictions"] = preds[:20]

    print("[QML] Done.", flush=True)
    return result
