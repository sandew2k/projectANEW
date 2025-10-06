
// ---------- default spec (will switch to CSV after upload) ----------
const spec = {
  version: "0.1",
  pipeline: "qml-classifier",
  qnn: { type: "estimator" },
  dataset: {
    // you can keep synthetic *or* your diabetes CSV; caps below keep it fast anyway
    type: "csv",
    path: "uploads/diabetes.csv",
    label_column: "Outcome",
    feature_columns: [
      "Pregnancies","Glucose","BloodPressure","SkinThickness",
      "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ],
    test_size: 0.2,
    seed: 42
  },
  encoder: { type: "zzfeaturemap" },
  circuit: { type: "realamplitudes", num_qubits: 4, reps: 1 }, // cap default to 4
  optimizer: { type: "cobyla", maxiter: 20 },                  // cap default to 20
  outputs: { return_predictions: true, return_generated_code: true }
};

// ---------- UI refs ----------
const info   = document.getElementById("info");
const result = document.getElementById("result");
const msg    = document.getElementById("msg");
const fileEl = document.getElementById("csvFile");

const texts = {
  dataset: "Upload a CSV or use the demo synthetic data. CSV should have numeric features and a label/class column.",
  encoder: "Encoder (ZZFeatureMap): turns numbers into qubit rotations so the circuit can 'see' them.",
  circuit: "Circuit (QNNCircuit + RealAmplitudes): trainable pattern; optimizer tunes its angles.",
  optimizer: "Optimizer (COBYLA): knob-turner that reduces mistakes during training.",
  output: "Training & Output: .fit() learns; .score() prints accuracy; predictions returned."
};

// ---------- helpers ----------
function setMsg(text, type = "info") {
  msg.textContent = text || "";
  msg.style.color = type === "err" ? "#ef4444" : type === "ok" ? "#22c55e" : "#93a3b8";
}

function inferLabelAndFeatures(columns) {
  const lower = columns.map(c => String(c).toLowerCase());
  const candidates = ["label", "class", "target", "y", "outcome", "outcomes"];
  let labelIdx = lower.findIndex(c => candidates.includes(c));
  if (labelIdx === -1) labelIdx = columns.length - 1;
  const label = columns[labelIdx];
  const features = columns.filter((_, i) => i !== labelIdx);
  return { label, features };
}

async function uploadCSV(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch("http://localhost:5000/upload", { method: "POST", body: form });
  const data = await resp.json();
  if (!resp.ok) {
    const detail = data?.detail || data?.error || resp.statusText;
    throw new Error(`Upload failed: ${detail}`);
  }
  return data; // { ok, path, columns, rows, preview }
}

// ---------- button handlers ----------
document.getElementById("btn-dataset").onclick   = () => (info.textContent = texts.dataset);
document.getElementById("btn-encoder").onclick   = () => (info.textContent = texts.encoder);
document.getElementById("btn-circuit").onclick   = () => (info.textContent = texts.circuit);
document.getElementById("btn-optimizer").onclick = () => (info.textContent = texts.optimizer);
document.getElementById("btn-output").onclick    = () => (info.textContent = texts.output);

document.getElementById("btn-export").onclick = () => {
  const blob = new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "pipeline.json"; a.click();
  URL.revokeObjectURL(url);
};

document.getElementById("btn-upload").onclick = async () => {
  try {
    if (!fileEl.files || fileEl.files.length === 0) {
      setMsg("Please choose a .csv file first.", "err");
      return;
    }
    const f = fileEl.files[0];
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setMsg("Only .csv files are allowed.", "err");
      return;
    }

    setMsg("Uploading CSV…");
    const up = await uploadCSV(f);
    const { label, features } = inferLabelAndFeatures(up.columns);

    // switch dataset to use uploaded CSV
    spec.dataset = {
      type: "csv",
      path: up.path,
      label_column: label,
      feature_columns: features,
      test_size: 0.2,
      seed: 42
    };

    // CAP to keep the demo snappy
    spec.circuit.num_qubits = Math.max(1, Math.min(features.length, 4));
    spec.optimizer.maxiter  = Math.min(spec.optimizer.maxiter ?? 20, 20);

    setMsg(`Upload OK. Using "${label}" as label and ${features.length} feature(s). Rows: ${up.rows}.`, "ok");
    result.textContent = JSON.stringify({ upload: up, chosen: spec.dataset, circuit: spec.circuit, optimizer: spec.optimizer }, null, 2);
  } catch (e) {
    setMsg(String(e.message || e), "err");
  }
};

document.getElementById("btn-run").onclick = async () => {
  result.textContent = "Running...";
  setMsg("");

  const longRunNotice = setTimeout(() => {
    setMsg("Still running… large circuits can be slow. We cap qubits (≤4) and iterations (≤20) to keep it responsive.");
  }, 4000);

  try {
    // force a lightweight configuration before sending to backend
    spec.circuit = { type: "realamplitudes", num_qubits: 4, reps: 1 };
    spec.optimizer = { type: "cobyla", maxiter: 15 };

    const resp = await fetch("http://localhost:5000/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec)
    });

    const data = await resp.json();
    clearTimeout(longRunNotice);

    if (!resp.ok) {
      const detail = data?.detail || data?.error || resp.statusText;
      throw new Error(`Run failed: ${detail}`);
    }
    result.textContent = JSON.stringify(data, null, 2);
    setMsg("Pipeline finished.", "ok");
  } catch (e) {
    clearTimeout(longRunNotice);
    setMsg(String(e.message || e), "err");
    result.textContent = "";
  }
};
