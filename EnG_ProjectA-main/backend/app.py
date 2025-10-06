from flask import Flask, request, jsonify
from flask_cors import CORS
from runner import run_pipeline
from codegen import write_generated_run
import traceback
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # allow calls from http://localhost:8080

# ---------- Upload config ----------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 521 * 1024 * 1024  # 50 MB limit (adjust if needed)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({"ok": True}), 200

@app.post("/upload")
def upload_csv():
    """Upload a CSV file and save it to backend/uploads."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part (expected form-data key 'file')"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not _allowed_file(file.filename):
            return jsonify({"error": "Only .csv files are allowed"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # quick sanity read + metadata
        df = pd.read_csv(save_path)
        if df.empty:
            os.remove(save_path)
            return jsonify({"error": "CSV is empty"}), 400

        cols = df.columns.tolist()
        preview = df.head(5).to_dict(orient="records")
        rel_path = os.path.relpath(save_path, os.path.dirname(__file__))  # e.g. "uploads/my.csv"

        return jsonify({
            "ok": True,
            "path": rel_path,
            "columns": cols,
            "rows": int(len(df)),
            "preview": preview
        }), 200

    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        return jsonify({"error": f"{e.__class__.__name__}: {str(e)}", "detail": tb}), 400

@app.post("/run")
def run():
    try:
        spec = request.get_json(force=True) or {}
        for key in ("dataset", "circuit", "qnn", "optimizer"):
            if key not in spec:
                return jsonify({"error": f"Missing '{key}' in spec"}), 400

        metrics = run_pipeline(spec)
        code_path = write_generated_run(spec)
        return jsonify({**metrics, "generated_code_path": code_path}), 200
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        return jsonify({"error": f"{e.__class__.__name__}: {str(e)}", "detail": tb}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)
