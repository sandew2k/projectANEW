from flask import Flask, request, jsonify
from flask_cors import CORS
from runner import run_pipeline
from codegen import write_generated_run

app = Flask(__name__)
CORS(app)  # allow calls from the frontend http://localhost:8080

@app.post("/run")
def run():
    try:
        spec = request.get_json(force=True)
        metrics = run_pipeline(spec)
        code_path = write_generated_run(spec)
        return jsonify({**metrics, "generated_code_path": code_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)
