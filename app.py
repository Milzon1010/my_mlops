from flask import Flask, jsonify
import mlflow

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "MLOps Project is running!"})

@app.route("/log")
def log_run():
    with mlflow.start_run():
        mlflow.log_param("example_param", 123)
        mlflow.log_metric("example_metric", 0.95)
    return jsonify({"status": "Run logged!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
