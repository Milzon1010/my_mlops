# app.py (dengan contoh training + log model)
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "MLOps app with MLflow is running"

@app.route("/train")
def train():
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    return f"Model trained and logged with accuracy: {acc:.4f}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# .gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
mlruns/

# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]

# docker-compose.yml
version: '3'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns --host 0.0.0.0
    ports:
      - "5001:5000"
    volumes:
      - ./mlruns:/mlruns
      - ./mlflow.db:/mlflow.db

# requirements.txt
flask
mlflow
scikit-learn

# clean_rebuild.bat (Windows batch script untuk clean + rebuild)
docker-compose down
rmdir /S /Q mlruns
if exist mlflow.db del mlflow.db
docker system prune -a --volumes -f
docker-compose up --build
