# My MLOps Project

This project contains a machine learning application deployed using Docker, Amazon ECR, and Amazon ECS Fargate.

## 🚀 Features
- Dockerized ML application
- Deployment to AWS ECS Fargate
- GitHub Actions CI/CD pipeline
- Publicly accessible service

## 🛠 Tech stack
- Python
- Docker
- Amazon ECR (Elastic Container Registry)
- Amazon ECS Fargate
- GitHub Actions

## 📦 How to use

### Build & run locally
```bash
docker build -t my-ml-app .
docker run -p 5000:5000 my-ml-app
