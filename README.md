# RUL Forecasting for Batteries - Python Dev Container

[![Podman](https://img.shields.io/badge/Podman-4.9.3-892CA0?logo=podman)](https://podman.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/) 
[![Code-Server](https://img.shields.io/badge/Code--Server-4.16.1-007ACC?logo=visual-studio-code)](https://github.com/coder/code-server)

A containerized Python environment for battery Remaining Useful Life (RUL) prediction using machine learning models — accessible right from your browser using Code-Server.

This container setup is perfect for experimenting with real-world battery datasets and building ML models with a clean dev experience.

---

## ⚙️ Features

- 🧠 ML-ready Python environment (NumPy, Pandas, Scikit-learn, PyTorch/TensorFlow optional)
- 🧪 Jupyter & VS Code in-browser
- 🔄 Live development & hot reloading
- 📦 Pre-configured for RUL workflows (data loading, EDA, modeling)
- 🐳 Rootless Podman container
- 🔐 Isolated and reproducible dev setup

---

## 🚧 Prerequisites

- Podman 4.0+ ([Installation Guide](https://podman.io/getting-started/installation))  
- podman-compose 1.0+

```bash
sudo apt-get install podman podman-docker podman-compose
podman-compose build
podman-compose up -d
```

## 🚀Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aenodehi/RUL_Battery.git
cd RUL_Battery
```

### 2. Build and start the container
```bash
podman-compose up -d --build
```

### 3. Access the development environment
http://localhost:4000

## Project Workflow
```bash
# Start Jupyter inside the container
podman exec -it rul_container bash

# Inside the container
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# OR work directly in VS Code via Code-Server



```
