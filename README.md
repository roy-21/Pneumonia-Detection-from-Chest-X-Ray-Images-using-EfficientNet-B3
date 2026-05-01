# 🫁 Chest X-Ray Pneumonia Classifier

[![CI/CD Pipeline](https://github.com/yourusername/chest-xray-pneumonia-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/chest-xray-pneumonia-classifier/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> A Deep Learning system for detecting Pneumonia from Chest X-Ray images, built with production-grade ML engineering standards.

## 📖 Overview

This project implements a binary image classification system (Normal vs Pneumonia) using transfer learning with an EfficientNet-B3 backbone. The model was developed with a focus on medical AI standards: prioritizing recall (sensitivity) to minimize false negatives, and providing model explainability via Grad-CAM heatmaps.

### Key Features
- **Transfer Learning & Fine-Tuning**: EfficientNet-B3 pre-trained on ImageNet.
- **Data Engineering**: Addressed class imbalance using `WeightedRandomSampler` and robust data augmentation.
- **Explainability (XAI)**: Integrated Grad-CAM to visualize regions of the X-Ray influencing the model's prediction.
- **Web Interface**: A locally deployed, premium Glassmorphism UI for doctors/users to easily test X-Rays.
- **MLOps**: Experiment tracking and model registry managed via MLflow. Data version control via DVC.
- **Deployment**: Containerized FastAPI backend ready for cloud deployment (HuggingFace Spaces, Cloud Run, Render).
- **CI/CD**: Automated testing and Docker image builds using GitHub Actions.

## 📊 Dataset

The dataset used is the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- **Size**: 5,856 JPEG images
- **Classes**: `NORMAL`, `PNEUMONIA`

## 🚀 Quick Start (Docker)

The fastest way to get the API running locally is via Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/chest-xray-pneumonia-classifier.git
cd chest-xray-pneumonia-classifier

# Start the API and MLflow tracking server
docker-compose up --build
```
- **Web Interface**: [http://localhost:8000/](http://localhost:8000/) (Upload X-Rays here)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **MLflow**: [http://localhost:5000](http://localhost:5000)

## 🛠️ Local Development Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Dataset**:
   Download the Kaggle dataset and place it in the `chest_xray/` directory.

4. **Train the Model**:
   ```bash
   # Stage 1: Train classifier head (frozen backbone)
   python src/train.py --arch efficientnet_b3
   
   # Stage 2: Fine-tune backbone (differential learning rates)
   python src/train.py --arch efficientnet_b3 --finetune
   ```

6. **Evaluate the Model**:
   ```bash
   python -m src.evaluate
   ```

7. **Run the API & Web App**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Access the web app at `http://localhost:8000`*

## 📁 Repository Structure

```text
chest-xray-pneumonia-classifier/
├── .github/workflows/   # CI/CD pipelines
├── app/                 # FastAPI application & Frontend
│   ├── main.py          # API Endpoints
│   ├── schemas.py       # Pydantic data models
│   └── static/          # Frontend Web Assets
│       └── index.html   # Glassmorphism UI
├── configs/             # Configuration files
│   └── config.yaml      # Centralized hyperparameters
├── src/                 # ML Source Code
│   ├── dataset.py       # Data loading, augmentation, and sampling
│   ├── model.py         # Architectures (EfficientNet, ResNet, etc.)
│   ├── train.py         # Training loop & MLflow tracking
│   ├── evaluate.py      # Evaluation metrics and plots
│   ├── predict.py       # Inference pipeline
│   └── grad_cam.py      # Explainability (Heatmaps)
├── Dockerfile           # Docker configuration for the API
├── docker-compose.yml   # Multi-container setup (API + MLflow)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 📈 ML Engineering Principles Followed

1. **Config-Driven**: No hardcoded values; managed via `config.yaml`.
2. **Reproducibility**: Global seed setting (`torch.manual_seed(42)`).
3. **Imbalance Handling**: Dynamic class weighting in loss function and data sampling.
4. **Medical Priority**: Evaluation emphasizes Recall/Sensitivity over raw accuracy.
5. **Transparency**: Grad-CAM heatmaps generated for every prediction.

## 👤 Author
**Sojib Chandra Roy**  
📧 Email: rcsojib.cse1@gmail.com

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
