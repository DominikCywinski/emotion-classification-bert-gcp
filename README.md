# Emotion Classification with fine-tuned BERT

This project is a **Fine-tuned BERT-based emotion classification model** deployed using **FastAPI** as the web framework, containerized with **Docker**, and hosted on **Google Cloud Platform (GCP)**. The project includes an **automated CI/CD pipeline** for seamless builds, tests, and deployments, as well as **unit tests** to ensure the reliability and accuracy of the model.

---

## Features
- **Emotion Classification**: Accepts text input and predicts the emotion using a fine-tuned BERT model.
- **FastAPI**: Provides an HTTP API to interact with the model.
- **Dockerized Deployment**: The app runs in a Docker container for portability and scalability.
- **GCP Integration**: Deployed to GCP Cloud Run for serverless and efficient hosting.
- **Automated CI/CD Pipeline**: Automatically builds, tests, and deploys the application with each push to the repository or pull request.
- **Unit Tests**: tests ensure the accuracy and reliability of the model.

---

## Requirements
- Python 3.8 or newer
- Docker
- Google Cloud SDK

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd emotion-classification-bert-gcp
```

### 2. Set Up a Virtual Environment (Optional for Local Testing)
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train model
```bash
python src/train.py
```

### 4. Interact with model (optional)
```bash
python src/predict.py
```
---

## Local Development

### Running the Application Locally
1. Start the FastAPI app:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080 &
   ```
2. Test the API:
   - Emotion prediction endpoint:
     ```bash
     curl -X POST http://127.0.0.1:8080/predict/ -H "Content-Type: application/json" -d '{"text": "I am very happy!"}'
     ```

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t emotion-classification:latest .
```

### 2. Run the Docker Container
```bash
docker run -d -p 8080:8080 emotion-classification:latest
```

### 3. Test the API
Use the same `curl` commands as in local development, replacing `127.0.0.1` with the container's IP if necessary.

---

## GCP Deployment

### 1. Authenticate with GCP
```bash
gcloud auth login
gcloud config set project <your_project_id>
```

### 2. Build and Push the Image to Google Container Registry
```bash
gcloud builds submit --tag gcr.io/<your_project_id>/predict_emotion
```

### 3. Deploy to Cloud Run
```bash
gcloud run --image gcr.io/<your_project_id>/predict_emotion \
  --platform managed \
  --region <region i.e europe-central2> \
  --allow-unauthenticated
```

### 4. Access the Service
After deployment, GCP will provide a public URL. Use it to send API requests:
```bash
curl -X POST https://<your-service-url>/predict/ -H "Content-Type: application/json" -d '{"text": "I feel sad today."}'
```
