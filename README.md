# DermaVision AI

<p align="center">
  <img src="dermav.png" alt="DERMAVISION Logo" width="200"/>
</p>

<p align="center">

<p align="center">
  <a href="#model-architecture">Model</a> •
  <a href="#api-documentation">API</a> •
  <a href="#system-architecture">System</a> •
  <a href="#getting-started">Setup</a> •
  <a href="#results">Results</a>
</p>

---

## Overview

DermaVision is an end-to-end AI-powered web application that leverages deep learning for real-time skin lesion classification. The platform combines a custom-trained CNN model with Spatial Transformer Networks (STN) to provide accurate preliminary skin disease diagnosis, integrated with a full telemedicine system for patient-doctor connectivity.

**Key Highlights:**
- 🧠 Custom deep learning model achieving **97.45% training accuracy** and **87.27% validation accuracy**
- 🔬 Spatial Transformer Network integration for improved spatial invariance
- 🚀 Scalable API deployed on Google Cloud Run
- 🏥 Complete telemedicine platform with appointment booking and role-based dashboards

---

## Model Architecture

### Deep Learning Pipeline

The classification model is built on a CNN architecture enhanced with a **Spatial Transformer Network (STN)** to handle variations in image capture (rotation, scaling, translation).

```
Input Image (224x224x3)
        ↓
┌─────────────────────────────────────┐
│   Spatial Transformer Network       │
│   • Localization Network            │
│   • Grid Generator                  │
│   • Sampler                         │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Convolutional Layers              │
│   • Feature Extraction              │
│   • Batch Normalization             │
│   • ReLU Activation                 │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Dense Layers + L2 Regularization  │
│   • Dropout for Regularization      │
│   • Softmax Output (7 classes)      │
└─────────────────────────────────────┘
        ↓
    Prediction
```

### Dataset

**Primary Dataset:** HAM10000 (Human Against Machine with 10,000 training images)

| Class | Condition | Description |
|-------|-----------|-------------|
| AKIEC | Actinic Keratoses | Pre-cancerous lesions from sun damage |
| BCC | Basal Cell Carcinoma | Common skin cancer with low metastasis |
| BKL | Benign Keratosis | Non-cancerous growths |
| DF | Dermatofibroma | Benign skin tumors |
| NV | Melanocytic Nevi | Common moles |
| MEL | Melanoma | Aggressive form of skin cancer |
| VASC | Vascular Lesions | Blood vessel-associated lesions |

### Data Preprocessing & Augmentation

```python
# Preprocessing Pipeline
- Image resizing:  224x224 pixels
- Pixel normalization: [0, 1] range
- Color balance standardization

# Augmentation Techniques
- Random rotation (±30°)
- Horizontal/Vertical flipping
- Random cropping
- Brightness adjustment (±20%)
- SMOTE for class imbalance
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 55 |
| Batch Size | 32 |
| Learning Rate | Adaptive (ReduceLROnPlateau) |
| Early Stopping | Validation loss patience:  10 |

---

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 97.45% |
| **Validation Accuracy** | 87.27% |

The model demonstrates strong generalization from training data, making it reliable for preliminary skin disease diagnosis. 

---

## API Documentation

### Deployment

The trained TensorFlow Lite model is deployed as a RESTful API using **Flask** on **Google Cloud Run**, providing:

- ⚡ **Scalability:** Auto-scaling based on request volume
- 🔄 **Zero-downtime updates:** Seamless model version updates
- 🌍 **Global accessibility:** Low-latency responses worldwide

### API Structure

```
skin-disease-api/
├── app. py                 # Flask API server
├── model_unquant. tflite   # TFLite model (~10MB)
├── Dockerfile             # Container configuration
├── requirements. txt       # Python dependencies
└── . gcloudignore          # Cloud deployment ignore file
```

### Endpoint

```http
POST /predict
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST \
  -F "image=@skin_lesion. jpg" \
  https://your-api-url/predict
```

**Response:**
```json
{
  "predictions": [
    {"class": "MEL", "confidence":  0.85, "label": "Melanoma"},
    {"class": "NV", "confidence":  0.10, "label": "Melanocytic Nevi"},
    {"class": "BKL", "confidence":  0.05, "label": "Benign Keratosis"}
  ],
  "top_prediction": "Melanoma",
  "severity": "High"
}
```

---

## System Architecture

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 13, React, TypeScript |
| **Styling** | Tailwind CSS, Framer Motion |
| **Backend** | Firebase (Auth, Firestore) |
| **ML API** | Python Flask, TensorFlow Lite |
| **Deployment** | Google Cloud Run, Firebase Hosting |

### Project Structure

```
DERMAVISION/
├── src/
│   ├── app/                 # Next.js 13 app router
│   ├── components/          # Reusable React components
│   ├── contexts/            # React context providers
│   ├── Firebase/            # Firebase configuration
│   ├── services/            # API service layer
│   ├── types/               # TypeScript type definitions
│   └── utils/               # Utility functions
├── skin-disease-api/        # Python ML API
│   ├── app.py               # Flask server
│   ├── model_unquant. tflite # Trained model
│   └── Dockerfile           # Container config
├── public/                  # Static assets
└── firebase.json            # Firebase configuration
```

### User Roles & Dashboards

| Role | Features |
|------|----------|
| **Patient** | Upload images, view analysis history, book appointments |
| **Doctor** | Manage appointments, view patient records, set availability |
| **Admin** | Verify doctors, manage users, view system statistics |

---

## Getting Started

### Prerequisites

- Node.js v18+
- Python 3.8+
- Firebase account
- Google Cloud account (for API deployment)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/devilsfave/NGUY_CURSORDIDIT. git
cd NGUY_CURSORDIDIT
```

**2. Install frontend dependencies**
```bash
npm install
# or
yarn install
```

**3. Configure environment variables**
```bash
cp .env. example .env. local
```

Add your Firebase configuration: 
```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_auth_domain
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_storage_bucket
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

**4. Start the development server**
```bash
npm run dev
```

**5. Run the ML API (separate terminal)**
```bash
cd skin-disease-api
pip install -r requirements. txt
python app. py
```

### Deploying the API to Google Cloud Run

```bash
cd skin-disease-api
gcloud builds submit --tag gcr.io/PROJECT_ID/skin-disease-api
gcloud run deploy --image gcr.io/PROJECT_ID/skin-disease-api --platform managed
```

---

## Security & Compliance

- 🔐 **Authentication:** Firebase Auth with role-based access control
- 🛡️ **Data Protection:** Secure data storage with Firestore security rules
- 📋 **Privacy:** HIPAA-compliant design principles
- 🔒 **API Security:** HTTPS encryption, CORS configuration

---

## Future Roadmap

- [ ] Geolocation for finding nearby dermatologists
- [ ] EHR (Electronic Health Records) integration
- [ ] Multilingual support
- [ ] Offline capabilities with on-device inference
- [ ] Explainable AI (XAI) for prediction transparency

---

## Acknowledgments

- **Dataset:** [ISIC HAM10000](https://dataverse.harvard.edu/dataset. xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **University:** University of Energy and Natural Resources, Sunyani

---

## Contact

**Email:** devilsfave39@gmail.com  
**Repository:** [github.com/devilsfave/NGUY_CURSORDIDIT](https://github.com/devilsfave/NGUY_CURSORDIDIT)

---

<p align="center">
  <em>Leveraging AI to democratize access to dermatological care</em>
</p>
