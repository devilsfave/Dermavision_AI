# DermaVision AI

<p align="center">
  <img src="dermav.png" alt="DERMAVISION Logo" width="200"/>
</p>

<p align="center">
  <a href="#model-architecture">Model</a> •
  <a href="#training-methodology">Training</a> •
  <a href="#api-documentation">API</a> •
  <a href="#system-architecture">System</a> •
  <a href="#getting-started">Setup</a> •
  <a href="#results">Results</a>
</p>

---

## Overview

DermaVision is an end-to-end AI-powered web application that leverages deep learning for real-time skin lesion classification. The platform combines a custom-trained CNN model with Spatial Transformer Networks (STN) and transfer learning from MobileNetV2 to deliver accurate, accessible dermatological screening.

**Key Highlights:**
- 🧠 MobileNetV2 with Spatial Transformer Network achieving **97.45% training accuracy** and **87.27% validation accuracy**
- 🔬 Transfer learning with strategic fine-tuning of top 50 layers
- 📊 Comprehensive class imbalance handling with computed class weights
- 🚀 Multi-format deployment:  TensorFlow SavedModel, TFLite, and TensorFlow. js
- 🏥 Complete telemedicine platform with appointment booking and role-based dashboards
- 🌍 Designed to democratize dermatological care access across Africa

> 📓 **[View the Complete Training Notebook](compressed_Model_Training_Notebook.ipynb)** - Full implementation with code, outputs, and training logs

---

## Model Architecture

### Deep Learning Pipeline

The classification model is built on **MobileNetV2** architecture enhanced with a **Spatial Transformer Network (STN)** to handle variations in image capture (rotation, scaling, translation). MobileNetV2 was selected for its efficiency and suitability for mobile and embedded vision applications, loaded with ImageNet weights (`alpha=0.75`) for transfer learning.

```
Input Image (224x224x3)
        ↓
┌─────────────────────────────────────┐
│   MobileNetV2 Preprocessing         │
│   • tf.keras.applications. mobilenet │
│     . preprocess_input               │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   MobileNetV2 Base (ImageNet)       │
│   • alpha=0.75 (optimized size)     │
│   • Top 50 layers unfrozen          │
│   • Transfer learning enabled       │
│   • 1,915,501 total parameters      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Spatial Transformer Network       │
│   • Localization Network (Conv+Pool)│
│   • Grid Generator (Affine params)  │
│   • Bilinear Sampler                │
│   • L2 Regularization (0.001)       │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Classification Head               │
│   • GlobalAveragePooling2D          │
│   • BatchNormalization              │
│   • Dropout(0.4)                    │
│   • Dense(7) + L2(0.01) + Softmax   │
└─────────────────────────────────────┘
        ↓
    Prediction (7 classes)
```

### Model Parameters

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| **Total** | 1,915,501 (7. 31 MB) | - |
| **Trainable** | 1,662,397 (6.34 MB) | ✓ |
| **Non-trainable** | 253,104 (988.69 KB) | ✗ |

### Spatial Transformer Network (STN)

The STN dynamically transforms input feature maps, enabling the network to learn spatial invariance—critical for medical imaging where precise alignment impacts diagnostic accuracy. 

```python
# STN Architecture (from compressed_Model_Training_Notebook.ipynb)
def spatial_transformer_network(inputs):
    # Localization Network
    localization = Conv2D(16, (5,5), activation='relu', padding='same', 
                          kernel_regularizer=regularizers.l2(0.001))(inputs)
    localization = MaxPooling2D((2,2))(localization)
    localization = BatchNormalization()(localization)
    localization = Conv2D(32, (3,3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(0.001))(localization)
    localization = MaxPooling2D((2,2))(localization)
    localization = BatchNormalization()(localization)
    localization = Flatten()(localization)
    localization = Dense(64, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001))(localization)
    localization = Dropout(0.3)(localization)
    
    # Affine transformation parameters (identity initialization)
    theta = Dense(6, weights=[np.zeros((64, 6)), 
                              np.array([1, 0, 0, 0, 1, 0])])(localization)
    
    # Grid Generator + Bilinear Sampler
    output = Lambda(lambda x: stn(x))([theta, inputs])
    return output
```

| Component | Function |
|-----------|----------|
| **Localization Network** | Predicts 6 affine transformation parameters using Conv + Pooling layers |
| **Grid Generator** | Creates sampling grid based on transformation parameters |
| **Bilinear Sampler** | Warps input feature maps using differentiable interpolation |

### Dataset

**Primary Dataset:** HAM10000 (Human Against Machine with 10,000 training images)
- **Total Images:** 10,015 dermatoscopic images
- **Training Set:** 8,012 images (80%)
- **Validation Set:** 2,003 images (20%)
- **Split Strategy:** Stratified by diagnosis class (`random_state=42`)

| Class | Condition | Description |
|-------|-----------|-------------|
| AKIEC | Actinic Keratoses | Pre-cancerous lesions from sun damage (Bowen's disease) |
| BCC | Basal Cell Carcinoma | Common skin cancer with low metastasis |
| BKL | Benign Keratosis | Non-cancerous growths (keratosis-like lesions) |
| DF | Dermatofibroma | Benign skin tumors |
| NV | Melanocytic Nevi | Common moles |
| MEL | Melanoma | Aggressive form of skin cancer |
| VASC | Vascular Lesions | Blood vessel-associated lesions |

---

## Training Methodology

> 📓 **Full training implementation:** [`compressed_Model_Training_Notebook.ipynb`](compressed_Model_Training_Notebook.ipynb)

### Data Augmentation Strategy

Aggressive data augmentation was applied using `ImageDataGenerator` to enhance generalization and mitigate overfitting by simulating real-world imaging conditions:

```python
# From compressed_Model_Training_Notebook.ipynb - Cell 4
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications. mobilenet.preprocess_input,
    rotation_range=20,           # Random rotations ±20°
    width_shift_range=0.2,       # Horizontal shifts ±20%
    height_shift_range=0.2,      # Vertical shifts ±20%
    shear_range=0.2,             # Shear transformations
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True,        # Horizontal flipping
    vertical_flip=True,          # Vertical flipping
    fill_mode='nearest'          # Pixel fill strategy
)
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate optimization |
| **Initial Learning Rate** | 0.001 → 1e-5 (fine-tuning) | Two-phase training strategy |
| **Loss Function** | Categorical Cross-Entropy | Multi-class classification |
| **Epochs** | 50 | Sufficient convergence with early stopping |
| **Batch Size** | 64 | Optimized for GPU memory |
| **Class Weights** | Computed via `sklearn.utils.class_weight` | Addresses HAM10000 class imbalance |
| **L2 Regularization** | 0.01 (dense), 0.001 (STN) | Prevents overfitting |
| **Dropout Rate** | 0.4 (classification), 0.3 (STN) | Additional regularization |

### Callback Strategy

```python
# From compressed_Model_Training_Notebook. ipynb - Cell 7
callbacks = [
    # Save best model based on validation top-3 accuracy
    ModelCheckpoint(
        filepath='/content/drive/My Drive/MODEL_CORRECTION/model_unquant.keras',
        monitor='val_top_3_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    
    # Dynamic learning rate reduction
    ReduceLROnPlateau(
        monitor='val_top_3_accuracy',
        factor=0.5,
        patience=2,
        verbose=1,
        mode='max',
        min_lr=1e-9
    ),
    
    # TensorBoard for visualization
    TensorBoard(log_dir='./logs')
]

# Class weight computation for imbalanced dataset
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
```

### Metrics Tracked

| Metric | Purpose |
|--------|---------|
| **Categorical Accuracy** | Primary classification accuracy |
| **Top-3 Accuracy** | Clinically relevant—correct diagnosis in top 3 predictions |
| **Per-class Precision** | Accuracy of positive predictions per condition |
| **Per-class Recall** | Sensitivity for each skin condition |
| **F1-Score** | Harmonic mean of precision and recall |

---

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 97.45% |
| **Validation Accuracy** | 87.27% |

### Evaluation Methodology

The model was rigorously evaluated using: 

- **Classification Report:** Per-class precision, recall, F1-score, and support metrics
- **Confusion Matrix:** Visual analysis of prediction patterns and misclassification tendencies

The model demonstrates strong generalization from training data, making it reliable for preliminary skin disease diagnosis while maintaining clinical relevance through top-3 accuracy monitoring.

---

## Model Export & Deployment

### Multi-Format Export Strategy

The trained model was exported in three formats to maximize deployment flexibility:

| Format | Use Case | Path |
|--------|----------|------|
| **TensorFlow SavedModel** | Server-side inference, TensorFlow Serving | `model_unquant_savedmodel/` |
| **TensorFlow Lite (. tflite)** | Mobile/embedded devices | `model_unquant. tflite` |
| **TensorFlow.js** | Browser-based inference | `tfjs_model/` |

```python
# From compressed_Model_Training_Notebook.ipynb - Cell 7
# Save in SavedModel format
saved_model_path = '/content/drive/My Drive/MODEL_CORRECTION/model_unquant'
tf.saved_model.save(model, saved_model_path)

# TFLite conversion (optimized for mobile)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# TensorFlow.js conversion (for web deployment)
# tfjs. converters.save_keras_model(model, 'tfjs_model')
```

### Cloud Deployment

The TensorFlow Lite model is deployed as a RESTful API using **Flask** on **Google Cloud Run**:

- ⚡ **Auto-scaling:** Handles variable request volumes
- 🔄 **Zero-downtime updates:** Seamless model version updates
- 🌍 **Global CDN:** Low-latency responses worldwide
- 💰 **Cost-efficient:** Pay-per-use serverless architecture

---

## API Documentation

### API Structure

```
skin-disease-api/
├── app.py                  # Flask API server
├── model_unquant.tflite    # TFLite model (~10MB)
├── Dockerfile              # Container configuration
├── requirements. txt        # Python dependencies
└── . gcloudignore           # Cloud deployment ignore file
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
    {"class": "MEL", "confidence": 0.85, "label": "Melanoma"},
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
| **ML Training** | TensorFlow/Keras, Google Colab |
| **ML Inference** | TensorFlow Lite, TensorFlow.js |
| **API Framework** | Python Flask |
| **Deployment** | Google Cloud Run, Firebase Hosting |

### Project Structure

```
DERMAVISION/
├── src/
│   ├── app/                              # Next.js 13 app router
│   ├── components/                       # Reusable React components
│   ├── contexts/                         # React context providers
│   ├── Firebase/                         # Firebase configuration
│   ├── services/                         # API service layer
│   ├── types/                            # TypeScript type definitions
│   └── utils/                            # Utility functions
├── skin-disease-api/                     # Python ML API
│   ├── app.py                            # Flask server
│   ├── model_unquant.tflite              # Trained model
│   └── Dockerfile                        # Container config
├── compressed_Model_Training_Notebook.ipynb  # 📓 ML Training Notebook
├── public/                               # Static assets
└── firebase.json                         # Firebase configuration
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
git clone https://github.com/devilsfave/Dermavision_AI.git
cd Dermavision_AI
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

### Training Your Own Model

To retrain or fine-tune the model, open the training notebook: 

```bash
# Open in Google Colab or Jupyter
jupyter notebook compressed_Model_Training_Notebook.ipynb
```

Or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devilsfave/Dermavision_AI/blob/main/compressed_Model_Training_Notebook.ipynb)

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

## Impact & Vision

### Addressing Healthcare Gaps in Africa

DermaVision AI was developed to address the critical shortage of dermatologists across the African continent. By leveraging AI for preliminary skin cancer screening, this platform aims to:

- **Democratize Access:** Provide quality dermatological screening in underserved regions
- **Enable Early Detection:** Identify potentially malignant lesions before they progress
- **Support Healthcare Workers:** Assist non-specialist clinicians with AI-powered second opinions
- **Reduce Diagnostic Delays:** Offer immediate preliminary assessments via mobile devices

### Future Roadmap

- [ ] Geolocation for finding nearby dermatologists
- [ ] EHR (Electronic Health Records) integration
- [ ] Multilingual support (French, Swahili, Arabic)
- [ ] Offline capabilities with on-device TFLite inference
- [ ] Explainable AI (XAI) with Grad-CAM visualizations
- [ ] Federated learning for privacy-preserving model improvements

---

## Technical Innovation Summary

| Innovation | Implementation |
|------------|----------------|
| **Transfer Learning** | MobileNetV2 with ImageNet weights, selective fine-tuning of top 50 layers |
| **Spatial Invariance** | Custom STN layer with bilinear interpolation for robust feature extraction |
| **Class Imbalance** | Dynamic class weight computation via scikit-learn |
| **Multi-platform Deployment** | SavedModel, TFLite, TensorFlow.js exports |
| **Production-ready API** | Containerized Flask on Google Cloud Run |

---

## Reproducibility

All training code, hyperparameters, and model weights are available for full reproducibility:

| Resource | Location |
|----------|----------|
| **Training Notebook** | [`compressed_Model_Training_Notebook.ipynb`](compressed_Model_Training_Notebook.ipynb) |
| **Dataset** | [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| **Model Weights** | Available upon request |
| **API Code** | `skin-disease-api/` directory |

---

## Acknowledgments

- **Dataset:** [ISIC HAM10000](https://dataverse.harvard. edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **University:** University of Energy and Natural Resources, Sunyani, Ghana
- **Frameworks:** TensorFlow, Keras, Next.js, Firebase

---

## Contact

**Developer:** devilsfave  
**Email:** devilsfave39@gmail.com  
**Repository:** [github.com/devilsfave/Dermavision_AI](https://github.com/devilsfave/Dermavision_AI)

---

<p align="center">
  <strong>🌍 Leveraging AI to democratize access to dermatological care across Africa 🌍</strong>
</p>

<p align="center">
  <em>Built with passion for healthcare equity and computational innovation</em>
</p>
