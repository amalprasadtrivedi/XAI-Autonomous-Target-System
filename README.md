# 🎯 XAI-Autonomous-Target-System

> An intelligent, interpretable, and autonomous military-grade object detection system powered by Explainable AI (XAI).

---

## 🧠 Project Summary

The **XAI-Autonomous-Target-System** is a real-time AI-based military surveillance platform integrating modern object detection models like YOLOv5 and Faster R-CNN with state-of-the-art explainable AI (XAI) tools such as **Grad-CAM**, **LIME**, and **SHAP**.

This system enables **decision transparency** for high-stakes autonomous tasks and supports a **human-in-the-loop** strategy for target validation and trust building.

---

## 🗂️ Project Structure

```bash
XAI-Autonomous-Target-System/
│
├── app/                          # 💻 Streamlit Frontend UI
│   ├── main.py                   # Streamlit App Entry Point
│   ├── pages/                    # Multi-tab interface
│   │   ├── 1_Live_Detection.py
│   │   ├── 2_Explanation_Review.py
│   │   ├── 3_Operator_Logs.py
│   │   └── 4_Analytics_Dashboard.py
│   ├── components/
│   │   ├── ui_helpers.py
│   │   ├── display.py
│   │   └── charts.py
│   └── utils/
│       └── config.py
│
├── backend/                      # 🧠 ML & Explainability Logic
│   ├── detection/
│   │   ├── object_detector.py
│   │   ├── model_loader.py
│   │   └── postprocessing.py
│   ├── xai/
│   │   ├── grad_cam.py
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   └── services/
│       ├── explain_and_detect.py
│       └── logger.py
│
├── models/                       # 📦 Trained weights
│   ├── yolo/
│   ├── faster_rcnn/
│   └── checkpoints/
│
├── datasets/                     # 📂 Datasets and labels
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── samples/
│
├── evaluation/                   # 📊 Evaluation Scripts
│   ├── metrics.py
│   ├── trust_score.py
│   └── error_analysis.py
│
├── reports/                      # 📑 Logs & Research Docs
│   ├── session_logs.csv
│   ├── plots/
│   └── research_paper_draft.docx
│
├── notebooks/                    # 🧪 Prototypes and Demos
│   ├── train_model.ipynb
│   ├── xai_experiments.ipynb
│   └── evaluation_report.ipynb
│
├── requirements.txt              # 📦 Dependencies
├── README.md                     # 📖 Project Documentation
├── .env                          # 🔐 Environment Variables
└── setup.sh                      # 🔧 Setup Script
```

## 🚀 Features

- 🛰️ **Real-Time Object Detection**
  - Supports both **YOLOv5** and **Faster R-CNN** models.
  - Detects objects from live video feed, CCTV, or uploaded files.

- 🔍 **Explainable AI Visualizations**
  - Integrated **Grad-CAM** for heatmap-based visual insights.
  - **LIME** for local interpretable model-agnostic explanations.
  - **SHAP** to show pixel-wise feature attribution.

- 🧠 **Human-in-the-Loop Decision System**
  - Operators can review AI outputs and accept/reject decisions.
  - Logs feedback for false positives and false negatives.

- 📈 **Analytics Dashboard**
  - Displays trust scores, performance metrics, and model accuracy.
  - Visual comparison of human vs. AI decisions.

- 📊 **Evaluation Tools**
  - Includes accuracy, F1-score, precision, recall, and mAP.
  - Supports detailed false positive/negative analysis.

- 💬 **Session Review & Logging**
  - All operator interactions and model outputs are logged.
  - Enables transparency, accountability, and re-trainable feedback loops.

- 🔧 **Modular Backend Architecture**
  - Clean separation between detection, explanation, and service layers.
  - Easy to plug-and-play different models or explanation engines.

- 📦 **Model Checkpoints Management**
  - Organize multiple saved weights for different environments.
  - Easy switching between YOLO and R-CNN models via config.

- 🧪 **Interactive Notebooks**
  - Jupyter-based prototyping for detection and explainability.
  - Useful for research, experimentation, and model validation.

- 🌐 **Streamlit Frontend UI**
  - Multi-page, responsive interface for real-time interactions.
  - Operator-friendly design for military and surveillance use cases.


## 🖥️ Screenshots

| Live Detection | Grad-CAM Overlay | Trust Dashboard |
|----------------|------------------|-----------------|
| ![](https://placehold.co/200x120?text=Live+View) | ![](https://placehold.co/200x120?text=Grad-CAM) | ![](https://placehold.co/200x120?text=Analytics) |

> *(Replace placeholders with actual screenshots in your project repository)*

---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/XAI-Autonomous-Target-System.git
cd XAI-Autonomous-Target-System
```

## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ Configure Environment
```bash
cp .env.example .env
# Add your custom paths, model checkpoints, or API keys
```

## 4️⃣ Launch the App
```bash
streamlit run app/main.py
```

## 🔧 Components Overview

### 🎯 Object Detection

- `object_detector.py`  
  Handles object detection logic using YOLOv5 or Faster R-CNN models. Accepts image or video frames, returns detected bounding boxes, class labels, and confidence scores.

- `model_loader.py`  
  Loads and initializes pre-trained weights for YOLOv5 or Faster R-CNN. Supports dynamic model switching via configuration.

- `postprocessing.py`  
  Applies filtering, non-maximum suppression (NMS), and visual overlays (boxes, labels, scores) to the detection output.

---

### 💡 Explainable AI Modules

- `grad_cam.py`  
  Generates **Grad-CAM** heatmaps for convolutional neural networks. Highlights class-discriminative regions contributing to decisions.

- `lime_explainer.py`  
  Implements **LIME (Local Interpretable Model-agnostic Explanations)**. Explains predictions by perturbing input and training local interpretable models.

- `shap_explainer.py`  
  Uses **SHAP (SHapley Additive exPlanations)** to provide feature importance values. Works with both image and structured data.

---

### 📈 Evaluation and Trust Metrics

- `metrics.py`  
  Computes standard evaluation metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **mean Average Precision (mAP)**.

- `trust_score.py`  
  Calculates **trust score** by comparing model predictions with operator decisions. Helps measure AI-human alignment.

- `error_analysis.py`  
  Allows side-by-side visualization of **False Positives (FP)** and **False Negatives (FN)** for performance diagnosis.

---

### 🧪 Prototyping and Research

- `train_model.ipynb`  
  Jupyter notebook for training YOLO or Faster R-CNN models on your dataset. Includes data augmentation and checkpoint saving.

- `xai_experiments.ipynb`  
  Compare and visualize outputs from **Grad-CAM**, **LIME**, and **SHAP** on the same sample to analyze interpretability.

- `evaluation_report.ipynb`  
  Generates visual summaries, confusion matrices, and performance plots for internal reporting and presentations.

---

### 🌐 Frontend (Streamlit UI)

- `main.py`  
  Main Streamlit script to launch the interactive dashboard.

- `pages/1_Live_Detection.py`  
  UI page to run real-time detection using webcam or video file.

- `pages/2_Explanation_Review.py`  
  Shows XAI overlays with options to compare explanations side-by-side.

- `pages/3_Operator_Logs.py`  
  Displays a searchable table of all past session logs and operator feedback.

- `pages/4_Analytics_Dashboard.py`  
  Shows graphs, statistics, trust metrics, and system performance over time.

---

### 🧩 UI Components & Utilities

- `components/ui_helpers.py`  
  Contains reusable widgets like sliders, toggles, collapsible sections, and card layouts.

- `components/display.py`  
  Displays annotated images, overlays, and detection heatmaps in Streamlit.

- `components/charts.py`  
  Generates matplotlib and seaborn charts for dashboard visuals.

- `utils/config.py`  
  Centralized configuration file containing paths, thresholds, and model settings used across the application.

