# Multimodal DeepFake Classification

## 📌 Project Overview
This project focuses on **Fine-Grained Multimodal DeepFake Classification** using the **FakeAVCeleb** dataset. The model is designed to classify videos into four categories:
- **RealVideo-RealAudio**
- **RealVideo-FakeAudio**
- **FakeVideo-RealAudio**
- **FakeVideo-FakeAudio**

The classification is achieved by processing both **video** (frames) and **audio** (waveforms) using deep learning architectures.

---

## 📂 Directory Structure
```
FakeAVCeleb/
├── dataset/
│   ├── dataset.py  # Dataset processing class
│   ├── audio_extraction.py  # Extracts audio from video
│   ├── video_frame_extraction.py  # Extracts frames from video
│
├── network/
│   ├── audio_processing_model.py  # Processes audio input
│   ├── video_processing_model.py  # Processes video input
│   ├── graph_video_audio_model.py  # Main model integrating video & audio features
│
├── summary/
│   ├── weight/  # Stores trained model checkpoints
│   ├── training_logs/  # Stores logs & evaluation reports
│
├── train.py  # Main training script
├── steps.py  # Script to test single video/audio processing
├── requirements.txt  # Python dependencies
├── README.md  # Project documentation
```

---

## 📦 Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

Ensure you have **PyTorch**, **MoviePy**, **TorchAudio**, and **OpenCV** installed.

---

## 🚀 Training the Model
To train the model, run:
```bash
python train.py
```
This will:
- Load the dataset.
- Train the model using a **Graph Attention Network (GAT)**.
- Save checkpoints every 2 epochs.
- Log accuracy, loss, and evaluation metrics.

---

## 📊 Model Evaluation & Results
After training, the model is evaluated on the **test dataset**.

### ✅ **Final Training Metrics**
```
Epoch [15/15] - Loss: 0.1584, Train Acc: 93.78%
```

### 🔍 **Classification Report**
| Class                      | Precision | Recall  | F1-score | Support |
|----------------------------|-----------|---------|----------|---------|
| RealVideo-RealAudio        | 0.897988  | 0.99590 | 0.92775  | 2000    |
| RealVideo-FakeAudio        | 0.865083  | 0.98741 | 0.92270  | 2000    |
| FakeVideo-RealAudio        | 0.956383  | 0.98399 | 0.97239  | 2000    |
| FakeVideo-FakeAudio        | 0.970878  | 0.97962 | 0.97522  | 2000    |
| **Accuracy**               |           | 0.95237 |          | 8000    |
| **Macro Avg**              | 0.922083  | 0.95237 | 0.94926  | 8000    |
| **Weighted Avg**           | 0.922083  | 0.95237 | 0.94926  | 8000    |

### 📉 **Confusion Matrix**
![Confusion Matrix](summary/weight/confusion_matrix.png)

### 📜 **Logs & Reports**
- [Classification Report](summary/weight/classification_report.csv)
- [Training Loss & Accuracy Plots](summary/weight/training_plot.png)

---

## 💡 Future Work
- Implementing **Larger Transformer-Based Models** (e.g., ViT, Wav2Vec).
- Using **Pretrained Video Models** (e.g., SlowFast, TimeSformer).
- Improving **Cross-Modal Attention** mechanisms.

---

## 🤝 Contributing
Feel free to raise issues or contribute via pull requests.

---

## 📜 License
MIT License © 2025
