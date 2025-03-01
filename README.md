# Multimodal DeepFake Classification

A deep learning-based **multimodal DeepFake detection** model that classifies videos into **four categories**:
- **Real Video - Real Audio**
- **Real Video - Fake Audio**
- **Fake Video - Real Audio**
- **Fake Video - Fake Audio**

This project uses **Vision Transformer (ViT) for video processing**, **1D-CNN for audio processing**, and **Graph Attention Networks (GATs) for feature fusion**.

---

## 📂 Project Structure
```
FakeAVCeleb/
│── dataset/
│   ├── dataset.py
│   ├── audio_extraction.py
│   ├── video_frame_extraction.py
│── network/
│   ├── graph_video_audio_model.py
│   ├── video_processing_model.py
│   ├── audio_processing_model.py
│── summary/ (Stores training logs, models, confusion matrices)
│── train.py (Main training script)
│── requirements.txt (Dependencies)
│── README.md (Project documentation)
```

---

## 📥 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/Fine-Grained-DeepFake-Detection.git
cd Fine-Grained-DeepFake-Detection
```

### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📝 Dataset

The dataset used for training is **FakeAVCeleb**, which contains **Real and Fake Videos**.

| **Class**               | **Train Samples** | **Test Samples** |
|-------------------------|------------------|------------------|
| **Real Video - Real Audio** | 400              | 100              |
| **Real Video - Fake Audio** | 400              | 100              |
| **Fake Video - Real Audio** | 7791             | 1918             |
| **Fake Video - Fake Audio** | 8776             | 2081             |

---

## 🚀 Training

### Run the Training Script
```bash
python train.py
```

### Training Steps:
1. Extracts **frames and audio** from each video.
2. Processes **video using Vision Transformer (ViT)**.
3. Extracts **audio features using 1D-CNN**.
4. Fuses **video-audio embeddings using Graph Attention Networks (GATs)**.
5. Classifies into one of the **four classes**.

---

## 📊 Model Outputs
After training, the model generates:
- **Training Loss & Accuracy Plot** (`summary/training_plot.png`)
- **Classification Report** (`summary/classification_report.csv`)
- **Confusion Matrix** (`summary/confusion_matrix.png`)

---

## 📌 Model Architecture

### 🔹 Video Processing:
- Extracts **4 frames per video**.
- Passes through **Vision Transformer (ViT)**.

### 🔹 Audio Processing:
- Extracts **16kHz resampled audio waveform**.
- Passes through **1D-CNN-based audio model**.

### 🔹 Feature Fusion:
- Combines **audio and video embeddings** using **Graph Attention Networks (GATs)**.

### 🔹 Classification:
- Fully connected layers classify into one of **four classes**.

---

## 🛠️ Troubleshooting

### 1️⃣ Multiprocessing Issue (Windows)
**Error:**
```
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```
**Solution:**  
Modify `train.py`:
```python
if __name__ == "__main__":
    train()
```
Or set:
```python
num_workers=0  # in DataLoader
```

### 2️⃣ Audio Extraction Issue
**Error:**
```
Couldn't find appropriate backend to handle uri ...
```
**Solution:**  
Run:
```bash
python steps.py
```
If it works, check `dataset/audio_extraction.py`.

---

## 📝 Future Improvements
- Implement **real-time DeepFake detection**.
- Enhance **Lipsync-based forgery detection**.
- Improve **generalization across different datasets**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📌 References
- **[FakeAVCeleb Dataset](https://your-dataset-link.com)**
- **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)**
- **[1D-CNN for Audio Processing](https://arxiv.org/abs/1807.03418)**

---

### 🚀 Run `python train.py` and start detecting DeepFakes! 🎭
