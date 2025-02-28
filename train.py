
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import datetime
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torchaudio
import moviepy.config as cfg

# Disable MoviePy logs globally
cfg.DEFAULT_LOGGER = None
warnings.filterwarnings("ignore")

from dataset.dataset import FakeAVCelebDataset
from network.graph_video_audio_model import GAT_video_audio

# **Set Device (CPU/GPU)**
Device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {Device}")

# **Define dataset paths**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "balanced_trainDataset1.csv")
TEST_CSV = os.path.join(BASE_DIR, "balanced_testDataset.csv")

# **Hyperparameters**
batch_size = 16  # Adjust for GPU/CPU
epochs = 15
num_classes = 4
learning_rate = 0.0005

# **Check dataset files**
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Training file not found: {TRAIN_CSV}")
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test file not found: {TEST_CSV}")

# **Fix Windows multiprocessing issue**
if __name__ == "__main__":
    # **Load datasets**
    train_dataset = FakeAVCelebDataset(TRAIN_CSV, BASE_DIR, phase="train")
    test_dataset = FakeAVCelebDataset(TEST_CSV, BASE_DIR, phase="test")

    # **Extract Labels from CSV**
    train_df = pd.read_csv(TRAIN_CSV)
    train_labels = train_df['label'].values 

    # **Class Counts for Weighted Loss**
    class_counts = np.bincount(train_labels, minlength=num_classes) 
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32) 
    class_weights /= class_weights.sum()  # Normalize weights

    # **Use WeightedRandomSampler for Handling Class Imbalance**
    sample_weights = torch.tensor([1.0 / (class_counts[label] + 1e-6) for label in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(train_labels), replacement=True)

    # **DataLoaders (Windows Fix: Use `num_workers=0`)**
    pin_memory = True if Device == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # **Load model**
    model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)

    # **Use weighted loss function**
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(Device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # AdamW for better regularization
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # **Save Paths**
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = os.path.join(BASE_DIR, "summary", "weight")
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"model_{time}.pth")

    # **Track loss & accuracy**
    train_losses, train_accuracies = [], []

    print(f"✅ Training on {Device}. Model will be saved at: {model_save_path}")

    # **Training loop**
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                video, audio, labels = batch
            except Exception as e:
                print(f"⚠️ Skipping batch due to error: {e}")
                continue  # Skip invalid batches

            video, audio, labels = video.to(Device), audio.to(Device), labels.to(Device)

            optimizer.zero_grad()
            outputs, _, _, _ = model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        scheduler.step()

        # **Save intermediate models every 2 epochs**
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Model checkpoint saved at {checkpoint_path}")

    # **Save final trained model**
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Training complete. Final model saved at {model_save_path}")

    # **Evaluation on Test Data**
    print("🔍 Running Model Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Evaluating on Test Set"):
            try:
                video, audio, labels = batch
            except Exception as e:
                print(f"⚠️ Skipping batch due to error: {e}")
                continue

            video, audio, labels = video.to(Device), audio.to(Device), labels.to(Device)
            outputs, _, _, _ = model(video, audio)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # **Classification Report**
    class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print("\n**Classification Report**\n")
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

    # **Save the Classification Report as CSV**
    report_csv_path = os.path.join(model_dir, f"classification_report_{time}.csv")
    df_report.to_csv(report_csv_path, index=True)
    print(f"✅ Classification report saved at: {report_csv_path}")

    # **Confusion Matrix**
    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    conf_matrix_path = os.path.join(model_dir, f"confusion_matrix_{time}.png")
    plt.savefig(conf_matrix_path)
    plt.show()

    print(f"✅ Confusion Matrix saved at: {conf_matrix_path}")
    print("✅ Test evaluation complete.")

