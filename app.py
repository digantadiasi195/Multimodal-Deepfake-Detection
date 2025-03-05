import streamlit as st
import os
import requests
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from network.graph_video_audio_model import GAT_video_audio
import torchaudio

# Fallback for missing imports
cv2 = None
mp = None
try:
    import cv2
    import moviepy.editor as mp
except ImportError as e:
    st.warning(f"Optional dependencies cv2 or moviepy not found: {e}. App will use pre-uploaded frames and torchaudio for audio. Upload frames as images.")
    def extract_frames_from_video(video_path, num_frames=4, image_size=128, frames=None):
        if frames is None:
            st.error("cv2 not available. Please upload pre-extracted frame images.")
            return None
        try:
            frame_tensors = []
            for frame_path in frames:
                img = Image.open(frame_path).convert("RGB").resize((image_size, image_size))
                frame = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
                frame_tensors.append(frame)
            return torch.stack(frame_tensors) if frame_tensors else None
        except Exception as e:
            st.error(f"Error processing uploaded frames: {e}")
            return None
else:
    def extract_frames_from_video(video_path, num_frames=4, image_size=128, frames=None):
        if frames is not None:
            frame_tensors = []
            for frame_path in frames:
                img = Image.open(frame_path).convert("RGB").resize((image_size, image_size))
                frame = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
                frame_tensors.append(frame)
            return torch.stack(frame_tensors) if frame_tensors else None
        cap = cv2.VideoCapture(video_path)
        frames_list = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, total_frames, max(1, total_frames // num_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (image_size, image_size))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames_list.append(frame)
        cap.release()
        return torch.stack(frames_list) if frames_list else None

def extract_audio_from_video(video_path, duration=5.0, sample_rate=16000):
    try:
        waveform, sr = torchaudio.load(video_path)
        if waveform.size(0) > 1:  # Stereo to mono
            waveform = waveform.mean(dim=0)
        target_samples = int(duration * sample_rate)
        if waveform.size(1) < target_samples:
            padding = target_samples - waveform.size(1)
            waveform = torch.cat((waveform, torch.zeros((1, padding))), dim=1)
        else:
            waveform = waveform[:, :target_samples]
        return waveform.squeeze(0)
    except Exception as e:
        st.error(f"Error extracting audio with torchaudio: {e}")
        return None

# **Set Device**
Device = "cuda" if torch.cuda.is_available() else "cpu"

# **Load or Download Model**
MODEL_PATH = "summary/weight/model_2025-03-05_17-56-30.pth"
os.makedirs("summary/weight", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found locally. Attempting to download...")
    try:
        response = requests.get("https://your-storage-link/model_2025-03-05_17-56-30.pth", timeout=10)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Load model
try:
    model = GAT_video_audio(num_classes=4, audio_nodes=4).to(Device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# **Class Names**
class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# **Main App**
def main():
    st.markdown("""
    <h1 style="
        background: linear-gradient(to right, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    ">🛡️ Multimodal Deepfake Detection System</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
    Harness the power of AI to detect synthetic media with precision
    </div>
    """, unsafe_allow_html=True)

    # **Video and Frame Upload**
    uploaded_file = st.file_uploader("📤 Upload Video (MP4)", type=["mp4"], help="Upload an MP4 video (Max 200MB)", accept_multiple_files=False)
    uploaded_frames = st.file_uploader("📤 Upload Pre-Extracted Frames (Images)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, help="Upload 4 frame images if OpenCV is unavailable")

    if uploaded_file or uploaded_frames:
        os.makedirs("temp", exist_ok=True)
        
        if uploaded_file:
            video_path = f"temp/{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(video_path)
            st.success("✅ Video Uploaded Successfully!")
        else:
            video_path = None

        with st.expander("🔍 Analysis Overview", expanded=True):
            st.write("• Frame Extraction: 4 key frames (or uploaded frames)")
            st.write("• Audio Analysis: Waveform & Spectrogram")
            st.write("• AI Model: Graph Attention Network")

        # **Extract Frames**
        st.subheader("🖼️ Video Frame Analysis")
        if uploaded_frames and len(uploaded_frames) >= 4:
            frames_tensor = extract_frames_from_video(None, frames=[f.name for f in uploaded_frames])
            for i, frame_file in enumerate(uploaded_frames):
                with open(f"temp/frame_{i}.png", "wb") as f:
                    f.write(frame_file.getbuffer())
                st.image(f"temp/frame_{i}.png", caption=f"Frame {i+1}", use_container_width=True)
        elif uploaded_file:
            frames_tensor = extract_frames_from_video(video_path)
            frame_cols = st.columns(4)
            for i, col in enumerate(frame_cols):
                frame = frames_tensor[i].permute(1, 2, 0).numpy()
                with col:
                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True, output_format="PNG")
        else:
            st.warning("Please upload a video or at least 4 frame images.")

        # **Extract Audio**
        st.subheader("🎵 Audio Analysis")
        if uploaded_file:
            audio_waveform = extract_audio_from_video(video_path)
            if audio_waveform is not None:
                st.write(f"Audio waveform shape: {audio_waveform.shape}")
                fig_wave, ax_wave = plt.subplots(figsize=(10, 2), facecolor='#f4f6f7')
                ax_wave.plot(audio_waveform.numpy(), color='#3498db', linewidth=1.5)
                ax_wave.set_title("Audio Waveform", fontsize=12)
                ax_wave.set_xlabel("Time")
                ax_wave.set_ylabel("Amplitude")
                st.pyplot(fig_wave)

                fig_spec, ax_spec = plt.subplots(figsize=(10, 3), facecolor='#f4f6f7')
                spec, freqs, t, im = ax_spec.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis', NFFT=256, noverlap=128)
                ax_spec.set_title("Audio Spectrogram", fontsize=12)
                ax_spec.set_xlabel("Time (s)")
                ax_spec.set_ylabel("Frequency (Hz)")
                plt.colorbar(im, ax=ax_spec, label="Intensity (dB)")
                plt.tight_layout()
                st.pyplot(fig_spec)
            else:
                st.warning("⚠️ No valid audio detected!")
        else:
            st.warning("Upload a video to analyze audio.")

        # **Analyze Button**
        if (uploaded_file and frames_tensor is not None) or (uploaded_frames and len(uploaded_frames) >= 4 and frames_tensor is not None):
            analyze_btn = st.button("🔍 Run DeepFake Detection", help="Initiate comprehensive multimodal analysis", use_container_width=True)
            if analyze_btn:
                with st.spinner('🚀 Running Advanced AI Analysis...'):
                    with torch.no_grad():
                        video_input = frames_tensor.unsqueeze(0).to(Device)
                        audio_input = audio_waveform.unsqueeze(0).to(Device) if audio_waveform is not None else torch.zeros(1, 64000).to(Device)
                        outputs = model(video_input, audio_input)

                    if isinstance(outputs, tuple) and len(outputs) >= 1:
                        probabilities = F.softmax(outputs[0], dim=1).cpu().numpy().flatten()
                    else:
                        probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
                    
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class] * 100

                    st.markdown("<h2 style='text-align: center;'>🎭 Detection Results</h2>", unsafe_allow_html=True)
                    with st.container():
                        st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🏷️ Predicted Category", class_names[predicted_class], f"{confidence:.2f}% Confidence")
                            st.progress(int(confidence))
                            risk_levels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
                            risk_index = min(int(confidence / 25), 3)
                            risk_color = "#2ecc71" if risk_index <= 1 else "#e74c3c"
                            st.markdown(f"<p style='color: {risk_color}; font-weight: bold;'>🚨 Risk Level: {risk_levels[risk_index]}</p>", unsafe_allow_html=True)
                        with col2:
                            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#f4f6f7')
                            sns.barplot(x=class_names, y=probabilities, ax=ax, palette="viridis")
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                            ax.set_title("Probability Distribution", fontsize=12)
                            ax.set_ylabel("Probability")
                            plt.tight_layout()
                            st.pyplot(fig)
                        st.download_button("📥 Download Detailed Report", data=pd.DataFrame({"Class": class_names, "Probability": [f"{p:.4f}" for p in probabilities]}).to_csv(index=False).encode("utf-8"), file_name="deepfake_detection_report.csv", mime="text/csv", help="Download the full analysis report", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please ensure a video is uploaded or 4 frame images are provided for analysis.")

# **Sidebar Configuration**
def sidebar():
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(to right, #3498db, #2ecc71);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 20px;
    ">
    <h2 style="color: white; margin: 0;">🛡️ DeepFake Guardian</h2>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### 🤖 Model Insights")
    try:
        st.sidebar.image("dataset/image.png", use_container_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Image not found. Please add 'dataset/image.png'.")
    st.sidebar.markdown("""
    #### Technical Details
    - **Architecture:** Graph Attention Networks
    - **Inputs:** Video Frames & Audio Waveforms
    - **Classes:** 4 Multimodal Categories
    """)
    st.sidebar.markdown("### 📊 Performance")
    st.sidebar.progress(85)
    st.sidebar.caption("85% Accuracy on Test Dataset")
    st.sidebar.markdown("### 👤 Developer Details")
    st.sidebar.markdown("**Diganta Diasi**")
    st.sidebar.markdown("📧 [Contact](mailto:digantadiasi7@gmail.com)")
    st.sidebar.markdown("📅 **Updated:** March 05, 2025")
    st.sidebar.markdown("### ⚠️ Disclaimer")
    with st.sidebar.expander("Important Notes"):
        st.write("- Results are probabilistic")
        st.write("- Not a definitive judgment")
        st.write("- Use as a decision-support tool")

if __name__ == "__main__":
    sidebar()
    main()
