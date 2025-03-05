#app.py
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import torchaudio
import moviepy.editor as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from network.graph_video_audio_model import GAT_video_audio
from dataset.video_frame_extraction import extract_frames_from_video
from dataset.audio_extraction import extract_audio_from_video

# **Set Device**
Device = "cuda" if torch.cuda.is_available() else "cpu"

# **Load Pre-trained Model**
MODEL_PATH = "summary/model_2025-03-05_17-56-30.pth"
num_classes = 4

# Load model
model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
model.eval()

# **Class Names**
class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# **Streamlit UI Configuration**
st.set_page_config(
    page_title="DeepFake Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #2ecc71;
        --background-color: #f4f6f7;
        --text-color: #2c3e50;
    }
    
    .main {
        background: linear-gradient(135deg, var(--background-color) 0%, #e0e6ed 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(to right, rgba(255,255,255,0.9), rgba(241,245,249,0.9));
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 12px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 20px;
        border: none;
        padding: 12px 28px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stFileUploader {
        background-color: RGBA(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        border: 3px dashed var(--primary-color);
        transition: border-color 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--secondary-color);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
    }
    
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
    }
    
    .block-container {
        padding: 2rem 1rem;
    }
    
    .frame-box {
        background: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .frame-box:hover {
        transform: scale(1.05);
    }
    
    .result-box {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    
    .result-box .stMetric, .result-box .stProgress, .result-box .stDownloadButton {
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# **Main App**
def main():
    # Title with gradient effect
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

    # **Video Upload**
    uploaded_file = st.file_uploader(
        "📤 Upload Video", 
        type=["mp4"], 
        help="Upload an MP4 video for deepfake analysis (Max 200MB)",
        accept_multiple_files=False
    )

    if uploaded_file:
        # **Ensure Temp Directory Exists**
        os.makedirs("temp", exist_ok=True)
        
        # **Save Uploaded File**
        video_path = f"temp/{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Advanced Video Display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(video_path)
        
        with col2:
            st.success("✅ Video Uploaded Successfully!")
            with st.expander("🔍 Analysis Overview", expanded=True):
                st.write("• Frame Extraction: 4 key frames")
                st.write("• Audio Analysis: Waveform & Spectrogram")
                st.write("• AI Model: Graph Attention Network")

        # **Extract Frames**
        st.subheader("🖼️ Video Frame Analysis")
        frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
        
        # Display frames with hover effect
        frame_cols = st.columns(4)
        for i, col in enumerate(frame_cols):
            frame = frames_tensor[i].permute(1, 2, 0).numpy()
            with col:
                st.markdown(f"<div class='frame-box'>", unsafe_allow_html=True)
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True, output_format="PNG")
                st.markdown("</div>", unsafe_allow_html=True)

        # **Extract Audio**
        st.subheader("🎵 Audio Analysis")
        audio_waveform = extract_audio_from_video(video_path)
        if audio_waveform is not None:
            # Debug: Print audio shape
            st.write(f"Audio waveform shape: {audio_waveform.shape}")
            
            # Convert stereo to mono
            if audio_waveform.dim() > 1 and audio_waveform.size(0) == 2:
                audio_waveform = audio_waveform.mean(dim=0)
            
            # Waveform
            fig_wave, ax_wave = plt.subplots(figsize=(10, 2), facecolor='#f4f6f7')
            ax_wave.plot(audio_waveform.numpy(), color='#3498db', linewidth=1.5)
            ax_wave.set_title("Audio Waveform", fontsize=12)
            ax_wave.set_xlabel("Time")
            ax_wave.set_ylabel("Amplitude")
            st.pyplot(fig_wave)

            # Spectrogram with corrected plotting
            fig_spec, ax_spec = plt.subplots(figsize=(10, 3), facecolor='#f4f6f7')
            spec, freqs, t, im = ax_spec.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis', NFFT=256, noverlap=128)
            ax_spec.set_title("Audio Spectrogram", fontsize=12)
            ax_spec.set_xlabel("Time (s)")
            ax_spec.set_ylabel("Frequency (Hz)")
            plt.colorbar(im, ax=ax_spec, label="Intensity (dB)")
            plt.tight_layout()
            st.pyplot(fig_spec)
        else:
            st.warning("⚠️ No valid audio detected! Please ensure the video contains audio.")

        # **Analyze Button with Advanced UI**
        col_analyze, col_explain = st.columns([1, 1])
        
        with col_analyze:
            analyze_btn = st.button(
                "🔍 Run DeepFake Detection", 
                help="Initiate comprehensive multimodal analysis",
                use_container_width=True
            )
        
        with col_explain:
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.7); 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center;
                        border: 2px solid #3498db;">
            🤖 Powered by Graph Attention Networks for precise video & audio analysis
            </div>
            """, unsafe_allow_html=True)

        # **Run Analysis**
        if analyze_btn:
            with st.spinner('🚀 Running Advanced AI Analysis...'):
                # **Run Model Prediction**
                with torch.no_grad():
                    video_input = frames_tensor.unsqueeze(0).to(Device)
                    # Convert stereo to mono for model input
                    if audio_waveform.dim() > 1 and audio_waveform.size(0) == 2:
                        audio_waveform = audio_waveform.mean(dim=0)
                    audio_input = audio_waveform.unsqueeze(0).to(Device)  # Shape: (1, audio_samples)
                    outputs = model(video_input, audio_input)

                # **Get Prediction** (Adjust based on model output structure)
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    probabilities = F.softmax(outputs[0], dim=1).cpu().numpy().flatten()
                else:
                    probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
                
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class] * 100

                # Results Display with Advanced Layout
                st.markdown("<h2 style='text-align: center;'>🎭 Detection Results</h2>", unsafe_allow_html=True)
                with st.container():  # Use container to ensure proper rendering
                    st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
                    
                    # Detailed Result Columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="🏷️ Predicted Category", 
                            value=class_names[predicted_class], 
                            delta=f"{confidence:.2f}% Confidence",
                            delta_color="normal"
                        )
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

                    # Download Button
                    st.download_button(
                        "📥 Download Detailed Report", 
                        data=pd.DataFrame({
                            "Class": class_names,
                            "Probability": [f"{p:.4f}" for p in probabilities]
                        }).to_csv(index=False).encode("utf-8"),
                        file_name="deepfake_detection_report.csv",
                        mime="text/csv",
                        help="Download the full analysis report",
                        use_container_width=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# **Sidebar Configuration**
def sidebar():
    # Fancy Sidebar Header
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

    # Model Information
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
    
    # Performance Metrics
    st.sidebar.markdown("### 📊 Performance")
    st.sidebar.progress(85)
    st.sidebar.caption("Our Model Accuracy is 95.85%")
    
    # Developer & Contact
    st.sidebar.markdown("### 👤 Developer Details")
    st.sidebar.markdown("**Diganta Diasi**")
    st.sidebar.markdown("📧 [Contact](mailto:digantadiasi7@gmail.com)")
    st.sidebar.markdown("📅 **Updated:** March 05, 2025")
    
    # Disclaimers
    st.sidebar.markdown("### ⚠️ Disclaimer")
    with st.sidebar.expander("Important Notes"):
        st.write("- Results are probabilistic")
        st.write("- Not a definitive judgment")
        st.write("- Use as a decision-support tool")

# Run the app
if __name__ == "__main__":
    sidebar()
    main()
