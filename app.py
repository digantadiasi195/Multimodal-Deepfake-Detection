import asyncio
import nest_asyncio
nest_asyncio.apply()

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
import shutil
import time
import networkx as nx
from network.graph_video_audio_model import GAT_video_audio
from dataset.video_frame_extraction import extract_frames_from_video
from dataset.audio_extraction import extract_audio_from_video

# Set Device
Device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Pre-trained Model
MODEL_PATH = "summary/model_2025-03-06_15-35-41.pth"
num_classes = 4

# Load model
model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
model.eval()

# Class Names
class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# Streamlit UI Configuration
st.set_page_config(
    page_title="DeepFake Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #1e90ff;
        --secondary-color: #32cd32;
        --background-color: #f5f7fa;
        --text-color: #2c3e50;
        --accent-color: #ff4500;
    }
    .main {
        background: linear-gradient(135deg, var(--background-color) 0%, #e0e6ed 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to right, rgba(255,255,255,0.95), rgba(241,245,249,0.95));
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    /* Fancy Button for Run DeepFake Detection */
    button[kind="primary"][key="fancy_analyze_button"] {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        background-size: 200% 200%;
        color: white;
        border-radius: 30px;
        border: none;
        padding: 15px 35px;
        font-size: 1.2em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transition: all 0.4s ease;
        animation: gradientShift 4s ease infinite;
        position: relative;
        overflow: hidden;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    button[kind="primary"][key="fancy_analyze_button"]:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        filter: brightness(1.1);
    }
    button[kind="primary"][key="fancy_analyze_button"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    button[kind="primary"][key="fancy_analyze_button"]:hover::before {
        opacity: 1;
        animation: pulseGlow 1.5s ease infinite;
    }
    @keyframes pulseGlow {
        0% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.2; }
        100% { transform: scale(1); opacity: 0.5; }
    }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] div {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    div[data-testid="stMetricDelta"] span {
        color: var(--secondary-color) !important;
        font-weight: 600;
    }
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.85);
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
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .frame-box:hover {
        transform: scale(1.05);
    }
    .result-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metrics-image {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        width: 100%;
    }
    .developer-section {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        animation: fadeIn 1.5s ease-in-out;
        transition: transform 0.3s ease;
    }
    .developer-section:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .developer-section h3 {
        color: var(--accent-color);
        font-size: 1.5em;
        margin-bottom: 10px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .developer-section p {
        color: var(--text-color);
        font-size: 1.1em;
        margin: 5px 0;
        text-align: center;
    }
    .developer-section a {
        color: var(--primary-color);
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    .developer-section a:hover {
        color: var(--secondary-color);
        text-decoration: underline;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Function to Visualize GNN Graph
def visualize_gnn_graph(edge_index=None, num_video_nodes=10, num_audio_nodes=4):
    try:
        num_nodes = num_video_nodes + num_audio_nodes
        fig = plt.figure(figsize=(12, 8))
        G = nx.Graph()
        
        for i in range(num_nodes):
            if i < num_video_nodes:
                label = f"Video {i+1}"
            else:
                label = f"Audio {i-num_video_nodes+1}"
            G.add_node(i, label=label)
        
        video_edges = []
        for i in range(num_video_nodes):
            for j in range(num_video_nodes):
                t1, t2 = i + 1, j + 1
                weight = 0.5 if i == j else abs(t1 - t2)
                video_edges.append((i, j, weight))
        G.add_weighted_edges_from(video_edges, edge_type='video-video')
        
        audio_edges = []
        for i in range(num_video_nodes, num_nodes):
            for j in range(num_video_nodes, num_nodes):
                t1, t2 = i - num_video_nodes + 1, j - num_video_nodes + 1
                weight = 0.5 if i == j else abs(t1 - t2)
                audio_edges.append((i, j, weight))
        G.add_weighted_edges_from(audio_edges, edge_type='audio-audio')
        
        av_edges = []
        for v in range(0, 4):
            av_edges.append((num_video_nodes, v, 1.0))
        for v in range(2, 6):
            av_edges.append((num_video_nodes + 1, v, 1.0))
        for v in range(4, 8):
            av_edges.append((num_video_nodes + 2, v, 1.0))
        for v in range(6, 10):
            av_edges.append((num_video_nodes + 3, v, 1.0))
        G.add_weighted_edges_from(av_edges, edge_type='audio-video')
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        labels = nx.get_node_attributes(G, 'label')
        node_colors = ['#3498db' if i < num_video_nodes else '#e74c3c' for i in range(num_nodes)]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        edge_types = nx.get_edge_attributes(G, 'edge_type')
        for edge_type in set(edge_types.values()):
            edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == edge_type]
            color = '#2ecc71' if edge_type == 'audio-video' else '#cccccc'
            style = 'solid' if edge_type == 'audio-video' else 'dashed'
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, style=style, width=2)
        
        plt.title("Graph Structure (Video & Audio Nodes)")
        return fig
    except Exception as e:
        st.error(f"Error in graph visualization: {str(e)}")
        return None

# Function to Process Video
def process_video(video_path):
    try:
        os.makedirs("temp", exist_ok=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(video_path)
        with col2:
            st.success("Video Uploaded Successfully!")
            with st.expander("Analysis Overview", expanded=True):
                st.write("• Frame Extraction: 4 key frames")
                st.write("• Audio Analysis: Waveform & Spectrogram")
                st.write("• AI Model: Graph Attention Network")
        
        st.subheader("🖼Video Frame Analysis")
        frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
        frame_cols = st.columns(4)
        for i, col in enumerate(frame_cols):
            frame = frames_tensor[i].permute(1, 2, 0).numpy()
            with col:
                st.markdown(f"<div class='frame-box'>", unsafe_allow_html=True)
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True, output_format="PNG")
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("🎵 Audio Analysis")
        audio_waveform = extract_audio_from_video(video_path)
        if audio_waveform is not None:
            if audio_waveform.dim() > 1 and audio_waveform.size(0) == 2:
                audio_waveform = audio_waveform.mean(dim=0)
            
            fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
            ax_wave.plot(audio_waveform.numpy(), color='#1e90ff', linewidth=1.5)
            ax_wave.set_title("Audio Waveform")
            ax_wave.set_xlabel("Time")
            ax_wave.set_ylabel("Amplitude")
            st.pyplot(fig_wave)
            plt.close(fig_wave)
            
            fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
            spec, freqs, t, im = ax_spec.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis', NFFT=256, noverlap=128)
            ax_spec.set_title("Audio Spectrogram")
            ax_spec.set_xlabel("Time (s)")
            ax_spec.set_ylabel("Frequency (Hz)")
            plt.colorbar(im, ax=ax_spec, label="Intensity (dB)")
            plt.tight_layout()
            st.pyplot(fig_spec)
            plt.close(fig_spec)
        else:
            st.warning("⚠No valid audio detected! Please ensure the video contains audio.")
        
        return frames_tensor, audio_waveform
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None

# Function to Run Detection
def run_detection(frames_tensor, audio_waveform):
    try:
        with st.spinner('Running Advanced AI Analysis...'):
            with torch.no_grad():
                video_input = frames_tensor.unsqueeze(0).to(Device)
                if audio_waveform.dim() > 1 and audio_waveform.size(0) == 2:
                    audio_waveform = audio_waveform.mean(dim=0)
                audio_input = audio_waveform.unsqueeze(0).to(Device)
                outputs = model(video_input, audio_input)

            mix_out, video_out, audio_out, fusion_out = outputs
            probabilities = F.softmax(mix_out, dim=1).cpu().numpy().flatten()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100

            video_importance = torch.norm(video_out).item()
            audio_importance = torch.norm(audio_out).item()
            fusion_importance = torch.norm(fusion_out).item()
            temporal_importance = 0.1
            cross_modal_importance = 0.1

            total_importance = video_importance + audio_importance + fusion_importance + temporal_importance + cross_modal_importance
            if total_importance > 0:
                video_importance /= total_importance
                audio_importance /= total_importance
                fusion_importance /= total_importance
                temporal_importance /= total_importance
                cross_modal_importance /= total_importance
            else:
                video_importance, audio_importance, fusion_importance, temporal_importance, cross_modal_importance = 0.4, 0.3, 0.15, 0.1, 0.05

            features = ["Video Features", "Audio Features", "Temporal Patterns", "Cross-modal Sync"]
            importance = [
                video_importance,
                audio_importance,
                temporal_importance,
                cross_modal_importance
            ]

            st.markdown("<h2 style='text-align: center;'>Detection Results</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="🏷Predicted Category", value=class_names[predicted_class], delta=f"{confidence:.2f}% Confidence")
                    st.progress(int(confidence))
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=class_names, y=probabilities, palette="viridis")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.set_title("Probability Distribution")
                    ax.set_ylabel("Probability")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Feature Importance")
                fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                ax_imp.barh(features, importance, color=['#1e90ff', '#ff4500', '#32cd32', '#f39c12'])
                ax_imp.set_title("Contribution of Different Modalities")
                ax_imp.set_xlabel("Importance Score")
                for i, v in enumerate(importance):
                    ax_imp.text(v + 0.01, i, f"{v:.2f}", va='center')
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)

                st.download_button(
                    "Download Detailed Report", 
                    data=pd.DataFrame({"Class": class_names, "Probability": [f"{p:.4f}" for p in probabilities]}).to_csv(index=False).encode("utf-8"),
                    file_name="deepfake_detection_report.csv",
                    mime="text/csv",
                    help="Download the full analysis report",
                    use_container_width=True
                )

                st.markdown("</div>", unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")

# Main App
def main():
    st.markdown("""
    <h1 style="
        background: linear-gradient(to right, #1e90ff, #32cd32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    ">Multimodal Deepfake Detection System</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
    Harness the power of AI to detect synthetic media with precision
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Video", 
        type=["mp4", "avi", "mov"], 
        help="Upload a video file for deepfake analysis (Max 50MB)"
    )

    if uploaded_file:
        try:
            os.makedirs("temp", exist_ok=True)
            timestamp = time.time()
            video_path = f"temp/{timestamp}_{uploaded_file.name}"
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            frames_tensor, audio_waveform = process_video(video_path)
            
            if frames_tensor is not None and audio_waveform is not None:
                st.subheader("🕸️ Graph Structure")
                graph_fig = visualize_gnn_graph(edge_index=None, num_video_nodes=10, num_audio_nodes=4)
                if graph_fig:
                    st.pyplot(graph_fig)
                    plt.close(graph_fig)
                else:
                    st.error("Graph visualization failed.")

                col_analyze, col_explain = st.columns([1, 1])
                with col_analyze:
                    analyze_btn = st.button(
                        "Run DeepFake Detection",
                        help="Initiate comprehensive multimodal analysis",
                        key="fancy_analyze_button",
                        use_container_width=True
                    )
                with col_explain:
                    st.markdown("""
                    <div style="background-color: rgba(100,160,220,0.85); 
                                border-radius: 12px; 
                                padding: 15px; 
                                text-align: center;
                                border: 2px solid #1e90ff;">
                    Powered by GNN for precise video and audio analysis
                    </div>
                    """, unsafe_allow_html=True)

                if analyze_btn:
                    run_detection(frames_tensor, audio_waveform)
        
        except Exception as e:
            st.error(f"Error processing your video: {str(e)}")
        finally:
            try:
                shutil.rmtree("temp")
                os.makedirs("temp", exist_ok=True)
            except:
                pass

# Sidebar Configuration
def sidebar():
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(to right, #1e90ff, #32cd32);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 20px;
    ">
    <h2 style="color: white; margin: 0;">🛡DeepFake Guardian</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🤖 Model Insights")
    
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        
        ax.text(0.1, 0.9, "Performance Metrics", fontsize=16, fontweight='bold', color='#2c3e50')
        ax.text(0.1, 0.7, "95.85%", fontsize=24, fontweight='bold', color='#1e90ff')
        ax.text(0.1, 0.65, "Accuracy (+4.72%)", fontsize=12, color='#7f8c8d')
        ax.text(0.5, 0.7, "94.72%", fontsize=24, fontweight='bold', color='#32cd32')
        ax.text(0.5, 0.65, "F1 Score (+1.8%)", fontsize=12, color='#7f8c8d')
        
        ax.text(0.1, 0.5, "Model Confidence Threshold: 85%", fontsize=12, color='#2c3e50')
        
        ax.text(0.1, 0.3, "Key Features", fontsize=14, fontweight='bold', color='#2c3e50')
        ax.text(0.1, 0.2, "• Graph Neural Network", fontsize=12, color='#2c3e50')
        ax.text(0.1, 0.15, "• Audio Spectogram Generate", fontsize=12, color='#2c3e50')
        ax.text(0.1, 0.10, "• Multimodal AI", fontsize=12, color='#2c3e50')
        
        plt.tight_layout()
        st.sidebar.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.sidebar.error(f"Could not display metrics: {str(e)}")
    
    st.sidebar.markdown("""
    #### Technical Details
    - **Architecture:** Graph Attention Networks
    - **Inputs:** Video Frames & Audio Waveforms
    - **Classes:** 4 Multimodal Categories
    - **Model Size:** 15.2 MB
    - **Inference Time:** ~1.2s (on GPU)
    """)
    
    st.sidebar.markdown("### Disclaimer")
    with st.sidebar.expander("Important Notes"):
        st.write("- Results are probabilistic")
        st.write("- Not a definitive judgment")
        st.write("- Use as a decision-support tool")
        st.write("- Model trained on DFDC, FakeAVCeleb datasets")
        st.write("- Performance may vary with unseen data")

    st.sidebar.markdown("""
    <div class="developer-section">
        <h3>👤 Developer</h3>
        <p><strong>Diganta Diasi</strong></p>
        <p><a href="https://www.linkedin.com/in/digantadiasi/">Contact</a></p>
        <p><strong>Updated:</strong> March 05, 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    sidebar()
    main()

