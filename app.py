# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import cv2
# import torchaudio
# import moviepy.editor as mp
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from network.graph_video_audio_model import GAT_video_audio
# from dataset.video_frame_extraction import extract_frames_from_video
# from dataset.audio_extraction import extract_audio_from_video

# # **Set Page Config FIRST**
# st.set_page_config(page_title="Multimodal Deepfake Detector", layout="wide", initial_sidebar_state="expanded")

# # Advanced CSS with gradient background and animations
# st.markdown("""
#     <style>
#     .main {
#         background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#         color: #ffffff;
#     }
#     .stButton>button {
#         background: linear-gradient(90deg, #00c6ff, #0072ff);
#         color: white;
#         border: none;
#         border-radius: 25px;
#         padding: 10px 20px;
#         font-weight: bold;
#         transition: transform 0.3s, box-shadow 0.3s;
#     }
#     .stButton>button:hover {
#         transform: scale(1.05);
#         box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4);
#     }
#     .stProgress .st-bo {
#         background: linear-gradient(90deg, #00ff87, #00c6ff);
#     }
#     .sidebar .sidebar-content {
#         background: rgba(255, 255, 255, 0.95);
#         border-radius: 15px;
#         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#         padding: 20px;
#     }
#     h1, h2 {
#         color: #ffffff;
#         text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
#     }
#     .stAlert {
#         border-radius: 10px;
#         background: rgba(255, 255, 255, 0.1);
#         color: #ffffff;
#     }
#     .frame-container {
#         background: rgba(255, 255, 255, 0.05);
#         padding: 15px;
#         border-radius: 10px;
#         transition: transform 0.3s;
#     }
#     .frame-container:hover {
#         transform: translateY(-5px);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # **Set Device**
# Device = "cuda" if torch.cuda.is_available() else "cpu"

# # **Load Pre-trained Model**
# MODEL_PATH = "summary\weight\model_2025-03-05_17-56-30.pth"
# num_classes = 4

# # Load model
# model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
# model.eval()

# # **Class Names**
# class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# # **Streamlit UI**
# st.title("🎭 Multimodal Deepfake Detection System")
# st.markdown("🔍 Unveil the truth with cutting-edge AI. Analyze videos for deepfake authenticity in real-time!")

# # **Video Upload Section**
# with st.container():
#     st.subheader("📤 Upload Your Video")
#     uploaded_file = st.file_uploader(
#         "Drop a video here (.mp4)", 
#         type=["mp4"], 
#         help="Supports MP4 format up to 200MB",
#         key="uploader"
#     )

# if uploaded_file:
#     # **Save Uploaded File**
#     video_path = f"temp/{uploaded_file.name}"
#     os.makedirs("temp", exist_ok=True)
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # **Display Video**
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.video(video_path)
#     with col2:
#         st.success("✅ Video Uploaded Successfully!")
#         st.metric("File Name", uploaded_file.name)
#         st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")

#     # **Extracted Frames Section**
#     st.subheader("🖼️ Extracted Video Frames")
#     frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
#     cols = st.columns(4)
#     for i, col in enumerate(cols):
#         with col:
#             frame = frames_tensor[i].permute(1, 2, 0).numpy()
#             st.markdown(f"<div class='frame-container'><img src='data:image/png;base64,{_encode_image(frame)}' style='width:100%; border-radius:10px;'><p style='text-align:center;'>Frame {i+1}</p></div>", unsafe_allow_html=True)

#     # **Extracted Audio Section**
#     st.subheader("🎵 Audio Waveform Analysis")
#     audio_waveform = extract_audio_from_video(video_path)
#     if audio_waveform is not None:
#         fig, ax = plt.subplots(figsize=(12, 4), facecolor='none')
#         ax.plot(audio_waveform.numpy(), color="#00ff87", linewidth=2)
#         ax.set_title("Audio Waveform", fontsize=16, color="white", pad=15)
#         ax.set_xlabel("Time", fontsize=12, color="white")
#         ax.set_ylabel("Amplitude", fontsize=12, color="white")
#         ax.set_facecolor('none')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_color('white')
#         ax.spines['bottom'].set_color('white')
#         ax.tick_params(colors='white')
#         plt.tight_layout()
#         st.pyplot(fig)
#     else:
#         st.warning("⚠️ No audio detected in the video!")

#     # **Analyze Button with Animation**
#     if st.button("🔍 Analyze Video", key="analyze", help="Launch deepfake detection"):
#         with st.spinner("🧠 Processing with AI..."):
#             # **Run Model Prediction**
#             with torch.no_grad():
#                 video_input = frames_tensor.unsqueeze(0).to(Device)
#                 audio_input = audio_waveform.unsqueeze(0).unsqueeze(0).to(Device)
#                 outputs, _, _, _ = model(video_input, audio_input)

#             # **Get Prediction**
#             probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
#             predicted_class = np.argmax(probabilities)
#             confidence = probabilities[predicted_class] * 100

#         # **Results Section**
#         st.subheader("📊 Deepfake Detection Results")
#         st.markdown(f"### 🎭 Verdict: **{class_names[predicted_class]}**")
#         st.progress(int(confidence))
#         st.markdown(f"**Confidence Level:** {confidence:.2f}%", unsafe_allow_html=True)

#         # **Advanced Probability Visualization**
#         fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
#         sns.barplot(x=class_names, y=probabilities, ax=ax, palette="magma")
#         ax.set_xlabel("Class", fontsize=12, color="white")
#         ax.set_ylabel("Probability", fontsize=12, color="white")
#         ax.set_title("Prediction Confidence", fontsize=16, color="white", pad=15)
#         ax.set_facecolor('none')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_color('white')
#         ax.spines['bottom'].set_color('white')
#         ax.tick_params(colors='white')
#         plt.xticks(rotation=15, color="white")
#         plt.tight_layout()
#         st.pyplot(fig)

#         # **Download Results with Animation**
#         results_df = pd.DataFrame({"Class": class_names, "Probability": probabilities})
#         csv_file = results_df.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="📥 Download Report",
#             data=csv_file,
#             file_name="deepfake_analysis_results.csv",
#             mime="text/csv",
#             help="Save the analysis as a CSV file",
#             key="download"
#         )

# # **Sidebar - Enhanced About Section**
# with st.sidebar:
#     st.header("ℹ️ System Overview")
#     st.markdown("""
#         <div>
#         Powered by a **multimodal Graph Attention Network (GAT)**, this system detects deepfakes by analyzing video frames and audio signals with unparalleled accuracy.
#         </div>
#     """, unsafe_allow_html=True)
#     st.info("**Capabilities:**\n- Frame-by-frame video analysis\n- Audio waveform processing\n- Real-time deepfake detection")
#     st.markdown("**Creator:** Diganta Diasi")
#     st.markdown("📧 **Email:** [digantadiasi7@gmail.com](mailto:digantadiasi7@gmail.com)")
#     st.markdown("📅 **Updated:** March 05, 2025")
#     st.image("https://img.icons8.com/fluency/48/000000/artificial-intelligence.png", caption="AI-Powered")

# # **Footer**
# st.markdown("---")
# st.markdown("<p style='text-align: center; color: rgba(255, 255, 255, 0.7);'>Built with ❤️ by Diganta Diasi</p>", unsafe_allow_html=True)

# # Helper function to encode images for custom display
# def _encode_image(image):
#     from io import BytesIO
#     import base64
#     img = Image.fromarray((image * 255).astype(np.uint8))
#     buffered = BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")



# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import cv2
# import torchaudio
# import moviepy.editor as mp
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from network.graph_video_audio_model import GAT_video_audio
# from dataset.video_frame_extraction import extract_frames_from_video
# from dataset.audio_extraction import extract_audio_from_video

# # **Set Device**
# Device = "cuda" if torch.cuda.is_available() else "cpu"

# # **Load Pre-trained Model**
# MODEL_PATH = "summary\weight\model_2025-03-05_17-56-30.pth"  # Change if needed
# num_classes = 4

# # Load model
# model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
# model.eval()

# # **Class Names**
# class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# # **Streamlit UI Configuration**
# st.set_page_config(
#     page_title="DeepFake Guardian",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Advanced Custom CSS
# st.markdown("""
# <style>
#     :root {
#         --primary-color: #3498db;
#         --secondary-color: #2ecc71;
#         --background-color: #f4f6f7;
#         --text-color: #2c3e50;
#     }
    
#     .reportview-container {
#         background: linear-gradient(135deg, var(--background-color) 0%, #e0e6ed 100%);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(to right, rgba(255,255,255,0.9), rgba(241,245,249,0.9));
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         border-radius: 12px;
#     }
    
#     .stButton>button {
#         background-color: var(--primary-color);
#         color: white;
#         border-radius: 20px;
#         border: none;
#         padding: 10px 24px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stButton>button:hover {
#         background-color: #2980b9;
#         transform: translateY(-3px);
#         box-shadow: 0 6px 8px rgba(0,0,0,0.15);
#     }
    
#     .stMetric>div {
#         background-color: white;
#         border-radius: 10px;
#         padding: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stFileUploader>div>div>div>div {
#         background-color: rgba(255, 255, 255, 0.8);
#         border-radius: 15px;
#         padding: 15px;
#         border: 2px dashed var(--primary-color);
#     }
    
#     .stProgressBar>div>div {
#         background-color: var(--secondary-color);
#     }
    
#     h1, h2, h3 {
#         color: var(--text-color);
#         font-weight: 700;
#     }
    
#     .block-container {
#         padding-top: 1rem;
#         padding-bottom: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # **Main App**
# def main():
#     # Title with gradient effect
#     st.markdown("""
#     <h1 style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 20px;
#     ">🛡️ Multimodal Deepfake Detection System</h1>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="text-align: center; color: #7f8c8d; margin-bottom: 20px;">
#     Advanced AI-powered detection of synthetic media using video and audio analysis
#     </div>
#     """, unsafe_allow_html=True)

#     # **Video Upload**
#     uploaded_file = st.file_uploader(
#         "📤 Upload Video", 
#         type=["mp4"], 
#         help="Upload a video file for deepfake analysis (Max 200MB)",
#         accept_multiple_files=False
#     )

#     if uploaded_file:
#         # **Save Uploaded File**
#         video_path = f"temp/{uploaded_file.name}"
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Advanced Video Display
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             st.video(video_path)
        
#         with col2:
#             st.success("✅ Video Uploaded Successfully!")
#             with st.expander("🔍 Analysis Preparation"):
#                 st.write("- Frame Extraction")
#                 st.write("- Audio Waveform Analysis")
#                 st.write("- AI Model Inference")

#         # **Extract Frames**
#         st.subheader("🖼️ Video Frame Analysis")
#         frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
        
#         # Display frames in a carousel-like grid
#         frame_cols = st.columns(4)
#         for i, col in enumerate(frame_cols):
#             frame = frames_tensor[i].permute(1, 2, 0).numpy()
#             with col:
#                 st.image(frame, caption=f"Frame {i+1}", use_container_width=True, 
#                          output_format="PNG")

#         # **Extract Audio**
#         st.subheader("🎵 Audio Waveform")
#         audio_waveform = extract_audio_from_video(video_path)
#         if audio_waveform is not None:
#             fig, ax = plt.subplots(figsize=(10, 3), facecolor='#f4f6f7')
#             ax.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis')
#             ax.set_title("Audio Spectrogram", fontsize=12)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Frequency")
#             st.pyplot(fig)
#         else:
#             st.warning("⚠️ No valid audio extracted!")

#         # **Analyze Button with Advanced UI**
#         col_analyze, col_explain = st.columns([1, 1])
        
#         with col_analyze:
#             analyze_btn = st.button(
#                 "🔍 Run DeepFake Detection", 
#                 help="Comprehensive multimodal analysis",
#                 use_container_width=True
#             )
        
#         with col_explain:
#             st.markdown("""
#             <div style="background-color: rgba(255,255,255,0.7); 
#                         border-radius: 10px; 
#                         padding: 10px; 
#                         text-align: center;
#                         border: 1px solid #3498db;">
#             🤖 AI uses Graph Attention Networks to analyze video and audio
#             </div>
#             """, unsafe_allow_html=True)

#         # **Run Analysis**
#         if analyze_btn:
#             with st.spinner('Advanced AI Analysis in Progress...'):
#                 # **Run Model Prediction**
#                 with torch.no_grad():
#                     video_input = frames_tensor.unsqueeze(0).to(Device)
#                     audio_input = audio_waveform.unsqueeze(0).unsqueeze(0).to(Device)
#                     outputs, _, _, _ = model(video_input, audio_input)

#                 # **Get Prediction**
#                 probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
#                 predicted_class = np.argmax(probabilities)
#                 confidence = probabilities[predicted_class] * 100

#                 # Results Display with Advanced Layout
#                 st.markdown("## 🎭 Detection Results")
                
#                 # Detailed Result Columns
#                 result_col1, result_col2 = st.columns(2)
                
#                 with result_col1:
#                     st.metric(
#                         label="🏷️ Predicted Category", 
#                         value=class_names[predicted_class], 
#                         delta=f"{confidence:.2f}% Confidence"
#                     )
                    
#                     # Confidence Visualization
#                     st.progress(int(confidence/100 * 100))
                    
#                     # Risk Assessment
#                     risk_levels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
#                     risk_index = int(confidence / 25)
#                     st.error(f"🚨 Risk Level: {risk_levels[risk_index]}")

#                 with result_col2:
#                     # Probability Bar Chart with Enhanced Styling
#                     fig, ax = plt.subplots(figsize=(6, 3), facecolor='#f4f6f7')
#                     sns.barplot(x=class_names, y=probabilities, ax=ax, palette="coolwarm")
#                     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#                     ax.set_title("Probability Distribution", fontsize=12)
#                     ax.set_ylabel("Probability")
#                     st.pyplot(fig)

#                 # **Download Results**
#                 results_df = pd.DataFrame({
#                     "Class": class_names,
#                     "Probability": probabilities
#                 })
#                 csv_file = results_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "📥 Download Detailed Report", 
#                     csv_file, 
#                     "deepfake_detection_report.csv", 
#                     "text/csv",
#                     help="Comprehensive analysis report",
#                     use_container_width=True
#                 )

# # **Sidebar Configuration**
# def sidebar():
#     # Fancy Sidebar Header
#     st.sidebar.markdown("""
#     <div style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         color: white;
#         padding: 15px;
#         text-align: center;
#         border-radius: 10px;
#         margin-bottom: 20px;
#     ">
#     <h2 style="color: white; margin: 0;">🛡️ DeepFake Guardian</h2>
#     </div>
#     """, unsafe_allow_html=True)

#     # Model Information
#     st.sidebar.markdown("### 🤖 Model Insights")
#     cols = st.sidebar.columns(3)
#     with cols[1]:
#         st.image("dataset\image.png", use_container_width=True)
    
#     st.sidebar.markdown("""
#     #### Technical Details
#     - **Architecture:** Graph Attention Networks
#     - **Inputs:** Video & Audio
#     - **Classes:** 4 Multimodal
#     """)
    
#     # Performance Metrics
#     st.sidebar.markdown("### 📊 Performance")
#     st.sidebar.progress(85)
#     st.sidebar.caption("85% Accuracy on Test Dataset")
    
#     # Developer & Contact
#     st.sidebar.markdown("### 👤 Developer Details")
#     st.sidebar.markdown("**Diganta Diasi**")
#     st.sidebar.markdown("📧 [Contact](mailto:digantadiasi7@gmail.com)")
    
#     # Disclaimers
#     st.sidebar.markdown("### ⚠️ Disclaimer")
#     with st.sidebar.expander("Important Notes"):
#         st.write("- AI detection is probabilistic")
#         st.write("- Not 100% foolproof")
#         st.write("- Use as a guidance tool")

# # Run the app
# if __name__ == "__main__":
#     sidebar()
#     main()


# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import cv2
# import torchaudio
# import moviepy.editor as mp
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from network.graph_video_audio_model import GAT_video_audio
# from dataset.video_frame_extraction import extract_frames_from_video
# from dataset.audio_extraction import extract_audio_from_video

# # **Set Device**
# Device = "cuda" if torch.cuda.is_available() else "cpu"

# # **Load Pre-trained Model**
# MODEL_PATH = "summary/weight/model_2025-03-05_17-56-30.pth"  # Updated path separator
# num_classes = 4

# # Load model
# model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
# model.eval()

# # **Class Names**
# class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# # **Streamlit UI Configuration**
# st.set_page_config(
#     page_title="DeepFake Guardian",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Advanced Custom CSS
# st.markdown("""
# <style>
#     :root {
#         --primary-color: #3498db;
#         --secondary-color: #2ecc71;
#         --background-color: #f4f6f7;
#         --text-color: #2c3e50;
#     }
    
#     .main {
#         background: linear-gradient(135deg, var(--background-color) 0%, #e0e6ed 100%);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(to right, rgba(255,255,255,0.9), rgba(241,245,249,0.9));
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         border-radius: 12px;
#     }
    
#     .stButton>button {
#         background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
#         color: white;
#         border-radius: 20px;
#         border: none;
#         padding: 12px 28px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 6px 12px rgba(0,0,0,0.15);
#     }
    
#     .stMetric {
#         background-color: white;
#         border-radius: 10px;
#         padding: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stFileUploader {
#         background-color: RGBA(255, 255, 255, 0.8);
#         border-radius: 15px;
#         padding: 20px;
#         border: 3px dashed var(--primary-color);
#         transition: border-color 0.3s ease;
#     }
    
#     .stFileUploader:hover {
#         border-color: var(--secondary-color);
#     }
    
#     .stProgress > div > div {
#         background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
#     }
    
#     h1, h2, h3 {
#         color: var(--text-color);
#         font-weight: 700;
#     }
    
#     .block-container {
#         padding: 2rem 1rem;
#     }
    
#     .frame-box {
#         background: white;
#         border-radius: 10px;
#         padding: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         transition: transform 0.3s ease;
#     }
    
#     .frame-box:hover {
#         transform: scale(1.05);
#     }
# </style>
# """, unsafe_allow_html=True)

# # **Main App**
# def main():
#     # Title with gradient effect
#     st.markdown("""
#     <h1 style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 20px;
#     ">🛡️ Multimodal Deepfake Detection System</h1>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
#     Harness the power of AI to detect synthetic media with precision
#     </div>
#     """, unsafe_allow_html=True)

#     # **Video Upload**
#     uploaded_file = st.file_uploader(
#         "📤 Upload Video", 
#         type=["mp4"], 
#         help="Upload an MP4 video for deepfake analysis (Max 200MB)",
#         accept_multiple_files=False
#     )

#     if uploaded_file:
#         # **Ensure Temp Directory Exists**
#         os.makedirs("temp", exist_ok=True)
        
#         # **Save Uploaded File**
#         video_path = f"temp/{uploaded_file.name}"
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Advanced Video Display
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             st.video(video_path)
        
#         with col2:
#             st.success("✅ Video Uploaded Successfully!")
#             with st.expander("🔍 Analysis Overview", expanded=True):
#                 st.write("• Frame Extraction: 4 key frames")
#                 st.write("• Audio Analysis: Waveform & Spectrogram")
#                 st.write("• AI Model: Graph Attention Network")

#         # **Extract Frames**
#         st.subheader("🖼️ Video Frame Analysis")
#         frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
        
#         # Display frames with hover effect
#         frame_cols = st.columns(4)
#         for i, col in enumerate(frame_cols):
#             frame = frames_tensor[i].permute(1, 2, 0).numpy()
#             with col:
#                 st.markdown(f"<div class='frame-box'>", unsafe_allow_html=True)
#                 st.image(frame, caption=f"Frame {i+1}", use_container_width=True, output_format="PNG")
#                 st.markdown("</div>", unsafe_allow_html=True)

#         # **Extract Audio**
#         st.subheader("🎵 Audio Analysis")
#         audio_waveform = extract_audio_from_video(video_path)
#         if audio_waveform is not None:
#             # Waveform
#             fig_wave, ax_wave = plt.subplots(figsize=(10, 2), facecolor='#f4f6f7')
#             ax_wave.plot(audio_waveform.numpy(), color='#3498db', linewidth=1.5)
#             ax_wave.set_title("Audio Waveform", fontsize=12)
#             ax_wave.set_xlabel("Time")
#             ax_wave.set_ylabel("Amplitude")
#             st.pyplot(fig_wave)

#             # Spectrogram
#             fig_spec, ax_spec = plt.subplots(figsize=(10, 3), facecolor='#f4f6f7')
#             ax_spec.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis')
#             ax_spec.set_title("Audio Spectrogram", fontsize=12)
#             ax_spec.set_xlabel("Time")
#             ax_spec.set_ylabel("Frequency")
#             st.pyplot(fig_spec)
#         else:
#             st.warning("⚠️ No valid audio detected!")

#         # **Analyze Button with Advanced UI**
#         col_analyze, col_explain = st.columns([1, 1])
        
#         with col_analyze:
#             analyze_btn = st.button(
#                 "🔍 Run DeepFake Detection", 
#                 help="Initiate comprehensive multimodal analysis",
#                 use_container_width=True
#             )
        
#         with col_explain:
#             st.markdown("""
#             <div style="background-color: rgba(255,255,255,0.7); 
#                         border-radius: 10px; 
#                         padding: 15px; 
#                         text-align: center;
#                         border: 2px solid #3498db;">
#             🤖 Powered by Graph Attention Networks for precise video & audio analysis
#             </div>
#             """, unsafe_allow_html=True)

#         # **Run Analysis**
#         if analyze_btn:
#             with st.spinner('🚀 Running Advanced AI Analysis...'):
#                 # **Run Model Prediction**
#                 with torch.no_grad():
#                     video_input = frames_tensor.unsqueeze(0).to(Device)
#                     audio_input = audio_waveform.unsqueeze(0).unsqueeze(0).to(Device)
#                     outputs, _, _, _ = model(video_input, audio_input)

#                 # **Get Prediction**
#                 probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
#                 predicted_class = np.argmax(probabilities)
#                 confidence = probabilities[predicted_class] * 100

#                 # Results Display with Advanced Layout
#                 st.markdown("<h2 style='text-align: center;'>🎭 Detection Results</h2>", unsafe_allow_html=True)
                
#                 # Detailed Result Columns
#                 result_col1, result_col2 = st.columns(2)
                
#                 with result_col1:
#                     st.metric(
#                         label="🏷️ Predicted Category", 
#                         value=class_names[predicted_class], 
#                         delta=f"{confidence:.2f}% Confidence",
#                         delta_color="normal"
#                     )
                    
#                     # Confidence Visualization
#                     st.progress(int(confidence))
                    
#                     # Risk Assessment
#                     risk_levels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
#                     risk_index = min(int(confidence / 25), 3)  # Cap at 3 to avoid index error
#                     risk_color = "#2ecc71" if risk_index <= 1 else "#e74c3c"
#                     st.markdown(f"<p style='color: {risk_color}; font-weight: bold;'>🚨 Risk Level: {risk_levels[risk_index]}</p>", unsafe_allow_html=True)

#                 with result_col2:
#                     # Probability Bar Chart with Enhanced Styling
#                     fig, ax = plt.subplots(figsize=(6, 4), facecolor='#f4f6f7')
#                     sns.barplot(x=class_names, y=probabilities, ax=ax, palette="viridis")
#                     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#                     ax.set_title("Probability Distribution", fontsize=12)
#                     ax.set_ylabel("Probability")
#                     plt.tight_layout()
#                     st.pyplot(fig)

#                 # **Download Results**
#                 results_df = pd.DataFrame({
#                     "Class": class_names,
#                     "Probability": [f"{p:.4f}" for p in probabilities]
#                 })
#                 csv_file = results_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "📥 Download Detailed Report", 
#                     csv_file, 
#                     "deepfake_detection_report.csv", 
#                     "text/csv",
#                     help="Download the full analysis report",
#                     use_container_width=True
#                 )

# # **Sidebar Configuration**
# def sidebar():
#     # Fancy Sidebar Header
#     st.sidebar.markdown("""
#     <div style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         color: white;
#         padding: 20px;
#         text-align: center;
#         border-radius: 12px;
#         margin-bottom: 20px;
#     ">
#     <h2 style="color: white; margin: 0;">🛡️ DeepFake Guardian</h2>
#     </div>
#     """, unsafe_allow_html=True)

#     # Model Information
#     st.sidebar.markdown("### 🤖 Model Insights")
#     try:
#         st.sidebar.image("dataset/image.png", use_container_width=True)
#     except FileNotFoundError:
#         st.sidebar.warning("Image not found. Please add 'dataset/image.png'.")

#     st.sidebar.markdown("""
#     #### Technical Details
#     - **Architecture:** Graph Attention Networks
#     - **Inputs:** Video Frames & Audio Waveforms
#     - **Classes:** 4 Multimodal Categories
#     """)
    
#     # Performance Metrics
#     st.sidebar.markdown("### 📊 Performance")
#     st.sidebar.progress(85)
#     st.sidebar.caption("85% Accuracy on Test Dataset")
    
#     # Developer & Contact
#     st.sidebar.markdown("### 👤 Developer Details")
#     st.sidebar.markdown("**Diganta Diasi**")
#     st.sidebar.markdown("📧 [Contact](mailto:digantadiasi7@gmail.com)")
#     st.sidebar.markdown("📅 **Updated:** March 05, 2025")
    
#     # Disclaimers
#     st.sidebar.markdown("### ⚠️ Disclaimer")
#     with st.sidebar.expander("Important Notes"):
#         st.write("- Results are probabilistic")
#         st.write("- Not a definitive judgment")
#         st.write("- Use as a decision-support tool")

# # Run the app
# if __name__ == "__main__":
#     sidebar()
#     main()


# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import cv2
# import torchaudio
# import moviepy.editor as mp
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from network.graph_video_audio_model import GAT_video_audio
# from dataset.video_frame_extraction import extract_frames_from_video
# from dataset.audio_extraction import extract_audio_from_video

# # **Set Device**
# Device = "cuda" if torch.cuda.is_available() else "cpu"

# # **Load Pre-trained Model**
# MODEL_PATH = "summary/weight/model_2025-03-05_17-56-30.pth"
# num_classes = 4

# # Load model
# model = GAT_video_audio(num_classes=num_classes, audio_nodes=4).to(Device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
# model.eval()

# # **Class Names**
# class_names = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

# # **Streamlit UI Configuration**
# st.set_page_config(
#     page_title="DeepFake Guardian",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Advanced Custom CSS
# st.markdown("""
# <style>
#     :root {
#         --primary-color: #3498db;
#         --secondary-color: #2ecc71;
#         --background-color: #f4f6f7;
#         --text-color: #2c3e50;
#     }
    
#     .main {
#         background: linear-gradient(135deg, var(--background-color) 0%, #e0e6ed 100%);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(to right, rgba(255,255,255,0.9), rgba(241,245,249,0.9));
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         border-radius: 12px;
#     }
    
#     .stButton>button {
#         background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
#         color: white;
#         border-radius: 20px;
#         border: none;
#         padding: 12px 28px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 6px 12px rgba(0,0,0,0.15);
#     }
    
#     .stMetric {
#         background-color: white;
#         border-radius: 10px;
#         padding: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stFileUploader {
#         background-color: RGBA(255, 255, 255, 0.8);
#         border-radius: 15px;
#         padding: 20px;
#         border: 3px dashed var(--primary-color);
#         transition: border-color 0.3s ease;
#     }
    
#     .stFileUploader:hover {
#         border-color: var(--secondary-color);
#     }
    
#     .stProgress > div > div {
#         background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
#     }
    
#     h1, h2, h3 {
#         color: var(--text-color);
#         font-weight: 700;
#     }
    
#     .block-container {
#         padding: 2rem 1rem;
#     }
    
#     .frame-box {
#         background: white;
#         border-radius: 10px;
#         padding: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         transition: transform 0.3s ease;
#     }
    
#     .frame-box:hover {
#         transform: scale(1.05);
#     }
# </style>
# """, unsafe_allow_html=True)

# # **Main App**
# def main():
#     # Title with gradient effect
#     st.markdown("""
#     <h1 style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 20px;
#     ">🛡️ Multimodal Deepfake Detection System</h1>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
#     Harness the power of AI to detect synthetic media with precision
#     </div>
#     """, unsafe_allow_html=True)

#     # **Video Upload**
#     uploaded_file = st.file_uploader(
#         "📤 Upload Video", 
#         type=["mp4"], 
#         help="Upload an MP4 video for deepfake analysis (Max 200MB)",
#         accept_multiple_files=False
#     )

#     if uploaded_file:
#         # **Ensure Temp Directory Exists**
#         os.makedirs("temp", exist_ok=True)
        
#         # **Save Uploaded File**
#         video_path = f"temp/{uploaded_file.name}"
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Advanced Video Display
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             st.video(video_path)
        
#         with col2:
#             st.success("✅ Video Uploaded Successfully!")
#             with st.expander("🔍 Analysis Overview", expanded=True):
#                 st.write("• Frame Extraction: 4 key frames")
#                 st.write("• Audio Analysis: Waveform & Spectrogram")
#                 st.write("• AI Model: Graph Attention Network")

#         # **Extract Frames**
#         st.subheader("🖼️ Video Frame Analysis")
#         frames_tensor = extract_frames_from_video(video_path, num_frames=4, image_size=128)
        
#         # Display frames with hover effect
#         frame_cols = st.columns(4)
#         for i, col in enumerate(frame_cols):
#             frame = frames_tensor[i].permute(1, 2, 0).numpy()
#             with col:
#                 st.markdown(f"<div class='frame-box'>", unsafe_allow_html=True)
#                 st.image(frame, caption=f"Frame {i+1}", use_container_width=True, output_format="PNG")
#                 st.markdown("</div>", unsafe_allow_html=True)

#         # **Extract Audio**
#         st.subheader("🎵 Audio Analysis")
#         audio_waveform = extract_audio_from_video(video_path)
#         if audio_waveform is not None:
#             # Waveform
#             fig_wave, ax_wave = plt.subplots(figsize=(10, 2), facecolor='#f4f6f7')
#             ax_wave.plot(audio_waveform.numpy(), color='#3498db', linewidth=1.5)
#             ax_wave.set_title("Audio Waveform", fontsize=12)
#             ax_wave.set_xlabel("Time")
#             ax_wave.set_ylabel("Amplitude")
#             st.pyplot(fig_wave)

#             # Spectrogram
#             fig_spec, ax_spec = plt.subplots(figsize=(10, 3), facecolor='#f4f6f7')
#             ax_spec.specgram(audio_waveform.numpy(), Fs=16000, cmap='viridis')
#             ax_spec.set_title("Audio Spectrogram", fontsize=12)
#             ax_spec.set_xlabel("Time")
#             ax_spec.set_ylabel("Frequency")
#             st.pyplot(fig_spec)
#         else:
#             st.warning("⚠️ No valid audio detected!")

#         # **Analyze Button with Advanced UI**
#         col_analyze, col_explain = st.columns([1, 1])
        
#         with col_analyze:
#             analyze_btn = st.button(
#                 "🔍 Run DeepFake Detection", 
#                 help="Initiate comprehensive multimodal analysis",
#                 use_container_width=True
#             )
        
#         with col_explain:
#             st.markdown("""
#             <div style="background-color: rgba(255,255,255,0.7); 
#                         border-radius: 10px; 
#                         padding: 15px; 
#                         text-align: center;
#                         border: 2px solid #3498db;">
#             🤖 Powered by Graph Attention Networks for precise video & audio analysis
#             </div>
#             """, unsafe_allow_html=True)

#         # **Run Analysis**
#         if analyze_btn:
#             with st.spinner('🚀 Running Advanced AI Analysis...'):
#                 # **Run Model Prediction**
#                 with torch.no_grad():
#                     video_input = frames_tensor.unsqueeze(0).to(Device)
#                     # Adjust audio input to match expected 2D shape (batch_size, audio_samples)
#                     audio_input = audio_waveform.unsqueeze(0).to(Device)  # Shape: (1, audio_samples)
#                     outputs, _, _, _ = model(video_input, audio_input)

#                 # **Get Prediction**
#                 probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
#                 predicted_class = np.argmax(probabilities)
#                 confidence = probabilities[predicted_class] * 100

#                 # Results Display with Advanced Layout
#                 st.markdown("<h2 style='text-align: center;'>🎭 Detection Results</h2>", unsafe_allow_html=True)
                
#                 # Detailed Result Columns
#                 result_col1, result_col2 = st.columns(2)
                
#                 with result_col1:
#                     st.metric(
#                         label="🏷️ Predicted Category", 
#                         value=class_names[predicted_class], 
#                         delta=f"{confidence:.2f}% Confidence",
#                         delta_color="normal"
#                     )
                    
#                     # Confidence Visualization
#                     st.progress(int(confidence))
                    
#                     # Risk Assessment
#                     risk_levels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
#                     risk_index = min(int(confidence / 25), 3)  # Cap at 3 to avoid index error
#                     risk_color = "#2ecc71" if risk_index <= 1 else "#e74c3c"
#                     st.markdown(f"<p style='color: {risk_color}; font-weight: bold;'>🚨 Risk Level: {risk_levels[risk_index]}</p>", unsafe_allow_html=True)

#                 with result_col2:
#                     # Probability Bar Chart with Enhanced Styling
#                     fig, ax = plt.subplots(figsize=(6, 4), facecolor='#f4f6f7')
#                     sns.barplot(x=class_names, y=probabilities, ax=ax, palette="viridis")
#                     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#                     ax.set_title("Probability Distribution", fontsize=12)
#                     ax.set_ylabel("Probability")
#                     plt.tight_layout()
#                     st.pyplot(fig)

#                 # **Download Results**
#                 results_df = pd.DataFrame({
#                     "Class": class_names,
#                     "Probability": [f"{p:.4f}" for p in probabilities]
#                 })
#                 csv_file = results_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "📥 Download Detailed Report", 
#                     csv_file, 
#                     "deepfake_detection_report.csv", 
#                     "text/csv",
#                     help="Download the full analysis report",
#                     use_container_width=True
#                 )

# # **Sidebar Configuration**
# def sidebar():
#     # Fancy Sidebar Header
#     st.sidebar.markdown("""
#     <div style="
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         color: white;
#         padding: 20px;
#         text-align: center;
#         border-radius: 12px;
#         margin-bottom: 20px;
#     ">
#     <h2 style="color: white; margin: 0;">🛡️ DeepFake Guardian</h2>
#     </div>
#     """, unsafe_allow_html=True)

#     # Model Information
#     st.sidebar.markdown("### 🤖 Model Insights")
#     try:
#         st.sidebar.image("dataset/image.png", use_container_width=True)
#     except FileNotFoundError:
#         st.sidebar.warning("Image not found. Please add 'dataset/image.png'.")

#     st.sidebar.markdown("""
#     #### Technical Details
#     - **Architecture:** Graph Attention Networks
#     - **Inputs:** Video Frames & Audio Waveforms
#     - **Classes:** 4 Multimodal Categories
#     """)
    
#     # Performance Metrics
#     st.sidebar.markdown("### 📊 Performance")
#     st.sidebar.progress(85)
#     st.sidebar.caption("85% Accuracy on Test Dataset")
    
#     # Developer & Contact
#     st.sidebar.markdown("### 👤 Developer Details")
#     st.sidebar.markdown("**Diganta Diasi**")
#     st.sidebar.markdown("📧 [Contact](mailto:digantadiasi7@gmail.com)")
#     st.sidebar.markdown("📅 **Updated:** March 05, 2025")
    
#     # Disclaimers
#     st.sidebar.markdown("### ⚠️ Disclaimer")
#     with st.sidebar.expander("Important Notes"):
#         st.write("- Results are probabilistic")
#         st.write("- Not a definitive judgment")
#         st.write("- Use as a decision-support tool")

# # Run the app
# if __name__ == "__main__":
#     sidebar()
#     main()


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
MODEL_PATH = "summary/weight/model_2025-03-05_17-56-30.pth"
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
