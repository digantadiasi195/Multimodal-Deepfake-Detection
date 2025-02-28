#network/graph_video_audio_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.audio_processing_model import audio_model
from network.video_processing_model import video_model


class GAT_video_audio(nn.Module):
    def __init__(self, num_classes=4, audio_nodes=4):
        super(GAT_video_audio, self).__init__()

        self.num_classes = num_classes

        self.video_encoder = video_model()
        self.audio_encoder = audio_model()

        self.video_feat_dim = 512  # Video feature size after encoding
        self.audio_feat_dim = 512  # Audio feature size after encoding

        # Projection layers to align video and audio feature dimensions
        self.video_proj = nn.Linear(self.video_feat_dim, self.audio_feat_dim)
        self.audio_proj = nn.Linear(self.audio_feat_dim, self.audio_feat_dim)

        # Graph attention layers
        self.GAT_layer1 = nn.Linear(self.audio_feat_dim, 256)
        self.GAT_layer2 = nn.Linear(256, 128)
        self.GAT_layer3 = nn.Linear(128, 64)

        # Output layers with corrected dimensions
        self.video_pre = nn.Linear(64, num_classes) 
        self.audio_pre = nn.Linear(64, num_classes)
        self.mix_pre = nn.Linear(128, num_classes) 

    def forward(self, vid_inp, aud_inp):

        x = self.video_encoder(vid_inp)
        y = self.audio_encoder(aud_inp)
        y = y.mean(dim=1) 

        x = self.video_proj(x)
        y = self.audio_proj(y)

        x = self.GAT_layer1(x)
        x = self.GAT_layer2(x)
        x = self.GAT_layer3(x)

        y = self.GAT_layer1(y)
        y = self.GAT_layer2(y)
        y = self.GAT_layer3(y)

        video_out = self.video_pre(x)
        audio_out = self.audio_pre(y)
        
        fusion_out = torch.cat([x, y], dim=1)
        mix_out = self.mix_pre(fusion_out)

        return mix_out, video_out, audio_out, fusion_out


if __name__ == "__main__":
    net = GAT_video_audio(num_classes=4, audio_nodes=4)

    video_input = torch.randn(4, 4, 3, 128, 128) 
    audio_input = torch.randn(4, 2, 64000)  

    output = net(video_input, audio_input)
