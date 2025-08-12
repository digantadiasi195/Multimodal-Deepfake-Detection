import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from torchsummary import summary
from audio_processing_model import audio_model
from video_processing_model import video_model



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.att_q = nn.Linear(in_dim, out_dim)
        self.att_k = nn.Linear(in_dim, out_dim)
        self.att_v = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Parameter(torch.zeros(1))


        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, groups=out_dim)

        # batch norm
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.5)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)
        x = self.proj_without_att(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm

        x = self._apply_BN(x, self.bn2)

        # apply activation
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''
        nb_nodes = x.size()[1]
        device = x.device

        q = F.normalize(self.att_q(x), dim=-1)          
        k = F.normalize(self.att_k(x), dim=-1).permute(0, 2, 1)
        att = torch.matmul(q, k)  

        a = torch.arange(0, nb_nodes, 1).to(device)
        b = a.unsqueeze(0).expand(nb_nodes, -1)
        b = b-a.unsqueeze(1) + (torch.eye(nb_nodes).to(device)*0.5)
        return torch.div(att+1, b.unsqueeze(0))

        # return att

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = F.softmax(att_map, dim=-1) 

        return att_map

    def _project(self, x, att_map):
        v = self.att_v(x)  #b, n, c
        v = torch.matmul(att_map, v)

        x = self.proj_without_att(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1).contiguous()

        return x + self.gamma * v

    def _apply_BN(self, x, bn):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class Pool(nn.Module):              # new pooling

    def __init__(self, k: float, in_dim: int, p, permute=False):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim
        self.flag = permute

    def forward(self, h, adj):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        scores = torch.matmul(Z, Z.permute(0, 2, 1))
        scores = torch.sum(scores, dim=-1, keepdim=False)

        new_h, adj = self.top_k_graph(scores, h, self.k, adj)

        return new_h, adj

    def top_k_graph(self, scores, h, k, adj):
        """
        args
        ==== 
        scores: attention-based weights (#bs,#node,1)
        h: graph (#bs,#node,#dim)
        k: ratio of remaining nodes, (float)
        """
        num_nodes = h.shape[1]
        batch_size = h.shape[0]

        # first reflect the weights and then rank them
        # H = h * scores
        # scores = scores.squeeze(-1)
        _, idx = torch.topk(scores, max(1, int(k * num_nodes)), largest=False, dim=1)
        idx = torch.sort(idx, dim=1)[0]
        new_g = []
        new_adj = []
        if self.flag:
            adj = adj.permute(0, 2, 1)

        for i in range(batch_size):
            new_g.append(h[i, idx[i][:int(len(idx[i]))], :])
            new_adj.append(adj[i, idx[i][:int(len(idx[i]))], :])
        new_g = torch.stack(new_g, dim=0)
        new_adj = torch.stack(new_adj, dim=0)

        if self.flag:
            new_adj = new_adj.permute(0, 2, 1)

        return new_g, new_adj


class GAT_video_audio(nn.Module):
    def __init__(self, num_classes=2, audio_nodes=4):
        super(GAT_video_audio, self).__init__()

        self.num_classes =num_classes

        self.video_encoder = video_model()
        self.audio_encoder = audio_model()

        self.init_adj = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],      # audio_node=4
                                      [0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]) 

        
        
        # Graph attention and pooling layer for Spectral GAT
        self.GAT_layer1 = GraphAttentionLayer(512, 256)
        self.pool1 = Pool(0.75, 256, 0.3, permute=False)
        self.audio_GAT_layer1 = GraphAttentionLayer(512, 256)
        self.audio_pool1 = Pool(0.75, 256, 0.3, permute=True)

        # Graph attention and pooling layer for Temporal GAT
        self.GAT_layer2 = GraphAttentionLayer(256, 128)
        self.pool2 = Pool(0.75, 128, 0.3, permute=False)
        self.audio_GAT_layer2 = GraphAttentionLayer(256, 128)
        self.audio_pool2 = Pool(0.75, 128, 0.3, permute=True)

        # Graph attention and pooling layer for Spectro-Temporal RawGAT
        self.GAT_layer3 = GraphAttentionLayer(128, 64)
        self.pool3 = Pool(0.75, 64, 0.3, permute=False)
        self.audio_GAT_layer3 = GraphAttentionLayer(128, 64)
        self.audio_pool3 = Pool(0.75, 64, 0.3, permute=True)

        self.video_pre = nn.Linear(192, 2)
        self.audio_pre = nn.Linear(64, 2)                        #64
        self.mix_pre = nn.Linear(256, self.num_classes)          #256

    def video_processing(self, model, inp):

        inp = inp.view((inp.size()[0], 10, -1,) + inp.size()[2:])

        x = []
        for i in range(10):
            x.append(model(inp[:, i, :, :, :].contiguous()))
        x = torch.stack(x, dim=1)

        return x


    def forward(self, vid_inp, aud_inp):
        adj = self.init_adj.unsqueeze(0).expand(vid_inp.size(0), -1, -1).permute(0, 2, 1).to(vid_inp.device)

        x = self.video_processing(self.video_encoder, vid_inp)
        y = self.audio_encoder(aud_inp)

        # [b, 10, 1024]
        # first layer
        x = self.GAT_layer1(x)  # (#bs,#node(F),feat_dim(C)) --> [#b, 10, 512]
        x, adj = self.pool1(x, adj)      # [#b, 10, 512] --> [#b, 7, 1, 512]
        y = self.audio_GAT_layer1(y)
        y, adj = self.audio_pool1(y, adj)

        # second layer
        x = self.GAT_layer2(x)  # [#b, 7, 512] --> [#b, 7, 256]
        x, adj = self.pool2(x, adj)        # [#b, 7, 256] --> [#b, 5, 1, 256]
        y = self.audio_GAT_layer2(y)
        y, adj = self.audio_pool2(y, adj)

        # third layer
        x = self.GAT_layer3(x)     # [#b, 7, 256] --> [#b, 5, 128]
        x, adj = self.pool3(x, adj)      # [#b, 5, 128] --> [#b, 2, 1, 64]
        y = self.audio_GAT_layer3(y)
        y, adj = self.audio_pool3(y, adj)

        x = x.flatten(1)
        y = y.flatten(1)

        video_out = self.video_pre(x)
        audio_out = self.audio_pre(y)
        fusion_out = torch.cat([x, y], dim=1)
        mix_out = self.mix_pre(fusion_out)

        return mix_out, video_out, audio_out, fusion_out


if __name__ == "__main__":

    net = GAT_video_audio(num_classes=2, audio_nodes=8)
    # print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    y = net(torch.randn(1, 120, 128, 128), torch.randn(1, 64000))

    # print(summary(net, (10, 512)))
