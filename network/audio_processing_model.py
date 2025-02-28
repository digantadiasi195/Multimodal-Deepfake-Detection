#network/audio_processing_model.py 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import Tensor
import moviepy.config as cfg
cfg.DEFAULT_LOGGER = None
class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, mask=False):
        super(CONV, self).__init__()
        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (got {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = np.linspace(0, self.sample_rate / 2, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsf = self.to_hz(np.linspace(min(fmel), max(fmel), out_channels + 1))

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x, mask=False):
        device = x.device
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hHigh - hLow)

        band_pass_filter = self.band_pass.to(device)

        if mask:  # Randomly mask some filters
            A = int(np.random.uniform(0, 14))
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0

        self.filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
            self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)

        self.selu = nn.SELU(inplace=True)
        self.conv_1 = nn.Conv2d(1, nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1)

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], padding=(0, 1), kernel_size=(1, 3), stride=1)

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.selu(self.bn1(x))
            out = self.conv1(out)
        else:
            out = self.conv_1(x)

        out = self.selu(self.bn2(out))
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return self.mp(out)


class audio_model(nn.Module):
    def __init__(self, num_nodes=4):
        super(audio_model, self).__init__()

        self.conv_time = CONV(out_channels=50, kernel_size=128, in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.encoder1 = nn.Sequential(
            Residual_block([32, 32], first=True),
            Residual_block([32, 32]),
            Residual_block([32, 64]),
            Residual_block([64, 64]),
            Residual_block([64, 64]),
            Residual_block([64, 32])
        )

        self.num_nodes = num_nodes

    def forward(self, x, Freq_aug=False):
        """
        x = (batch_size, audio_samples) where audio_samples should be 64000
        """

        if x.dim() == 3:
            x = x[:, 0, :] 

        batch_size, audio_samples = x.shape

        # Ensure audio length is 64000
        if audio_samples != 64000:
            x = F.pad(x, (0, max(0, 64000 - audio_samples)))
            x = x[:, :64000] 

        # Reshape for processing
        x = x.view(batch_size, 1, 64000)

        # Frequency masking during training
        x = self.conv_time(x, mask=Freq_aug)

        x = x.unsqueeze(dim=1)  # Convert to 2D representation
        x = F.max_pool2d(torch.abs(x), (3, 3))  # Apply pooling

        x = self.first_bn(x)
        x = self.selu(x)

        temp = torch.chunk(x, self.num_nodes, dim=3)
        out = []

        for i in range(self.num_nodes):
            t = self.encoder1(temp[i])  # Pass through residual blocks
            x_max, _ = torch.max(torch.abs(t), dim=3)
            out.append(x_max)

        out = torch.stack(out, dim=1)
        out = out.view(out.size(0), out.size(1), -1)
        return out


if __name__ == '__main__':
    model = audio_model(num_nodes=4)
    test_input = torch.randn(2, 64000)  # Simulating batch_size=2, 64000 samples
    model(test_input)
