import torch
import torch.nn as nn
import torch.nn.functional as F

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        self.eps = eps
        
        mask = (x[:, 0:1, :, :] > -20.0).float()

        means = []
        stds = []
        
        for c in range(x.shape[1]):
            channel_data = x[:, c:c+1, :, :]

            valid_pixels = channel_data[mask.bool()]
            
            means.append(valid_pixels.mean())
            stds.append(valid_pixels.std())

        self.mean = torch.stack(means).view(1, -1, 1, 1)
        self.std = torch.stack(stds).view(1, -1, 1, 1)

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return (x * (self.std + self.eps)) + self.mean
        
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[-2], x.shape[-1]//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        self.fc0 = nn.Linear(3, self.width)

        # Block 0
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2) 
        self.w0 = nn.Conv2d(self.width, self.width, 1)                                

        # Block 1
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        # Block 2
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Block 3
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # --- The Output Projection ---
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)         
        x = x1 + x2              
        x = F.gelu(x)            

        # Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Block 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 3. Project to Output
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) 
        
        x = x.permute(0, 3, 1, 2)
        return x
