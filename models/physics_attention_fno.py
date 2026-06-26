import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes):
        super(AttentionSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes, max_modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes, max_modes, dtype=torch.cfloat))

        # --- THE ATTENTION HEAD ---
        # MLP that looks at the frequency magnitudes and generates a gating mask
        self.attention_mlp = nn.Sequential(
            nn.Linear(max_modes * max_modes, max_modes * max_modes),
            nn.GELU(),
            nn.Linear(max_modes * max_modes, max_modes * max_modes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        x_ft_c1 = x_ft[:, :, :self.max_modes, :self.max_modes]
        amp_c1 = torch.abs(x_ft_c1).view(batchsize, self.in_channels, -1)

        amp_c1_pooled = torch.mean(amp_c1, dim=1) 
        attention_scores_c1 = self.attention_mlp(amp_c1_pooled).view(batchsize, 1, self.max_modes, self.max_modes)

        out_ft[:, :, :self.max_modes, :self.max_modes] = torch.einsum("bixy,ioxy->boxy", x_ft_c1, self.weights1) * attention_scores_c1

        x_ft_c2 = x_ft[:, :, -self.max_modes:, :self.max_modes]
        amp_c2 = torch.abs(x_ft_c2).view(batchsize, self.in_channels, -1)
        amp_c2_pooled = torch.mean(amp_c2, dim=1)
        attention_scores_c2 = self.attention_mlp(amp_c2_pooled).view(batchsize, 1, self.max_modes, self.max_modes)
        
        out_ft[:, :, -self.max_modes:, :self.max_modes] = torch.einsum("bixy,ioxy->boxy", x_ft_c2, self.weights2) * attention_scores_c2

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# Tried adding physics constraints directly into the FNO architecture, but it didn't work well. Instead, we apply hard physics constraints after the model's output.

class PhysicsAttentionFNO(nn.Module):
    def __init__(self, max_modes=32, width=64, num_in_channels=3, num_out_channels=3):
        super(PhysicsAttentionFNO, self).__init__()

        self.p = nn.Linear(num_in_channels, width)

        self.conv0 = AttentionSpectralConv2d(width, width, max_modes)
        self.conv1 = AttentionSpectralConv2d(width, width, max_modes)
        self.conv2 = AttentionSpectralConv2d(width, width, max_modes)
        self.conv3 = AttentionSpectralConv2d(width, width, max_modes)
        
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        self.q = nn.Linear(width, num_out_channels)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)
        
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)
        
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = F.gelu(x3)
        
        x4 = self.conv3(x3) + self.w3(x3)

        x4 = x4.permute(0, 2, 3, 1)
        out = self.q(x4)
        out = out.permute(0, 3, 1, 2)
 
        return out

def apply_hard_physics_constraint(preds_norm, raw_inputs, y_normalizer):

    device = preds_norm.device

    raw_mask = raw_inputs[:, 1:2, :, :].to(device)
    binary_mask = (raw_mask > 0.0).float()
    
    means = y_normalizer.mean.view(-1)
    stds = y_normalizer.std.view(-1)
    
    constrained_channels = []

    for c in range(preds_norm.shape[1]):
        pred_c = preds_norm[:, c:c+1, :, :]

        if c < 2 and c < len(means):
            scaled_zero = -means[c] / stds[c]
            constrained_c = torch.where(binary_mask == 1, pred_c, scaled_zero)
        else:
            constrained_c = pred_c
            
        constrained_channels.append(constrained_c)
        
    return torch.cat(constrained_channels, dim=1)