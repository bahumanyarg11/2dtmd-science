import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer: Performs convolution in frequency domain.
    This is the core engine of the infinite-resolution digital twin.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Fourier Transform
        # Use rfft2 logic for real-valued inputs
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Upper block
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        # Lower block
        # For rfft2, we need to be careful with indices, but this symmetric block is standard for FNO
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SynthesisFNO(nn.Module):
    """
    The Digital Twin Model.
    Input: (Temp, Pressure, Flow_X, Flow_Y) on a coarse grid.
    Output: High-res Concentration Field of Precursors.
    """
    def __init__(self, modes=12, width=32):
        super(SynthesisFNO, self).__init__()
        self.modes = modes
        self.width = width
        
        # Lift input to higher dimension
        self.fc0 = nn.Linear(4, self.width) # 4 input channels (T, P, Vx, Vy)

        # 4 Fourier Layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Project back to scalar (Concentration)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (Batch, GridX, GridY, Features)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # Channel first for Conv

        # FNO Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # FNO Block 2... (Repeat for depth)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Projection
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
