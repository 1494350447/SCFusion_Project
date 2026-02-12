
import torch
import torch.nn as nn
from .basic_layers import ConvBlock, fft_amp_phase, FrequencySelectiveKernel, SpatialChannelAttention


class LearnableSelectiveFilterGenerator(nn.Module):
    """Learnable Selective Filter Generator (LSFG) - Core of FREFormer.

    Implements the dynamic filter generation mechanism described in the paper:
    1. Global semantic capture: E(X) = Σ(X_i,j) / (H*W)
    2. Filter generation: G(E(X)) = Σ(E(X)_i,j / Σ E(X)_i,j)
    3. Dynamic modulation: F(Y_i) = G(MLP(Softmax(Y_i)))
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 8)

        # Global average pooling for semantic capture E(X)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Generator network G(E(X)) with normalization
        self.generator = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Softmax for filter weight normalization
        self.softmax = nn.Softmax(dim=1)

        # MLP for instance-specific feature transformation
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            filter_weights: Dynamic filter weights (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Step 1: Global semantic capture E(X)
        global_context = self.global_pool(x)  # (B, C, 1, 1)

        # Step 2: Generate base filter weights G(E(X))
        base_weights = self.generator(global_context)  # (B, C, 1, 1)

        # Step 3: Instance-aware processing
        # Apply MLP and Softmax to input features
        instance_features = self.mlp(x)  # (B, C, H, W)

        # Normalize across spatial dimensions for each channel
        B, C, H, W = instance_features.shape
        instance_features_flat = instance_features.view(B, C, -1)  # (B, C, H*W)
        normalized_features = self.softmax(instance_features_flat)  # (B, C, H*W)
        normalized_features = normalized_features.view(B, C, H, W)  # (B, C, H, W)

        # Step 4: Combine base weights with instance-specific features
        filter_weights = base_weights * (1.0 + normalized_features)

        return filter_weights


class FREFormerBlock(nn.Module):
    """Enhanced FREFormer block implementing the paper's architecture.

    Combines:
    1. Learnable Selective Filter Generator (LSFG) for dynamic frequency filtering
    2. 2D FFT/IFFT for frequency domain processing
    3. Channel MLP for inter-channel feature mixing
    4. Residual connections with normalization
    """

    def __init__(self, channels):
        super().__init__()

        # Normalization layers
        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])

        # Learnable Selective Filter Generator (LSFG)
        self.lsfg = LearnableSelectiveFilterGenerator(channels)

        # Channel MLP for feature mixing (as described in paper)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            out: Processed features (B, C, H, W)
        """
        identity = x
        B, C, H, W = x.shape

        # Apply first normalization
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Step 1: Transform to frequency domain via 2D FFT
        x_fft = torch.fft.rfft2(x_norm, norm='ortho')  # (B, C, H, W//2+1) complex

        # Step 2: Generate dynamic filter weights using LSFG
        filter_weights = self.lsfg(x_norm)  # (B, C, H, W)

        # Adapt filter weights to match FFT output size
        filter_weights_fft = torch.fft.rfft2(filter_weights, norm='ortho')  # Complex weights

        # Step 3: Apply dynamic filtering in frequency domain (element-wise multiplication)
        x_filtered = x_fft * filter_weights_fft

        # Step 4: Transform back to spatial domain via 2D IFFT
        x_spatial = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')  # (B, C, H, W)

        # Residual connection
        x = identity + x_spatial

        # Step 5: Channel MLP with second normalization
        x_norm2 = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = x_norm2.permute(0, 3, 1, 2)  # (B, C, H, W)

        x = x + self.channel_mlp(x_norm2)

        return x


class DownsamplingLayer(nn.Module):
    """Downsampling layer as described in paper: LN -> Conv2d -> LN

    Equation (1): Y_i = LN(Conv2d(LN(X_i)))
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm([in_channels])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.LayerNorm([out_channels])

    def forward(self, x):
        B, C, H, W = x.shape
        # First normalization
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Convolution
        x = self.conv(x)

        # Second normalization
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x


class FREFormerEncoder(nn.Module):
    """FREFormer Encoder implementing the paper's hierarchical architecture.

    Architecture:
    - Two-stage hierarchical structure
    - Each stage: Downsampling Layer + FREFormer Blocks (×4)
    - Combines local inductive bias (conv) and global receptive field (FFT)
    """

    def __init__(self, in_ch=3, base_ch=32, num_blocks=4):
        super().__init__()

        # Initial stem with convolution for local inductive bias
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        # Stage 0: base_ch channels
        self.down0 = DownsamplingLayer(base_ch, base_ch)
        self.blocks0 = nn.ModuleList([
            FREFormerBlock(base_ch) for _ in range(num_blocks)
        ])

        # Stage 1: base_ch*2 channels
        self.down1 = DownsamplingLayer(base_ch, base_ch * 2)
        self.blocks1 = nn.ModuleList([
            FREFormerBlock(base_ch * 2) for _ in range(num_blocks)
        ])

        # Stage 2: base_ch*4 channels
        self.down2 = DownsamplingLayer(base_ch * 2, base_ch * 4)
        self.blocks2 = nn.ModuleList([
            FREFormerBlock(base_ch * 4) for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            List of multi-scale features [f0, f1, f2]
        """
        # Stem
        x = self.stem(x)  # (B, base_ch, H/2, W/2)

        # Stage 0
        f0 = self.down0(x)  # (B, base_ch, H/4, W/4)
        for block in self.blocks0:
            f0 = block(f0)

        # Stage 1
        f1 = self.down1(f0)  # (B, base_ch*2, H/8, W/8)
        for block in self.blocks1:
            f1 = block(f1)

        # Stage 2
        f2 = self.down2(f1)  # (B, base_ch*4, H/16, W/16)
        for block in self.blocks2:
            f2 = block(f2)

        return [f0, f1, f2]

