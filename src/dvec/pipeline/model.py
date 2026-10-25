"""Causal Target Speaker Extraction (TSE) Neural Network Architecture.

Features:
- Causal 2D Complex Convolutions (no future lookahead)
- FiLM (Feature-wise Linear Modulation) conditioning with target speaker embedding
- Stateful Recurrent Bottleneck (LSTM with persistent hidden/cell states)
- Complex Ideal Ratio Mask (cIRM) prediction with spectral floor clamping
- Auxiliary Personal Voice Activity Detection (pVAD) head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """2D Convolution with strictly causal padding along the time dimension."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 0),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.freq_pad = padding[0]
        self.time_pad = kernel_size[1] - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(self.freq_pad, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_pad > 0:
            x = F.pad(x, (self.time_pad, 0, 0, 0))
        return self.conv(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer.
    
    Transforms feature maps based on speaker embedding:
    FiLM(x) = gamma(e) * x + beta(e)
    """

    def __init__(self, embed_dim: int = 256, num_channels: int = 64):
        super().__init__()
        self.gamma_proj = nn.Linear(embed_dim, num_channels)
        self.beta_proj = nn.Linear(embed_dim, num_channels)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(embedding).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class CausalTSEModel(nn.Module):
    """Causal Complex Convolutional Recurrent Network for Target Speaker Extraction."""

    def __init__(
        self,
        num_freq_bins: int = 257,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        spectral_floor: float = 0.02,
    ):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.spectral_floor = spectral_floor

        # Encoder: 2 channels (Real, Imag) -> 32 -> 64
        self.enc_conv1 = CausalConv2d(2, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn2 = nn.BatchNorm2d(64)

        # FiLM Conditioning at bottleneck
        self.film = FiLM(embed_dim=embed_dim, num_channels=64)

        # Bottleneck frequency dimension after stride 2: ceil(257 / 2) = 129
        self.bottleneck_freq = (num_freq_bins + 1) // 2
        
        # Bottleneck compression: 64 ch -> 16 ch, 129 freq -> 33 freq
        self.bot_down = nn.Conv2d(64, 16, kernel_size=(3, 1), stride=(4, 1), padding=(1, 0))
        self.bot_bn_down = nn.BatchNorm2d(16)
        self.compressed_freq = (self.bottleneck_freq + 2 * 1 - 3) // 4 + 1  # 33
        lstm_input_dim = 16 * self.compressed_freq  # 528

        # Recurrent core
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Projection back to feature space
        self.lstm_proj = nn.Linear(hidden_dim, lstm_input_dim)

        # Bottleneck expansion: 16 ch -> 64 ch, 33 freq -> 129 freq
        self.bot_up = nn.ConvTranspose2d(16, 64, kernel_size=(3, 1), stride=(4, 1), padding=(1, 0), output_padding=(0, 0))
        self.bot_bn_up = nn.BatchNorm2d(64)

        # Decoder for cIRM: 64 -> 32 -> 2 (Mask Real, Mask Imag)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(0, 0))
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_conv2 = nn.Conv2d(32, 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        # Personal VAD classification head
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward_sequence(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        embedding: torch.Tensor,
    ):
        """Full sequence forward pass for parallel GPU training.
        
        Args:
            x_real: (B, 1, F, T)
            x_imag: (B, 1, F, T)
            embedding: (B, embed_dim)
        Returns:
            mask_real: (B, 1, F, T)
            mask_imag: (B, 1, F, T)
            vad_probs: (B, T)
        """
        x = torch.cat([x_real, x_imag], dim=1)  # (B, 2, F, T)

        # Encoder
        feat = F.elu(self.enc_bn1(self.enc_conv1(x)))
        feat = F.elu(self.enc_bn2(self.enc_conv2(feat)))  # (B, 64, bottleneck_freq, T)

        # FiLM Conditioning
        feat = self.film(feat, embedding)

        # Compress bottleneck: (B, 64, bottleneck_freq, T) -> (B, 16, compressed_freq, T)
        bot = F.elu(self.bot_bn_down(self.bot_down(feat)))

        # Flatten for LSTM: (B, T, 16 * compressed_freq)
        b, c, f, t = bot.shape
        lstm_in = bot.permute(0, 3, 1, 2).reshape(b, t, c * f)

        # Vectorized recurrent sequence pass
        lstm_out, _ = self.lstm(lstm_in)

        # Frame-wise Personal VAD probabilities: (B, T)
        vad_probs = self.vad_head(lstm_out).squeeze(-1)

        # Projection and expand back to (B, 64, bottleneck_freq, T)
        proj = self.lstm_proj(lstm_out)
        proj = proj.reshape(b, t, c, f).permute(0, 2, 3, 1)
        proj = F.elu(self.bot_bn_up(self.bot_up(proj)))

        # Decoder
        d1 = F.elu(self.dec_bn1(self.dec_conv1(proj)))
        if d1.shape[2] != self.num_freq_bins:
            d1 = F.interpolate(d1, size=(self.num_freq_bins, t), mode="nearest")

        mask = self.dec_conv2(d1)

        # Bound masks with tanh: range [-1.2, 1.2]
        mask_real = 1.2 * torch.tanh(mask[:, 0:1, :, :])
        mask_imag = 1.2 * torch.tanh(mask[:, 1:2, :, :])

        # Enforce spectral floor only during inference (training unconstrained down to 0.0)
        if not self.training and self.spectral_floor > 0.0:
            mask_mag = torch.sqrt(mask_real ** 2 + mask_imag ** 2 + 1e-8)
            scale = torch.clamp(mask_mag, min=self.spectral_floor) / mask_mag
            mask_real = mask_real * scale
            mask_imag = mask_imag * scale

        return mask_real, mask_imag, vad_probs

    def forward_streaming_frame(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        embedding: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ):
        """Single 10ms frame streaming step for real-time ONNX execution.
        
        Args:
            x_real: (B, 1, F, 1)
            x_imag: (B, 1, F, 1)
            embedding: (B, embed_dim)
            h_prev: (num_lstm_layers, B, hidden_dim)
            c_prev: (num_lstm_layers, B, hidden_dim)
        Returns:
            mask_real, mask_imag, vad_prob, h_next, c_next
        """
        x = torch.cat([x_real, x_imag], dim=1)

        # Encoder
        feat = F.elu(self.enc_bn1(self.enc_conv1(x)))
        feat = F.elu(self.enc_bn2(self.enc_conv2(feat)))

        # FiLM Conditioning
        feat = self.film(feat, embedding)

        # Compress bottleneck: (B, 64, bottleneck_freq, 1) -> (B, 16, compressed_freq, 1)
        bot = F.elu(self.bot_bn_down(self.bot_down(feat)))

        # Flatten for LSTM: (B, 1, 16 * compressed_freq)
        b, c, f, t = bot.shape
        lstm_in = bot.permute(0, 3, 1, 2).reshape(b, t, c * f)

        # Recurrent Step
        lstm_out, (h_next, c_next) = self.lstm(lstm_in, (h_prev, c_prev))

        # Personal VAD probability
        vad_prob = self.vad_head(lstm_out[:, -1, :])

        # Projection and expand back to (B, 64, bottleneck_freq, 1)
        proj = self.lstm_proj(lstm_out)
        proj = proj.reshape(b, t, c, f).permute(0, 2, 3, 1)
        proj = F.elu(self.bot_bn_up(self.bot_up(proj)))

        # Decoder
        d1 = F.elu(self.dec_bn1(self.dec_conv1(proj)))
        if d1.shape[2] != self.num_freq_bins:
            d1 = F.interpolate(d1, size=(self.num_freq_bins, t), mode="nearest")

        mask = self.dec_conv2(d1)

        # Bound masks with tanh: range [-1.2, 1.2]
        mask_real = 1.2 * torch.tanh(mask[:, 0:1, :, :])
        mask_imag = 1.2 * torch.tanh(mask[:, 1:2, :, :])

        # Enforce spectral floor only during inference (training unconstrained down to 0.0)
        if not self.training and self.spectral_floor > 0.0:
            mask_mag = torch.sqrt(mask_real ** 2 + mask_imag ** 2 + 1e-8)
            scale = torch.clamp(mask_mag, min=self.spectral_floor) / mask_mag
            mask_real = mask_real * scale
            mask_imag = mask_imag * scale

        return mask_real, mask_imag, vad_prob, h_next, c_next
