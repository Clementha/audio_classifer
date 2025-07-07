import torch
import torch.nn as nn

class EnvSoundTransformer(nn.Module):
    def __init__(self, num_classes=10, d_model=128, nhead=4, num_layers=2):
        super(EnvSoundTransformer, self).__init__()
        # Conv1d to map (freq) → d_model
        self.conv1d = nn.Conv1d(
            in_channels=128,   # input n_mels
            out_channels=d_model,
            kernel_size=1
        )

        # Transformer encoder with batch_first
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True   # ✅ Use (batch, time, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final classifier
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, freq, time) → from your Dataset
        x = self.conv1d(x)  # (batch, d_model, time)

        # Permute to (batch, time, d_model) for batch_first Transformer
        x = x.permute(0, 2, 1)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Mean pool over time dimension
        x = x.mean(dim=1)  # (batch, d_model)

        return self.fc(x)