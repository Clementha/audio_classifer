import torch.nn as nn

class EnvSoundTransformer(nn.Module):
    def __init__(self, num_classes=10, d_model=128, nhead=4, num_layers=2):
        super(EnvSoundTransformer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, freq, time)
        x = self.conv1d(x)  # (batch, d_model, time)
        x = x.permute(2, 0, 1)  # (time, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # mean over time steps
        x = self.fc(x)
        return x