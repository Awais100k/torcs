import torch
import torch.nn as nn

class DrivingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Head for continuous controls (accel, brake, steer)
        self.control_head = nn.Linear(hidden_dim, 3)
        # Head for gear classification (-1, 0, 1, ..., 6)
        self.gear_head = nn.Linear(hidden_dim, 8)  # 8 classes

    def forward(self, x):
        # Pass through shared backbone
        x = self.backbone(x)
        # Get outputs from each head
        control = self.control_head(x)  # [batch_size, 3]
        gear_logits = self.gear_head(x)  # [batch_size, 8]
        return control, gear_logits