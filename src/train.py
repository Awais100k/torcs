import torch
import torch.nn as nn
import numpy as np
from data_utils import load_data
from model import DrivingMLP
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
num_epochs = 300
lr = 1e-3
control_loss_weight = 1.0
gear_loss_weight = 1.0

# 1) Load data
train_loader, val_loader, input_dim, scaler = load_data(batch_size=batch_size)
print(f"Input dimension: {input_dim}")
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")

# 2) Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrivingMLP(input_dim=input_dim).to(device)
control_criterion = nn.MSELoss()
gear_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 3) Training loop with loss tracking
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    # --- Train ---
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # Split yb into control (accel, brake, steer) and gear
        yb_control = yb[:, :3]  # [batch_size, 3]
        yb_gear = yb[:, 3].long() + 1  # [batch_size], shift gear (-1 to 6) to (0 to 7)
        
        # Forward pass
        control_pred, gear_logits = model(xb)
        
        # Compute losses
        control_loss = control_criterion(control_pred, yb_control)
        gear_loss = gear_criterion(gear_logits, yb_gear)
        loss = control_loss_weight * control_loss + gear_loss_weight * gear_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train += loss.item() * xb.size(0)
    epoch_train_loss = running_train / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # --- Validate ---
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_control = yb[:, :3]
            yb_gear = yb[:, 3].long() + 1
            control_pred, gear_logits = model(xb)
            control_loss = control_criterion(control_pred, yb_control)
            gear_loss = gear_criterion(gear_logits, yb_gear)
            loss = control_loss_weight * control_loss + gear_loss_weight * gear_loss
            running_val += loss.item() * xb.size(0)
    epoch_val_loss = running_val / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch:3d}: Train Loss {epoch_train_loss:.4f}, Val Loss {epoch_val_loss:.4f}")

# 4) Save the trained model
torch.save(model.state_dict(), 'driving_clone.pt')
print("Model saved to driving_clone.pt")

# 5) Compute test-set metrics on validation split
model.eval()
all_control_preds = []
all_gear_preds = []
all_control_true = []
all_gear_true = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        control_pred, gear_logits = model(xb)
        gear_pred = torch.argmax(gear_logits, dim=1).cpu().numpy() - 1  # Shift back to -1 to 6
        control_pred = control_pred.cpu().numpy()
        all_control_preds.append(control_pred)
        all_gear_preds.append(gear_pred)
        all_control_true.append(yb[:, :3].numpy())
        all_gear_true.append(yb[:, 3].numpy())
all_control_preds = np.vstack(all_control_preds)
all_gear_preds = np.hstack(all_gear_preds)
all_control_true = np.vstack(all_control_true)
all_gear_true = np.hstack(all_gear_true)

# Continuous targets: accel, brake, steer
mae = mean_absolute_error(all_control_true, all_control_preds)
r2 = r2_score(all_control_true, all_control_preds)
print(f"MAE (accel/brake/steer): {mae:.4f}")
print(f"R² (accel/brake/steer): {r2:.4f}")

# Gear accuracy
gear_acc = np.mean(all_gear_preds == all_gear_true)
print(f"Gear classification accuracy: {gear_acc * 100:.1f}%")

# 6) Plot training and validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training & Validation Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot.png')