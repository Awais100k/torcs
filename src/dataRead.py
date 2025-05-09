import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1) Load your CSV
df = pd.read_csv('client_log.csv')

# 2) Sensor fields
SENSOR_FIELDS = [
    "angle", "curLapTime", "damage", "distFromStart", "distRaced",
    "fuel", "gear", "lastLapTime", "opponents", "racePos", "rpm",
    "speedX", "speedY", "speedZ", "track", "trackPos", "wheelSpinVel",
    "z", "focus"
]

# 3) Parse functions
def parse_sensor_field(s):
    if pd.isna(s) or s == '':
        return []
    if isinstance(s, str):
        return [float(x) for x in s.split()]
    return [float(s)]

# Build X
feature_list = []
for _, row in df.iterrows():
    vals = []
    for f in SENSOR_FIELDS:
        vals.extend(parse_sensor_field(row[f]))
    feature_list.append(vals)
X = np.array(feature_list, dtype=np.float32)

# Parse y
action_pattern = re.compile(r'\((\w+)\s+([-\d\. ]+)\)')
y_list = []
for cmd in df['Sent']:
    accel = brake = steer = 0.0
    gear = 1
    if isinstance(cmd, str):
        for tag, nums in action_pattern.findall(cmd):
            vals = [float(v) for v in nums.split()]
            if tag == 'accel': accel = vals[0]
            elif tag == 'brake': brake = vals[0]
            elif tag == 'steer': steer = vals[0]
            elif tag == 'gear': gear = int(vals[0])
    y_list.append([accel, brake, steer, gear])
y = np.array(y_list, dtype=np.float32)

# 4) Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) DataLoaders
def to_ds(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

train_loader = DataLoader(to_ds(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(to_ds(X_val, y_val),   batch_size=64)

# 6) Define model
class DrivingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 7) Instantiate & train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrivingMLP(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 201):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    total_val = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total_val += criterion(model(xb), yb).item() * xb.size(0)
    val_loss = total_val / len(val_loader.dataset)

    print(f"Epoch {epoch:2d}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")


# Grab one batch from the validation set
xb, yb = next(iter(val_loader))
xb, yb = xb.to(device), yb.to(device)

model.eval()
with torch.no_grad():
    preds = model(xb)

# Compare first 5 examples
for i in range(5):
    true = yb[i].cpu().numpy()
    pred = preds[i].cpu().numpy()
    print(f"Sample {i}:")
    print(f"  True → accel {true[0]:.2f}, brake {true[1]:.2f}, steer {true[2]:.2f}, gear {true[3]:.0f}")
    print(f"  Pred → accel {pred[0]:.2f}, brake {pred[1]:.2f}, steer {pred[2]:.2f}, gear {pred[3]:.0f}")
    print()



torch.save(model.state_dict(), 'driving_clone.pt')
