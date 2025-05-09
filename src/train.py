# train.py
import torch
import torch.nn as nn
from data_utils import load_data
from model import DrivingMLP

# hyperparameters
batch_size = 64
num_epochs = 20
lr = 1e-3

# load data
train_loader, val_loader, input_dim = load_data(batch_size=batch_size)

# model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrivingMLP(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:2d}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

# save
torch.save(model.state_dict(), 'driving_clone.pt')

