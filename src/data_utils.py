import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Sensor and action fields
SENSOR_FIELDS = [
    "angle", "curLapTime", "damage", "distFromStart", "distRaced",
    "fuel", "gear", "lastLapTime", "opponents", "racePos", "rpm",
    "speedX", "speedY", "speedZ", "track", "trackPos", "wheelSpinVel",
    "z", "focus"
]
ACTION_FIELDS = ["accel", "brake", "steer", "gear"]

FIELD_LENGTHS = {
    'angle': 1, 'curLapTime': 1, 'damage': 1, 'distFromStart': 1, 'distRaced': 1,
    'fuel': 1, 'gear': 1, 'lastLapTime': 1, 'opponents': 36, 'racePos': 1, 'rpm': 1,
    'speedX': 1, 'speedY': 1, 'speedZ': 1, 'track': 19, 'trackPos': 1, 'wheelSpinVel': 4,
    'z': 1, 'focus': 5
}

# Parse sensor or action field
def parse_field(s):
    if pd.isna(s) or s == '':
        return []
    if isinstance(s, str):
        return [float(x) for x in s.split()]
    return [float(s)]

# Parse one UDP message into features
def features_from_msg(msg_str):
    from msgParser import MsgParser
    parser = MsgParser()
    sensors = parser.parse(msg_str)
    feats = []
    for f in SENSOR_FIELDS:
        vals = sensors.get(f, [])
        feats.extend([float(v) for v in vals])
    return np.array(feats, dtype=np.float32)

# Load CSV, split into DataLoaders
def load_data(csv_path='client_log.csv', batch_size=64, test_size=0.2):
    df = pd.read_csv(csv_path)
    
    # Filter out rows with negative curLapTime
    df = df[df['curLapTime'] >= 0]
    
    # Build X (sensor features)
    X_list = []
    for _, row in df.iterrows():
        vals = []
        for f in SENSOR_FIELDS:
            vals.extend(parse_field(row[f]))
        X_list.append(vals)
    X = np.array(X_list, dtype=np.float32)
    
    # Normalize sensor features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Build y (action features) using last 4 columns
    action_columns = df.columns[-4:]  # Ensures accel, brake, steer, action gear
    y_list = []
    for _, row in df.iterrows():
        vals = [float(row[col]) for col in action_columns]
        y_list.append(vals)
    y = np.array(y_list, dtype=np.float32)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Make DataLoaders
    def to_ds(Xa, ya): return TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
    train_loader = DataLoader(to_ds(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_ds(X_val, y_val), batch_size=batch_size)
    
    # Input dimension
    input_dim = X.shape[1]
    return train_loader, val_loader, input_dim, scaler