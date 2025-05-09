import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1) Sensor fields and fixed lengths
SENSOR_FIELDS = [
    "angle", "curLapTime", "damage", "distFromStart", "distRaced",
    "fuel", "gear", "lastLapTime", "opponents", "racePos", "rpm",
    "speedX", "speedY", "speedZ", "track", "trackPos", "wheelSpinVel",
    "z", "focus"
]
FIELD_LENGTHS = {
    'angle':1,'curLapTime':1,'damage':1,'distFromStart':1,'distRaced':1,
    'fuel':1,'gear':1,'lastLapTime':1,'opponents':36,'racePos':1,'rpm':1,
    'speedX':1,'speedY':1,'speedZ':1,'track':19,'trackPos':1,'wheelSpinVel':4,
    'z':1,'focus':5
}

# pattern for parsing control string
_action_pattern = re.compile(r'\((\w+)\s+([\-\d\. ]+)\)')

# parse sensor CSV cell
import pandas as pd

def parse_sensor_field(s):
    if pd.isna(s) or s == '':
        return []
    if isinstance(s, str):
        return [float(x) for x in s.split()]
    return [float(s)]

# parse one UDP message into features
def features_from_msg(msg_str):
    from msgParser import MsgParser
    parser = MsgParser()
    sensors = parser.parse(msg_str)
    feats = []
    for f in SENSOR_FIELDS:
        vals = sensors.get(f, [])
        feats.extend([float(v) for v in vals])
    return np.array(feats, dtype=np.float32)

# load CSV, split into DataLoaders

def load_data(csv_path='client_log.csv', batch_size=64, test_size=0.2):
    df = pd.read_csv(csv_path)
    # build X
    X_list = []
    for _, row in df.iterrows():
        vals = []
        for f in SENSOR_FIELDS:
            vals.extend(parse_sensor_field(row[f]))
        X_list.append(vals)
    X = np.array(X_list, dtype=np.float32)
    # build y
    y_list = []
    for cmd in df['Sent']:
        accel = brake = steer = 0.0
        gear = 1
        if isinstance(cmd, str):
            for tag, nums in _action_pattern.findall(cmd):
                vals = [float(v) for v in nums.split()]
                if tag == 'accel': accel = vals[0]
                elif tag == 'brake': brake = vals[0]
                elif tag == 'steer': steer = vals[0]
                elif tag == 'gear': gear = int(vals[0])
        y_list.append([accel, brake, steer, gear])
    y = np.array(y_list, dtype=np.float32)
    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # make loaders
    def to_ds(Xa, ya): return TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
    train_loader = DataLoader(to_ds(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(to_ds(X_val,   y_val),   batch_size=batch_size)
    # input dim
    input_dim = X.shape[1]
    return train_loader, val_loader, input_dim


