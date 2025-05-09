import sys
import argparse
import socket
import driver
import torch
import numpy as np
from model import DrivingMLP
from data_utils import SENSOR_FIELDS, FIELD_LENGTHS, features_from_msg

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--model', action='store', dest='model_path', default='driving_clone.pt',
                    help='Path to cloned model')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Using model:', arguments.model_path)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

# Set up UDP socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)
# one second timeout
sock.settimeout(1.0)

# Load cloned model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = sum(FIELD_LENGTHS[f] for f in SENSOR_FIELDS)
model = DrivingMLP(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(arguments.model_path, map_location=device))
model.eval()

shutdownClient = False
curEpisode = 0
verbose = False

d = driver.Driver(arguments.stage)

# Initial handshake
while not shutdownClient:
    while True:
        print('Sending id to server: ', arguments.id)
        buf = arguments.id + d.init()
        sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
        try:
            resp, _ = sock.recvfrom(1000)
            resp = resp.decode('utf-8')
        except socket.error:
            continue
        if '***identified***' in resp:
            print('Received:', resp)
            break
    break

# Main loop
while not shutdownClient:
    currentStep = 0
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            msg = data.decode('utf-8')
        except socket.error:
            continue

        if verbose:
            print('Received:', msg)

        if '***shutdown***' in msg:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        if '***restart***' in msg:
            d.onRestart()
            print('Client Restart')
            break

        currentStep += 1
        if currentStep != arguments.max_steps:
            # Use cloned model for inference
            feat = features_from_msg(msg)
            tensor = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor).cpu().numpy()[0]
            accel, brake, steer, gear_pred = out
            gear_cmd = int(round(gear_pred))
            buf = f"(accel {accel:.3f})(brake {brake:.3f})(steer {steer:.3f})(gear {gear_cmd})"
        else:
            buf = '(meta 1)'

        if verbose:
            print('Sending:', buf)
        sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))

    curEpisode += 1
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()
