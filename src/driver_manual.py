import sys
import os
import argparse
import socket
import time

# Use msvcrt for Windows non-blocking input; otherwise use select.
if os.name == 'nt':
    import msvcrt

    def non_blocking_input():
        """Check for input on Windows using msvcrt."""
        if msvcrt.kbhit():
            # getch returns a byte string; decode to UTF-8
            return msvcrt.getch().decode('utf-8')
        return None
else:
    import select

    def non_blocking_input():
        """Check for input on Unix using select."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.readline().strip()
        return None

from carControl import CarControl
from carState import CarState
from msgParser import MsgParser

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Manual client to connect to the TORCS SCRC server with manual control.'
    )
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
    return parser.parse_args()

def main():
    arguments = parse_arguments()

    print(f"Connecting to server at {arguments.host_ip}:{arguments.host_port}")
    print(f"Bot ID: {arguments.id}")
    print(f"Maximum episodes: {arguments.max_episodes}")
    print(f"Maximum steps: {arguments.max_steps}")
    print(f"Track: {arguments.track}")
    print(f"Stage: {arguments.stage}")
    print("*********************************************")
    print("Manual control keys:")
    print("  W: Increase acceleration")
    print("  S: Increase brake")
    print("  A: Steer left")
    print("  D: Steer right")
    print("  X: Reset controls (neutral)")
    print("  Q: Quit")
    print("*********************************************")
    
    # Create UDP socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as msg:
        print("Could not create a socket.")
        sys.exit(-1)
    # Set socket timeout (e.g., 1 second)
    sock.settimeout(1.0)
    
    # Create instances for control and state handling
    control = CarControl()
    state = CarState()
    parser_instance = MsgParser()
    
    # Open file to append sensor data for dataset creation
    dataset_filename = "sensor_dataset.txt"
    dataset_file = open(dataset_filename, "a")

    print("Starting manual control loop. Press Q to quit.")
    try:
        while True:
            # Receive sensor data from the simulator (non-blocking via timeout)
            try:
                data, _ = sock.recvfrom(4096)
                sensor_msg = data.decode("utf-8")
                # Update car state from the sensor message
                state.setFromMsg(sensor_msg)
                # Save raw sensor message to dataset file with timestamp
                dataset_file.write(f"{time.time()} {sensor_msg}\n")
                dataset_file.flush()
            except socket.timeout:
                # No sensor message received this iteration
                sensor_msg = None

            # Check for non-blocking keyboard input
            user_input = non_blocking_input()
            if user_input:
                # Process input commands (case-insensitive)
                key = user_input.lower()
                if key == 'q':
                    print("Quitting manual control.")
                    break
                elif key == 'w':
                    # Increase acceleration (up to a maximum of 1.0)
                    new_accel = min(control.getAccel() + 0.1, 1.0)
                    control.setAccel(new_accel)
                    # Ensure brakes are off when accelerating
                    control.setBrake(0)
                    print(f"Accelerating: {new_accel:.2f}")
                elif key == 's':
                    # Increase brake (up to a maximum of 1.0)
                    new_brake = min(control.getBrake() + 0.1, 1.0)
                    control.setBrake(new_brake)
                    # Ensure acceleration is off when braking
                    control.setAccel(0)
                    print(f"Braking: {new_brake:.2f}")
                elif key == 'a':
                    # Steer left (limit steer value to -1.0)
                    new_steer = max(control.getSteer() - 0.1, -1.0)
                    control.setSteer(new_steer)
                    print(f"Steering left: {new_steer:.2f}")
                elif key == 'd':
                    # Steer right (limit steer value to 1.0)
                    new_steer = min(control.getSteer() + 0.1, 1.0)
                    control.setSteer(new_steer)
                    print(f"Steering right: {new_steer:.2f}")
                elif key == 'x':
                    # Reset controls to neutral
                    control.setAccel(0)
                    control.setBrake(0)
                    control.setSteer(0)
                    print("Controls reset to neutral.")

            # Build the control message using the CarControl instance
            msg_to_send = control.toMsg()
            sock.sendto(msg_to_send.encode("utf-8"), (arguments.host_ip, arguments.host_port))
            
            # Sleep briefly to prevent a tight loop; adjust as needed
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    finally:
        dataset_file.close()
        sock.close()

if __name__ == '__main__':
    main()
