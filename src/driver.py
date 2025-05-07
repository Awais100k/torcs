import sys
import pygame
import msgParser
import carState
import carControl
import time
import keyboard
import csv

class Driver(object):
    '''
    A driver object for the SCRC with manual control using pygame,
    but with automatic gear shifting.
    
    This version logs each turn's sensor data and sent control message
    to a CSV file ("client_log.csv"). Each row contains a timestamp,
    one column per sensor field, and the control (sent) message.
    '''

    # Define the sensor fields that we want to log (order matters).
    SENSOR_FIELDS = [
        "angle", "curLapTime", "damage", "distFromStart", "distRaced",
        "fuel", "gear", "lastLapTime", "opponents", "racePos", "rpm",
        "speedX", "speedY", "speedZ", "track", "trackPos", "wheelSpinVel",
        "z", "focus"
    ]
    
    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.manual_gear = 1  # start in first gear
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 200
        self.prev_rpm = None
        self.steer = 0.0

        # Open a CSV log file and write a header.
        self.log_file = open("client_log.csv", "a", newline="")
        self.csv_writer = csv.writer(self.log_file, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        # If file is empty, write header: timestamp, each sensor field, and "Sent".
        if self.log_file.tell() == 0:
            header = ["Timestamp"] + Driver.SENSOR_FIELDS + ["Sent"]
            self.csv_writer.writerow(header)
        
        # Initialize pygame and create a small window for capturing events.
        pygame.init()
        self.screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption("Manual Control (Auto Gear)")
        self.clock = pygame.time.Clock()

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        # Get current timestamp.
        current_time = time.time()
        # Parse the sensor message into a dictionary.
        sensor_dict = self.parser.parse(msg)
        # Log sensor data into CSV (each field in its own column) plus the sent control message.
        send_msg = self._log_sensor_data(sensor_dict, current_time)
        
        # Update car state based on the sensor message.
        self.state.setFromMsg(msg)
        # Process keyboard events (steering, acceleration, braking).
        self.handle_keyboard_input()
        # Apply automatic gear shifting.
        self.auto_gear()
        # Form the control message to be sent.
        send_msg = self.control.toMsg()
        
        # Optionally, update the log row with the control message if needed.
        # (For simplicity, here we include the sent message in the same row.)
        
        # Process pygame events to keep the window responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.onShutDown()
                sys.exit()
                
        # Debug: Display key information on the pygame window.
        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Speed: {self.state.getSpeedX()}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        text = font.render(f"RPM: {self.state.getRpm()}", True, (255, 255, 255))
        self.screen.blit(text, (10, 50))
        text = font.render(f"Gear: {self.state.getGear()}", True, (255, 255, 255))
        self.screen.blit(text, (10, 90))
        text = font.render(f"Steer: {self.steer:.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 130))
        text = font.render(f"Accel: {self.control.getAccel():.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 170))
        text = font.render(f"Brake: {self.control.getBrake():.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 210))
        text = font.render(f"Clutch: {self.control.getClutch():.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 250))
        text = font.render(f"Meta: {self.control.getMeta()}", True, (255, 255, 255))
        self.screen.blit(text, (10, 290))
        pygame.display.flip()

        return send_msg
    
    def _log_sensor_data(self, sensor_dict, timestamp):
        """
        Logs the sensor dictionary to CSV as a single row.
        Lists are joined by a space.
        Returns the control message that will be sent (empty here; update later if needed).
        """
        row = [f"{timestamp:.3f}"]
        for field in Driver.SENSOR_FIELDS:
            value = sensor_dict.get(field, "")
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            row.append(value)
        # For now, leave the "Sent" field empty; it will be updated after forming the control message.
        # Alternatively, you can log the sent control message here.
        row.append("")  
        self.csv_writer.writerow(row)
        self.log_file.flush()
        return ""
    
    def handle_keyboard_input(self):
        '''Process keyboard events using pygame and the keyboard library for steering and acceleration.
           Gear shifting is handled automatically.
        '''
        # Process pending pygame events (to keep window responsive).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.onShutDown()
                sys.exit()
        
        keys = pygame.key.get_pressed()
        
        # Get current speed from car state.
        speed = self.state.getSpeedX() if self.state.getSpeedX() is not None else 0.0
        accel = self.control.getAccel()  # current acceleration value
        gear = self.manual_gear
        
        # Steering control using the keyboard library.
        if keyboard.is_pressed('a'):
            self.steer += 0.1
            if self.steer > 1.0:
                self.steer = 1.0
            self.control.setSteer(self.steer)
        elif keyboard.is_pressed('d'):
            self.steer -= 0.1
            if self.steer < -1.0:
                self.steer = -1.0
            self.control.setSteer(self.steer)
        else:
            self.steer = 0.0
            self.control.setSteer(self.steer)
        
        # Acceleration/Braking and Reverse logic.
        if keyboard.is_pressed('s'):
            if gear >= 1 and speed > 0.1:
                self.control.setBrake(1.0)
                self.control.setAccel(0.0)
            else:
                if gear != -1:
                    self.manual_gear = -1
                    self.control.setGear(-1)
                    accel = 0.0
                accel += 0.1
                if accel > 1.0:
                    accel = 1.0
                self.control.setBrake(0.0)
                self.control.setAccel(accel)
        elif keyboard.is_pressed('w'):
            if gear < 0 and abs(speed) > 0.1:
                self.control.setBrake(1.0)
                self.control.setAccel(0.0)
            else:
                if gear < 0:
                    self.manual_gear = 1
                    self.control.setGear(1)
                    accel = 0.0
                accel += 0.1
                if accel > 1.0:
                    accel = 1.0
                self.control.setBrake(0.0)
                self.control.setAccel(accel)
        else:
            self.control.setAccel(0.0)
            self.control.setBrake(0.0)
        self.clock.tick(60)
    
    def auto_gear(self):
        '''Automatically adjust gears based on RPM for forward gears only.'''
        # If reverse is engaged in our manual gear, don't change it.
        if self.manual_gear < 0:
            return

        rpm = self.state.getRpm()
        gear = self.manual_gear  # using our manual gear state

        if gear is None:
            gear = 1

        if self.prev_rpm is None:
            self.prev_rpm = rpm

        current_time = time.time()
        min_shift_interval = 0.5
        if hasattr(self, "last_shift_time"):
            if current_time - self.last_shift_time < min_shift_interval:
                return
        else:
            self.last_shift_time = current_time

        upshift_threshold = 7300
        downshift_threshold = 2700

        new_gear = gear

        speed = self.state.getSpeedX() if self.state.getSpeedX() is not None else 0.0
        if abs(speed) < 10:
            return

        if rpm > upshift_threshold and gear < 6:
            new_gear = gear + 1
        elif rpm < downshift_threshold and gear > 1:
            new_gear = gear - 1

        if new_gear != gear:
            self.manual_gear = new_gear
            self.control.setGear(new_gear)
            self.last_shift_time = current_time

        self.prev_rpm = rpm

    def onShutDown(self):
        self.log_file.close()
        pygame.quit()
    
    def onRestart(self):
        pass
