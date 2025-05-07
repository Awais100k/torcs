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
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        '''Handle one timestep: parse sensors, update controls, log and return command.'''
        current_time = time.time()
        sensor_dict = self.parser.parse(msg)

        # Update car state and controls
        self.state.setFromMsg(msg)
        self.handle_keyboard_input()
        self.auto_gear()
        send_msg = self.control.toMsg()

        # Log sensors + control action
        self._log_sensor_and_control(sensor_dict, current_time, send_msg)

        # Keep pygame window responsive and display debug info
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.onShutDown()
                sys.exit()
        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 24)
        info = [
            f"Speed: {self.state.getSpeedX()}",
            f"RPM: {self.state.getRpm()}",
            f"Gear: {self.state.getGear()}",
            f"Steer: {self.steer:.2f}",
            f"Accel: {self.control.getAccel():.2f}",
            f"Brake: {self.control.getBrake():.2f}",
            f"Clutch: {self.control.getClutch():.2f}",
            f"Meta: {self.control.getMeta()}"
        ]
        y = 10
        for line in info:
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 30
        pygame.display.flip()

        return send_msg
    
    def _log_sensor_and_control(self, sensor_dict, timestamp, send_msg):
        '''Logs a row of sensor values followed by the sent control message.'''
        row = [f"{timestamp:.3f}"]
        for field in Driver.SENSOR_FIELDS:
            val = sensor_dict.get(field, [])
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            row.append(val)
        row.append(send_msg)
        self.csv_writer.writerow(row)
        self.log_file.flush()

    def handle_keyboard_input(self):
        '''Process keyboard for steer, accel, brake; gear auto-handled.'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.onShutDown()
                sys.exit()
        keys = pygame.key.get_pressed()
        speed = self.state.getSpeedX() or 0.0
        accel = self.control.getAccel()
        gear = self.manual_gear
        # steering
        if keyboard.is_pressed('a'):
            self.steer = max(self.steer + 0.1, -1.0)
        elif keyboard.is_pressed('d'):
            self.steer = min(self.steer - 0.1, 1.0)
        else:
            self.steer = 0.0
        self.control.setSteer(self.steer)
        # brake/reverse
        if keyboard.is_pressed('s'):
            if gear >= 1 and speed > 0.1:
                self.control.setBrake(1.0)
                self.control.setAccel(0.0)
            else:
                if gear != -1:
                    self.manual_gear = -1
                    self.control.setGear(-1)
                accel = min(accel + 0.1, 1.0)
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
                accel = min(accel + 0.1, 1.0)
                self.control.setBrake(0.0)
                self.control.setAccel(accel)
        else:
            self.control.setAccel(0.0)
            self.control.setBrake(0.0)
        self.clock.tick(60)
    
    def auto_gear(self):
        '''Auto gear shift logic based on RPM thresholds.'''
        if self.manual_gear < 0:
            return
        rpm = self.state.getRpm() or 0.0
        gear = self.manual_gear
        if self.prev_rpm is None:
            self.prev_rpm = rpm
        now = time.time()
        if hasattr(self, 'last_shift_time') and now - self.last_shift_time < 0.5:
            return
        up_th, down_th = 7300, 2700
        speed = abs(self.state.getSpeedX() or 0.0)
        if speed < 10:
            return
        new_gear = gear
        if rpm > up_th and gear < 6:
            new_gear += 1
        elif rpm < down_th and gear > 1:
            new_gear -= 1
        if new_gear != gear:
            self.manual_gear = new_gear
            self.control.setGear(new_gear)
            self.last_shift_time = now
        self.prev_rpm = rpm

    def onShutDown(self):
        self.log_file.close()
        pygame.quit()
    
    def onRestart(self):
        pass
