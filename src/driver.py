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
    but with automatic and manual gear control.
    Logs each turn's sensor data and sent control message to CSV.
    '''

    SENSOR_FIELDS = [
        "angle", "curLapTime", "damage", "distFromStart", "distRaced",
        "fuel", "gear", "lastLapTime", "opponents", "racePos", "rpm",
        "speedX", "speedY", "speedZ", "track", "trackPos", "wheelSpinVel",
        "z", "focus"
    ]

    def __init__(self, stage):
        self.stage = stage
        self.manual_gear = 1
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.prev_shift_time = 0
        # Hysteresis thresholds
        self.up_threshold = 6000
        self.down_threshold = 3000
        # Steering parameters
        self.steer = 0.0
        self.steer_inc = 0.02
        self.steer_decay = 0.90
        # Acceleration/brake parameters
        self.accel_step = 0.2
        self.accel_decay = 0.95
        # Direction: 1=forward, -1=reverse, 0=neutral
        self.direction = 0

        self.log_file = open("client_log.csv", "a", newline="")
        self.csv_writer = csv.writer(self.log_file, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        if self.log_file.tell() == 0:
            header = ["Timestamp"] + Driver.SENSOR_FIELDS + ["Sent"]
            self.csv_writer.writerow(header)

        pygame.init()
        self.screen = pygame.display.set_mode((200, 300))
        pygame.display.set_caption("Smooth Control (Auto Gear)")
        self.clock = pygame.time.Clock()

    def init(self):
        angles = [0]*19
        for i in range(5):
            angles[i] = -90 + i*15
            angles[18-i] = 90 - i*15
        for i in range(5,9):
            angles[i] = -20 + (i-5)*5
            angles[18-i] = 20 - (i-5)*5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        ts = time.time()
        sensor_dict = self.parser.parse(msg)
        self.state.setFromMsg(msg)
        self._handle_steering()
        self._handle_accel_brake()
        self._auto_gear()
        action = self.control.toMsg()
        self._log(sensor_dict, ts, action)
        self._render()
        return action

    def _handle_steering(self):
        if keyboard.is_pressed('a'):
            self.steer = min(self.steer + self.steer_inc, 1.0)  # LEFT
        elif keyboard.is_pressed('d'):
            self.steer = max(self.steer - self.steer_inc, -1.0)  # RIGHT
        else:
            self.steer *= self.steer_decay
            if abs(self.steer) < 0.01:
                self.steer = 0.0
        self.control.setSteer(self.steer)

    def _handle_accel_brake(self):
        speed = self.state.getSpeedX() or 0.0
        accel = self.control.getAccel()
        brake = self.control.getBrake()
        gear = self.control.getGear() or self.manual_gear

        # Reverse throttle: when in reverse gear and pressing 's'
        if gear == -1 and keyboard.is_pressed('s'):
            accel = min(accel + self.accel_step, 1.0)
            brake = 0.0
            self.direction = -1

        # Forward braking: when in forward gear or neutral and pressing 's'
        elif keyboard.is_pressed('s'):
            if speed > 1.0:
                brake = min(brake + self.accel_step, 1.0)
                accel = 0.0
            else:
                # nearly stopped, shift to reverse if desired
                accel = 0.0
                brake = 0.0
                if self.direction != -1:
                    self.manual_gear = -1
                    self.control.setGear(-1)
                    self.direction = -1

        # Forward throttle: only when pressing 'w'
        elif keyboard.is_pressed('w'):
            if speed < -1.0:
                # moving backward, brake first
                brake = min(brake + self.accel_step, 1.0)
                accel = 0.0
            else:
                if self.direction != 1:
                    self.manual_gear = 1
                    self.control.setGear(1)
                    self.direction = 1
                accel = min(accel + self.accel_step, 1.0)
                brake = 0.0

        # Reset controls
        elif keyboard.is_pressed('x'):
            accel, brake = 0.0, 0.0

        # No input: decay throttle
        else:
            accel *= self.accel_decay
            if accel < 0.01:
                accel = 0.0
            brake = 0.0

        self.control.setAccel(accel)
        self.control.setBrake(brake)

    def _auto_gear(self):
        if self.direction != 1:
            return
        track_pos = self.state.getTrackPos() or 0.0
        if abs(track_pos) > 1.1:
            return
        rpm = self.state.getRpm() or 0.0
        speed = abs(self.state.getSpeedX() or 0.0)
        now = time.time()
        if now - self.prev_shift_time < 0.5:
            return
        gear = self.manual_gear
        if gear < 6 and rpm > self.up_threshold and speed > gear * 20:
            self._shift(gear + 1)
        elif gear > 1 and rpm < self.down_threshold:
            self._shift(gear - 1)

    def _shift(self, new_gear):
        old_acc = self.control.getAccel()
        self.control.setAccel(0.0)
        time.sleep(0.05)
        self.control.setGear(new_gear)
        self.manual_gear = new_gear
        self.prev_shift_time = time.time()
        time.sleep(0.05)
        self.control.setAccel(old_acc)

    def _log(self, sd, ts, action):
        row = [f"{ts:.3f}"]
        for f in Driver.SENSOR_FIELDS:
            v = sd.get(f, [])
            row.append(" ".join(str(x) for x in v) if isinstance(v, list) else v)
        row.append(action)
        self.csv_writer.writerow(row)
        self.log_file.flush()

    def _render(self):
        self.screen.fill((0,0,0))
        font = pygame.font.Font(None, 24)
        lines = [
            f"Speed: {self.state.getSpeedX():.1f}",
            f"RPM: {self.state.getRpm():.0f}",
            f"Gear: {self.state.getGear()}",
            f"Steer: {self.steer:.2f}",
            f"Accel: {self.control.getAccel():.2f}",
            f"Brake: {self.control.getBrake():.2f}"
        ]
        y = 10
        for l in lines:
            self.screen.blit(font.render(l, True, (255,255,255)), (10, y))
            y += 30
        pygame.display.flip()
        self.clock.tick(60)

    def onShutDown(self):
        self.log_file.close()
        pygame.quit()

    def onRestart(self):
        pass
