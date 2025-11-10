# Patrol-robo
#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
from datetime import datetime
from threading import Thread
from collections import deque
from gpiozero import Robot, DistanceSensor, MotionSensor, Buzzer, LED
from signal import pause

LEFT_MOTOR_PINS  = (5, 6)
RIGHT_MOTOR_PINS = (13, 19)
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24
PIR_PIN         = 17
BUZZER_PIN      = 27
LED_PIN         = 22

PATROL_SPEED        = 0.4
OBSTACLE_STOP_CM    = 25.0
RANDOM_TURN_SEC     = (0.6, 1.2)
FORWARD_STEP_SEC    = 2.0

CAM_INDEX           = 0
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
CONF_THRESHOLD      = 0.45
USE_DNN_FIRST       = True

LOG_DIR             = os.path.expanduser("~/intruder_logs")
CLIP_SECONDS        = 8
FPS                 = 10

PROTOTXT_PATH       = os.path.expanduser("~/models/deploy.prototxt")
MODEL_PATH          = os.path.expanduser("~/models/MobileNetSSD_deploy.caffemodel")

os.makedirs(LOG_DIR, exist_ok=True)

robot  = Robot(left=LEFT_MOTOR_PINS, right=RIGHT_MOTOR_PINS, pwm=True)
sonar  = DistanceSensor(echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIG, max_distance=2.0)
pir    = MotionSensor(PIR_PIN)
buzzer = Buzzer(BUZZER_PIN)
led    = LED(LED_PIN)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

net = None
CLASS_NAMES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

if USE_DNN_FIRST and os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    except:
        net = None

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

bg_history = deque(maxlen=4)
MOTION_THRESH = 25
MOTION_MIN_AREA = 1500

def detect_person_dnn(frame):
    if net is None:
        return False
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    (h, w) = frame.shape[:2]
    found = False
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        idx  = int(detections[0, 0, i, 1])
        if conf >= CONF_THRESHOLD and 0 <= idx < len(CLASS_NAMES):
            label = CLASS_NAMES[idx]
            if label == "person":
                found = True
                break
    return found

def detect_person_hog(frame):
    resized = cv2.resize(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2))
    rects, weights = hog.detectMultiScale(resized, winStride=(8, 8),padding=(8, 8), scale=1.05)
    return len(rects) > 0

def detect_motion(frame_gray):
    bg_history.append(frame_gray)
    if len(bg_history) < bg_history.maxlen:
        return False
    delta = cv2.absdiff(bg_history[0], bg_history[-1])
    thresh = cv2.threshold(delta, MOTION_THRESH, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > MOTION_MIN_AREA:
            return True
    return False

def record_clip(frames, basepath):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    out_path = f"{basepath}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path

def grab_frames_for(seconds):
    frames = []
    end_time = time.time() + seconds
    while time.time() < end_time and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        time.sleep(1.0 / FPS)
    return frames

def alert_sequence(frame):
    led.blink(on_time=0.1, off_time=0.1)
    buzzer.on()
    time.sleep(0.4)
    buzzer.off()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(LOG_DIR, f"intruder_{ts}")
    cv2.imwrite(f"{base}.jpg", frame)
    clip_frames = grab_frames_for(CLIP_SECONDS)
    record_clip(clip_frames, base)
    led.off()

def obstacle_ahead():
    try:
        d = sonar.distance * 100.0
        return d < OBSTACLE_STOP_CM
    except:
        return False

def safe_forward(duration=FORWARD_STEP_SEC, speed=PATROL_SPEED):
    start = time.time()
    while time.time() - start < duration:
        if obstacle_ahead():
            robot.stop()
            robot.left(speed)
            time.sleep(np.random.uniform(*RANDOM_TURN_SEC))
            robot.stop()
        else:
            robot.forward(speed)
            time.sleep(0.1)
    robot.stop()

def random_turn(speed=PATROL_SPEED):
    if np.random.rand() < 0.5:
        robot.left(speed)
    else:
        robot.right(speed)
    time.sleep(np.random.uniform(*RANDOM_TURN_SEC))
    robot.stop()

def main():
    led.on()
    time.sleep(0.5)
    led.off()
    while True:
        safe_forward()
        random_turn()
        intruder = False
        if pir.motion_detected:
            intruder = True
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not intruder and net is not None:
                intruder = detect_person_dnn(frame)
            if not intruder:
                intruder = detect_person_hog(frame)
            if not intruder:
                intruder = detect_motion(gray)
        if intruder:
            robot.stop()
            if ok:
                alert_sequence(frame)
            else:
                led.blink(on_time=0.1, off_time=0.1, n=10)
                buzzer.beep(on_time=0.2, off_time=0.1, n=5)
            time.sleep(3)
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
        led.off()
        buzzer.off()
        if cap and cap.isOpened():
            cap.release()
