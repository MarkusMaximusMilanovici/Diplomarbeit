import cv2
import numpy as np
import platform
import time

system = platform.system()
if system == "Linux":
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        use_picamera = True
    except ImportError:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        use_picamera = False
else:
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    use_picamera = False

fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=300.0, detectShadows=False)
kernel_erode = np.ones((4, 4), np.uint8)
kernel_dilate = np.ones((8, 8), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)
kernel_open = np.ones((5, 5), np.uint8)

master_mask = None
max_area_sum = 0

MIN_CONTOUR_AREA = 150
HYSTERESIS_HIGH = 500
HYSTERESIS_LOW = 200
DISPLAY_SCALE = 1

minimal_motion_frames = 0
MINIMAL_MOTION_FRAMES_THRESHOLD = 10

CALIBRATION_TIME = 3.0
start_time = time.time()
is_calibrating = True

print("[INFO] Starting... Please step out of camera view for calibration.")

while True:
    time_now = time.time()
    time_since_start = time_now - start_time

    # Kameraaufnahme
    if use_picamera:
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            break

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # Kalibrierungsphase
    if is_calibrating:
        fgmask = fgbg.apply(gray, learningRate=0.1)
        remaining = max(0, int(CALIBRATION_TIME - time_since_start + 1))
        calib_screen = np.zeros((720, 1280), dtype=np.uint8)
        text = f"Calibrating... {remaining}s"
        cv2.putText(calib_screen, text, (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("Person Mask", calib_screen)
        if time_since_start >= CALIBRATION_TIME:
            is_calibrating = False
            print("[INFO] Calibration complete. Ready for detection!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    fgmask = fgbg.apply(gray, learningRate=0.001)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)
    # KEIN weiteres Blur/MedianBlur nach Flächenfüllung, um Motion Blur zu minimieren!
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    area_sum = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)
            area_sum += area

    # Motion Detection-Logik
    if area_sum > HYSTERESIS_LOW:
        minimal_motion_frames = 0
        # Sobald Bewegung, immer letzte Maske überschreiben
        master_mask = mask_filled.copy()
        max_area_sum = area_sum
        status = "ACTIVE"
    else:
        minimal_motion_frames += 1
        status = "IDLE"

    # Frozen-Logik: Erst nach mehreren ruhigen Frames wird eingefroren und letzte Maske angezeigt
    if minimal_motion_frames >= MINIMAL_MOTION_FRAMES_THRESHOLD:
        display_mask = master_mask if master_mask is not None else mask_filled
        status = "FROZEN"
    else:
        display_mask = mask_filled

    height, width = display_mask.shape
    display_upscaled = cv2.resize(display_mask, (int(width*DISPLAY_SCALE), int(height*DISPLAY_SCALE)), interpolation=cv2.INTER_CUBIC)

    cv2.putText(display_upscaled, f"Status: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Person Mask", display_upscaled)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        master_mask = None
        max_area_sum = 0
        minimal_motion_frames = 0
        print("[RESET] Master mask cleared")

if use_picamera:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()
print("[INFO] Cleanup complete")
