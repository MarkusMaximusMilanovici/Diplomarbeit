import cv2
import numpy as np
import platform
import time

# Detect platform and initialize camera
system = platform.system()
print(f"[INFO] Detected system: {system}")

if system == "Linux":
    try:
        from picamera2 import Picamera2

        print("[INFO] Initializing Picamera2...")
        cam = Picamera2()
        config = cam.create_preview_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        use_picamera = True
    except ImportError:
        print("[WARNING] Picamera2 not found, using webcam...")
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        use_picamera = False
else:
    print("[INFO] Initializing webcam...")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    use_picamera = False

# Background subtractor - moderater gegen Noise
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=450.0, detectShadows=False)

# FEINE Morphology kernels - wie vorher!
kernel_erode = np.ones((4, 4), np.uint8)
kernel_dilate = np.ones((8, 8), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)
kernel_open = np.ones((5, 5), np.uint8)

master_mask = None
max_area_sum = 0

# Timer-basiertes System
start_time = time.time()
last_motion_time = time.time()
last_save_time = time.time()
CALIBRATION_TIME = 3.0
MOTION_TIMEOUT = 2.0
SAVE_INTERVAL = 0.2

MIN_CONTOUR_AREA = 150
HYSTERESIS_HIGH = 350
HYSTERESIS_LOW = 150

DISPLAY_SCALE = 1

print("[INFO] Calibrating background for 3 seconds...")
print("[INFO] Please step out of camera view!")

is_calibrated = False

while True:
    current_time = time.time()
    time_since_start = current_time - start_time

    # Capture frame
    if use_picamera:
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

    # Ensure BGR format
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # LEICHTES Denoising - behält Details
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # KALIBRIERUNGSPHASE
    if time_since_start < CALIBRATION_TIME:
        fgmask = fgbg.apply(gray, learningRate=0.1)

        calib_screen = np.zeros((720, 1280), dtype=np.uint8)
        remaining = int(CALIBRATION_TIME - time_since_start) + 1
        text = f"Calibrating... {remaining}s"
        cv2.putText(calib_screen, text, (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("Person Mask", calib_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if not is_calibrated:
        is_calibrated = True
        print("[INFO] Calibration complete! System ready.")

    # Normale Verarbeitung
    fgmask = fgbg.apply(gray, learningRate=0.001)

    # FEINE Morphologische Operationen - nur 1 Iteration!
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # Konturen OHNE Convex Hull - behält Finger!
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    area_sum = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            # KEIN Convex Hull - originale Kontur für Details!
            cv2.drawContours(mask_filled, [c], -1, 255, cv2.FILLED)
            area_sum += area

    if area_sum > 0:
        mask_filled = cv2.medianBlur(mask_filled, 5)

    # Timer-basierte Bewegungserkennung
    time_since_motion = current_time - last_motion_time
    time_since_save = current_time - last_save_time

    if area_sum > MIN_CONTOUR_AREA:
        last_motion_time = current_time
        has_motion = True
    else:
        has_motion = False

    # Verbessertes Speichern
    if area_sum > HYSTERESIS_HIGH and time_since_save > SAVE_INTERVAL:
        if master_mask is None or area_sum > max_area_sum * 0.75:
            if area_sum > max_area_sum:
                max_area_sum = area_sum
            master_mask = mask_filled.copy()
            last_save_time = current_time
            print(f"[SAVE] Area: {area_sum:.0f} | Max: {max_area_sum:.0f}")

    # Anzeige-Logik
    if has_motion and area_sum > HYSTERESIS_LOW:
        display_mask = mask_filled
        status = "ACTIVE"
    elif time_since_motion < MOTION_TIMEOUT:
        display_mask = mask_filled if area_sum > 50 else (master_mask if master_mask is not None else mask_filled)
        status = f"TRANSITION ({MOTION_TIMEOUT - time_since_motion:.1f}s)"
    else:
        if master_mask is not None:
            display_mask = master_mask
            status = "FROZEN (saved)"
        else:
            display_mask = mask_filled
            status = "FROZEN (no save)"

    # Upscaling
    height, width = display_mask.shape
    new_width = int(width * DISPLAY_SCALE)
    new_height = int(height * DISPLAY_SCALE)
    display_upscaled = cv2.resize(display_mask, (new_width, new_height),
                                  interpolation=cv2.INTER_CUBIC)

    # Info-Overlay
    font_scale = 0.6
    thickness = 2
    info1 = f"Status: {status}"
    info2 = f"Area: {area_sum:.0f} | Motion: {time_since_motion:.1f}s | Saved: {'YES' if master_mask is not None else 'NO'}"

    cv2.putText(display_upscaled, info1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)
    cv2.putText(display_upscaled, info2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

    cv2.imshow("Person Mask", display_upscaled)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        master_mask = None
        max_area_sum = 0
        last_motion_time = current_time
        print(f"[RESET]")
    elif key == ord('s'):
        if area_sum > 0:
            master_mask = mask_filled.copy()
            max_area_sum = area_sum
            print(f"[MANUAL SAVE] Area: {area_sum:.0f}")
    elif key == ord('c'):
        start_time = time.time()
        is_calibrated = False
        print("[INFO] Recalibrating...")

# Cleanup
if use_picamera:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()
print("[INFO] Cleanup complete")
