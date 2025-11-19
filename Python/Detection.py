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

def get_gray(frame):
    # If IR camera delivers single channel, use directly; else convert
    if len(frame.shape) == 2:
        return frame
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

fgbg = cv2.createBackgroundSubtractorKNN(history=90, dist2Threshold=150.0, detectShadows=False)
kernel_erode = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)
kernel_close = np.ones((13, 13), np.uint8)
kernel_open = np.ones((7, 7), np.uint8)

master_mask = None
max_area_sum = 0
MIN_CONTOUR_AREA = 120
HYSTERESIS_LOW = 90
DISPLAY_SCALE = 1
minimal_motion_frames = 0
MINIMAL_MOTION_FRAMES_THRESHOLD = 10
CALIBRATION_TIME = 2.5
start_time = time.time()
is_calibrating = True

print("[INFO] Calibration phase. Please clear the view for {CALIBRATION_TIME}s...")

while True:
    time_now = time.time()
    time_since_start = time_now - start_time

    if use_picamera:
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            break

    gray = get_gray(frame)
    # Bilateral filter for noise, keep edges (best in IR)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    # Contrast boost
    gray = cv2.equalizeHist(gray)

    if is_calibrating:
        fgmask = fgbg.apply(gray, learningRate=0.15)
        remaining = max(0, int(CALIBRATION_TIME - time_since_start + 1))
        calib_screen = np.zeros((720, 1280), dtype=np.uint8)
        text = f"Calibrating... {remaining}s"
        cv2.putText(calib_screen, text, (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("Person Mask", calib_screen)
        if time_since_start >= CALIBRATION_TIME:
            is_calibrating = False
            print("[INFO] Calibration complete. Ready for IR detection!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Adaptive threshold to pick up weak features
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 7)
    # Combine with motion mask
    motion_mask = cv2.bitwise_or(fgbg.apply(gray, learningRate=0.003), adaptive)
    # Clean up with morphology
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_open)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_close)
    motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=2)
    motion_mask = cv2.erode(motion_mask, kernel_erode, iterations=1)
    _, motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)

    # Edge detection (optional, as overlay)
    edges = cv2.Canny(gray, 12, 42)
    edge_overlay = cv2.bitwise_and(edges, motion_mask)

    # Find contours only in full mask (not edges)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(motion_mask)
    contour_list = []
    area_sum = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            contour_list.append(c)
            area_sum += area
    # Fill largest contour (person!)
    if contour_list:
        biggest = max(contour_list, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)
    # Add edge overlay for visual feedback
    mask_filled = cv2.bitwise_or(mask_filled, edge_overlay)

    # Motion freeze logic
    if area_sum > HYSTERESIS_LOW:
        minimal_motion_frames = 0
        master_mask = mask_filled.copy()
        status = "ACTIVE"
    else:
        minimal_motion_frames += 1
        status = "IDLE"

    if minimal_motion_frames >= MINIMAL_MOTION_FRAMES_THRESHOLD:
        display_mask = master_mask if master_mask is not None else mask_filled
        status = "FROZEN"
    else:
        display_mask = mask_filled

    height, width = display_mask.shape
    display_upscaled = cv2.resize(display_mask,
        (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)), interpolation=cv2.INTER_CUBIC)
    cv2.putText(display_upscaled, f"Status: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.imshow("Person Mask IR", display_upscaled)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        master_mask = None
        minimal_motion_frames = 0
        print("[RESET] Master mask cleared")

if use_picamera:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()
print("[INFO] Cleanup complete")
