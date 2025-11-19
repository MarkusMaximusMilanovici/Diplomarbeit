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
    if len(frame.shape) == 2:
        return frame
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

fgbg = cv2.createBackgroundSubtractorKNN(history=80, dist2Threshold=150.0, detectShadows=False)
kernel_erode = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((11, 11), np.uint8)  # Etwas größer, glättet besser
kernel_close = np.ones((17, 17), np.uint8)
kernel_open = np.ones((7, 7), np.uint8)

MIN_CONTOUR_AREA = 300
CALIBRATION_TIME = 2.0
start_time = time.time()
is_calibrating = True

print("[INFO] Calibration phase, please step out of view for {:.1f}s...".format(CALIBRATION_TIME))

while True:
    if use_picamera:
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            break

    gray = get_gray(frame)
    gray = cv2.bilateralFilter(gray, 11, 70, 70)
    gray = cv2.equalizeHist(gray)

    # -- Kalibrierung --
    if is_calibrating:
        fgmask = fgbg.apply(gray, learningRate=0.2)
        time_left = max(0, int(CALIBRATION_TIME - (time.time() - start_time) + 1))
        calib_screen = np.ones_like(gray)*255
        text = f"Calibrating... {time_left}s"
        cv2.putText(calib_screen, text, (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        cv2.imshow("TikTok Silhouette", calib_screen)
        if (time.time() - start_time) >= CALIBRATION_TIME:
            is_calibrating = False
            print("[INFO] Calibration complete!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # -- Bewegungsmaske + adaptive Schwelle --
    motion_mask = fgbg.apply(gray, learningRate=0.003)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)
    combined_mask = cv2.bitwise_or(motion_mask, adaptive)

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=2)
    combined_mask = cv2.erode(combined_mask, kernel_erode, iterations=1)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    # -- NUR größte Kontur: Person schwarz, Rest weiß --
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(combined_mask) * 255  # Start: alles weiß
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > MIN_CONTOUR_AREA:
            cv2.drawContours(mask, [biggest], -1, 0, cv2.FILLED)  # Person wird SCHWARZ

    cv2.imshow("TikTok Silhouette", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean-up
if use_picamera:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()
print("[INFO] Cleanup complete")
