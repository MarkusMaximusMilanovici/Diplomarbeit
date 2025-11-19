import cv2
import numpy as np
import platform

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

# Background subtractor mit höherer History für mehr Stabilität
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=500.0, detectShadows=False)

# Morphology kernels - angepasst für besseres Noise Filtering
kernel_erode = np.ones((4, 4), np.uint8)
kernel_dilate = np.ones((8, 8), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)
kernel_open = np.ones((5, 5), np.uint8)  # Für Opening (Noise removal)

master_mask = None
max_area_sum = 0

# Hysterese-Schwellenwerte für stabileres Umschalten
HYSTERESIS_HIGH = 500  # Mindestfläche zum Speichern neuer Maske
HYSTERESIS_LOW = 200  # Mindestfläche zum Anzeigen aktueller Maske
MIN_CONTOUR_AREA = 150  # Höherer Wert gegen kleine Störungen

# Frame-Buffer für Stabilität
frame_buffer = []
BUFFER_SIZE = 3

print("[INFO] Starting main loop...")

while True:
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

    # Besseres Denoising VOR equalizeHist
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    fgmask = fgbg.apply(gray, learningRate=0.001)  # Langsameres Lernen

    # Opening ZUERST - entfernt kleine Noise-Punkte
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

    # Dann Closing - schließt Lücken
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

    # Dilation und Erosion
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    # Gaussian Blur für weichere Kanten (reduziert Flackern)
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)

    # Threshold
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # Konturen mit höherem Schwellenwert
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    area_sum = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:  # Höherer Schwellenwert
            cv2.drawContours(mask_filled, [c], -1, 255, cv2.FILLED)
            area_sum += area

    # Median Blur nur wenn nötig
    if area_sum > 0:
        mask_filled = cv2.medianBlur(mask_filled, 5)

    # Frame-Buffer für temporale Stabilität
    frame_buffer.append(mask_filled.copy())
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    # Average der letzten Frames
    if len(frame_buffer) == BUFFER_SIZE:
        avg_mask = np.mean(frame_buffer, axis=0).astype(np.uint8)
        _, avg_mask = cv2.threshold(avg_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        avg_mask = mask_filled

    # Hysterese: Nur bei signifikanter Änderung Master-Maske updaten
    if area_sum > HYSTERESIS_HIGH and area_sum > max_area_sum:
        max_area_sum = area_sum
        master_mask = avg_mask.copy()
        print(f"[INFO] New best mask: {max_area_sum:.0f}")

    # Stabileres Umschalten mit Hysterese
    if area_sum > HYSTERESIS_LOW:
        display_mask = avg_mask
    elif master_mask is not None:
        display_mask = master_mask
    else:
        display_mask = avg_mask

    cv2.imshow("Person Mask", display_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if use_picamera:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()
print("[INFO] Cleanup complete")
