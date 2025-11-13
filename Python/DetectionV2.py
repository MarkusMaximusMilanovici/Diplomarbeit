import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# --- Alternative: KNN statt MOG2 ---
fgbg = cv2.createBackgroundSubtractorKNN(history=70, dist2Threshold=400.0, detectShadows=False)
# Für MOG2: # fgbg = cv2.createBackgroundSubtractorMOG2(history=70, varThreshold=25, detectShadows=False)

kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((13, 13), np.uint8)

while True:
    frame = cam.capture_array()

    # Ensure BGR format
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hintergrund-Subtraktion
    fgmask = fgbg.apply(gray)

    # Morphologische Operationen
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)

    # Binarisierung zur Maskenbereinigung
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # Konturen berechnen und größte Fläche als Person maskieren
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 600:  # Mindestgröße (anpassen!)
            cv2.drawContours(mask_filled, [max_contour], -1, 255, -1)

    # Glättung (optional)
    mask_filled = cv2.medianBlur(mask_filled, 7)

    # Anzeige
    cv2.imshow("Person Mask", mask_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
