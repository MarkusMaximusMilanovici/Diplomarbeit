import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# --- Background subtractor ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=10, detectShadows=False)

# Morphologische Kernel (Größe kann angepasst werden)
kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((13, 13), np.uint8)

while True:
    frame = cam.capture_array()

    # Ensure BGR format
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    fgmask = fgbg.apply(gray)

    # --- Morphologische Operationen ---
    # Kleine weiße Flecken entfernen
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    # Lücken in erkannten Personen schließen
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)

    # Zusätzliche Glättung
    fgmask = cv2.medianBlur(fgmask, 7)

    # Anzeige
    cv2.imshow("Person Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
