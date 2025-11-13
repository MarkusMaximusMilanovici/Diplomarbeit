import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# Schneller reagierende Hintergrundsubtraktion
fgbg = cv2.createBackgroundSubtractorKNN(history=70, dist2Threshold=400.0, detectShadows=False)

kernel_erode = np.ones((2, 2), np.uint8)  # Feinere Erode-Kernel
kernel_dilate = np.ones((7, 7), np.uint8)  # Feinere Dilate-Kernel

while True:
    frame = cam.capture_array()

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Optional: Histogrammausgleich für mehr Kontrast
    gray = cv2.equalizeHist(gray)

    fgmask = fgbg.apply(gray)

    # Morphologische Operationen (erst Dilate, dann Erode für sharpes Closing)
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    # Binarisierung (harte Schwelle)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # Konturen berechnen
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 600:
            cv2.drawContours(mask_filled, [max_contour], -1, 255, -1)

    # Optional: mildes Glätten für saubere Kanten (kleiner Wert!)
    mask_filled = cv2.medianBlur(mask_filled, 3)

    # Anzeige
    cv2.imshow("Person Mask", mask_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
