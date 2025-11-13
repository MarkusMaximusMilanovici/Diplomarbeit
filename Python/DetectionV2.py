import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

fgbg = cv2.createBackgroundSubtractorKNN(history=70, dist2Threshold=400.0, detectShadows=False)

# Morphologie-Kernel angepasst
kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((13, 13), np.uint8)
kernel_close = np.ones((12, 12), np.uint8)  # Größer für besseres Closing

while True:
    frame = cam.capture_array()

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    fgmask = fgbg.apply(gray)

    # Erst grobes Closing (Lücken verbinden)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    # Dann feine Dilation/Erosion
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    # Durch hartes Thresholding binarisieren
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # ALLE relevanten Konturen füllen
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    for c in contours:
        if cv2.contourArea(c) > 600:
            cv2.drawContours(mask_filled, [c], -1, 255, -1)

    # Optional: mildes Glätten
    mask_filled = cv2.medianBlur(mask_filled, 3)

    # Anzeige
    cv2.imshow("Person Mask", mask_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
