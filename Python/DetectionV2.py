import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# History erhöht für weniger Geistereffekt
fgbg = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=350.0, detectShadows=False)

# Größere Kernel für besseres Schließen von Lücken
kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((9, 9), np.uint8)
kernel_close = np.ones((18, 18), np.uint8)  # Größer!

master_mask = None
max_area_sum = 0

while True:
    frame = cam.capture_array()

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    fgmask = fgbg.apply(gray)

    # Starkes Closing zuerst
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    # Mehr Dilation für Lückenschluss
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=3)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    min_area = 80
    area_sum = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            # cv2.FILLED statt -1 für garantierte Füllung
            cv2.drawContours(mask_filled, [c], -1, 255, cv2.FILLED)
            area_sum += cv2.contourArea(c)

    mask_filled = cv2.medianBlur(mask_filled, 3)

    # Beste Silhouette speichern
    if area_sum > max_area_sum and area_sum > 400:
        max_area_sum = area_sum
        master_mask = mask_filled.copy()

    if area_sum > 120:
        display_mask = mask_filled
    elif master_mask is not None:
        display_mask = master_mask
    else:
        display_mask = mask_filled

    cv2.imshow("Person Mask", display_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
