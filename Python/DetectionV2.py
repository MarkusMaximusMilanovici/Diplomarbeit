import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

fgbg = cv2.createBackgroundSubtractorKNN(history=70, dist2Threshold=400.0, detectShadows=False)

kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((7, 7), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

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
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)

    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    min_area = 80
    area_sum = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(mask_filled, [c], -1, 255, -1)
            area_sum += cv2.contourArea(c)

    mask_filled = cv2.medianBlur(mask_filled, 3)

    # --- Beste Silhouette nur dann speichern, wenn sie maximal groß ist ---
    if area_sum > max_area_sum and area_sum > 400:
        max_area_sum = area_sum
        master_mask = mask_filled.copy()

    # --- Anzeige: aktuelle Maske, sonst die "beste" ---
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
