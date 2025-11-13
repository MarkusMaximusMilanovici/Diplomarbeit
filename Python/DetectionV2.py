import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

fgbg = cv2.createBackgroundSubtractorKNN(history=70, dist2Threshold=400.0, detectShadows=False)

kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((7, 7), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

min_area = 80
active_area_sum = 120

last_mask = None

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
    area_sum = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(mask_filled, [c], -1, 255, -1)
            area_sum += cv2.contourArea(c)
    active_person_found = area_sum > active_area_sum
    print(active_person_found, area_sum)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    min_area = 150        # Toleranz für kleine Konturen erhöht
    area_sum = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(mask_filled, [c], -1, 255, -1)
            area_sum += cv2.contourArea(c)
    active_person_found = area_sum > 350   # Gesamtfläche für „aktiv“ deutlich niedriger

    mask_filled = cv2.medianBlur(mask_filled, 3)

    # --- Zeige letzte Maske, falls gerade keine erkannt ---
    if active_person_found:
        last_mask = mask_filled.copy()
    elif last_mask is not None:
        mask_filled = last_mask.copy()

    cv2.imshow("Person Mask", mask_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
