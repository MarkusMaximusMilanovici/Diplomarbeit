import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

fgbg = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400.0, detectShadows=False)

kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((9, 9), np.uint8)
kernel_close = np.ones((18, 18), np.uint8)

# Grid-Einstellungen: Teile das Bild in 3x3 Bereiche
GRID_ROWS = 3
GRID_COLS = 3

# Speicher für jede Grid-Zelle
grid_memory = {}
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        grid_memory[(row, col)] = None

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
    fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=3)
    fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    # Konturen auf der gesamten Maske
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(fgmask)
    for c in contours:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(mask_filled, [c], -1, 255, cv2.FILLED)

    mask_filled = cv2.medianBlur(mask_filled, 3)

    # Grid-Dimensionen berechnen
    height, width = mask_filled.shape
    cell_height = height // GRID_ROWS
    cell_width = width // GRID_COLS

    # Finale Maske mit Grid-Memory
    final_mask = np.zeros_like(mask_filled)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Grid-Zelle definieren
            y_start = row * cell_height
            y_end = (row + 1) * cell_height if row < GRID_ROWS - 1 else height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width if col < GRID_COLS - 1 else width

            # Aktueller Bereich aus der Maske
            cell_current = mask_filled[y_start:y_end, x_start:x_end]

            # Prüfe ob Bewegung in dieser Zelle (weiße Pixel vorhanden)
            motion_detected = np.count_nonzero(cell_current) > 100  # Schwellenwert anpassbar

            if motion_detected:
                # Aktualisiere Speicher für diese Zelle
                grid_memory[(row, col)] = cell_current.copy()
                final_mask[y_start:y_end, x_start:x_end] = cell_current
            else:
                # Keine Bewegung: Zeige gespeicherte Maske
                if grid_memory[(row, col)] is not None:
                    final_mask[y_start:y_end, x_start:x_end] = grid_memory[(row, col)]

    cv2.imshow("Person Mask", final_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
