import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100
print("Kalibrierung: Bitte den sichtbaren Bereich verlassen. Die Kamera lernt den Hintergrund.")

for i in range(calibration_frames):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg.apply(gray, learningRate=0.5)
    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow('Kalibrierung', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

print("Kalibrierung abgeschlossen! Du kannst jetzt ins Bild.")

# ====== Hauptloop ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    median = cv2.medianBlur(blurred, 9)

    # MediaPipe Person Segmentierung
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Bewegungsmaske (fgmask) und Morphologische Reinigung VOR Canny
    fgmask = fgbg.apply(gray, learningRate=0)
    fgmask = cv2.medianBlur(fgmask, 7)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    # Vorverarbeitung: Erode und Dilate
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Canny-Kantenerkennung auf dem vorverarbeiteten fgmask
    edges = cv2.Canny(fgmask, 60, 130)

    # --- Opening (Erosion gefolgt von Dilation) auf Kantenbild ---
    edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # --- Closing (Dilation gefolgt von Erosion) auf Kantenbild ---
    edges_clean = cv2.morphologyEx(edges_open, cv2.MORPH_CLOSE, kernel)

    # Hybrid-Maske bilden
    final_mask = ki_mask.copy()
    final_mask = cv2.bitwise_or(final_mask, edges_clean)

    out = np.zeros_like(frame)
    out[final_mask > 0] = [255, 255, 255]

    cv2.imshow('Hybrid Silhouette', out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
