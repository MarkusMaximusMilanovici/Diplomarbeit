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
    cv2.putText(frame, f'Kalibrierung: {i+1}/{calibration_frames}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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

    edges = cv2.Canny(median, 70, 150)
    # MediaPipe Person Segmentierung
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Bewegungsmaske + Kanten-Extraktion
    fgmask = fgbg.apply(gray, learningRate=0)
    fgmask = cv2.medianBlur(fgmask, 7)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(fgmask, 60, 130)

    # === Opening: Erosion & Dilation für saubere Kanten ===
    kernel = np.ones((5, 5), np.uint8)
    edges_eroded = cv2.erode(edges, kernel, iterations=1)
    edges_clean = cv2.dilate(edges_eroded, kernel, iterations=1)

    # Hybrid-Maske bilden
    final_mask = ki_mask.copy()
    final_mask = cv2.bitwise_or(final_mask, edges_clean)

    out = np.zeros_like(frame)
    out[final_mask > 0] = [255,255,255]

    cv2.imshow('Hybrid Silhouette', out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
