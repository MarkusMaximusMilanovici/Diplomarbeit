import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.medianBlur(fgmask, 7)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(fgmask, 60, 130)
    edges_dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # Hybrid-Maske bilden
    final_mask = ki_mask.copy()
    final_mask = cv2.bitwise_or(final_mask, edges_dil)

    # Endbild: Silhouette weiß, Rest schwarz
    out = np.zeros_like(frame)
    out[final_mask > 0] = [255,255,255]

    cv2.imshow('Hybrid Silhouette', out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
