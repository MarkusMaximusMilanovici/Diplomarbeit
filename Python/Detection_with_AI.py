import cv2
import mediapipe as mp
import numpy as np

# Tipp: Gamma-Korrektur (hilft bei NoIR/farbverfremdeten Modulen)
gamma = 1.3
look_up_table = np.array([
    ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
]).astype("uint8")

cap = cv2.VideoCapture(0)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- 1. Gamma-Korrektur anwenden ----
    frame_gamma = cv2.LUT(frame, look_up_table)

    # ---- 2. Spiegelung (wie Selfie) & RGB-Konvertierung ----
    frame_rgb = cv2.cvtColor(cv2.flip(frame_gamma, 1), cv2.COLOR_BGR2RGB)

    # ---- 3. KI-Personenerkennung (MediaPipe) ----
    results = segmenter.process(frame_rgb)
    mask = results.segmentation_mask  # Wertebereich [0,1], 1 = sicher Person

    # ---- 4. Schwellenwert für feine Kontur (Finger!): Nicht Standard (0.5), sondern etwas niedriger ----
    # Bei zu niedrigem Wert (z.B. <0.2) können "Ghost"-Effekte entstehen!
    person_mask = (mask > 0.35).astype(np.uint8) * 255

    # ---- 5. Morphologie mit kleinen Kernels – NUR minimal, damit Finger erhalten bleiben ----
    # Zu große Kernel verschmieren! (max (3,3) – optional auch verzichten!)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ---- 6. Blur nur sanft, hilft gegen Rauschen OHNE Finger zu verwischen ----
    # Optional, kann ganz weggelassen werden!
    # person_mask = cv2.GaussianBlur(person_mask, (3,3), 0)

    # ---- 7. Immer nur aktuellen Frame anzeigen, KEINE Überlagerung! ----
    cv2.imshow("Optimierte Silhouette: Präzise Finger, Kein Ghosting", person_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
