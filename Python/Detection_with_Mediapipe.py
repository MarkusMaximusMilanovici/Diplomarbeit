import cv2
import numpy as np
import mediapipe as mp
import ImagetoMatrix

# ============================================================
# Kamera-Auswahl: Raspberry Pi (PiCamera2) ODER Laptop (cv2.VideoCapture)
# ============================================================
USE_PI = False
try:
    from picamera2 import Picamera2
    USE_PI = True
    print("PiCamera2 gefunden – benutze Raspberry-Kamera.")
except ImportError:
    USE_PI = False
    print("PiCamera2 nicht gefunden – benutze cv2.VideoCapture.")

def init_camera():
    if USE_PI:
        picam2 = Picamera2()
        # Auflösung anpassen, falls nötig
        config = picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        return picam2
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

def get_frame(cam):
    if USE_PI:
        frame = cam.capture_array()
        return True, frame
    else:
        return cam.read()

def release_camera(cam):
    if USE_PI:
        cam.stop()
    else:
        cam.release()

# ============================================================
# MediaPipe + Background-Subtractor
# ============================================================

mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

cam = init_camera()

# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100
print("Kalibrierung: Bitte den sichtbaren Bereich verlassen. Die Kamera lernt den Hintergrund.")

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    if not ret:
        break

    # KEIN Rotieren
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg.apply(gray, learningRate=0.5)

    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Kalibrierung', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

print("Kalibrierung abgeschlossen! Du kannst jetzt ins Bild.")

# ====== Hauptloop ======
while True:
    ret, frame = get_frame(cam)
    if not ret:
        break

    # KEIN Rotieren
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Person Segmentierung (RGB erwartet)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255

    # ===== Hände erkennen und in die KI-Maske malen =====
    hand_res = hands.process(rgb)

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape
        for handLms in hand_res.multi_hand_landmarks:
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                # kleiner weißer Punkt an jeder Landmark-Position
                cv2.circle(ki_mask, (cx, cy), 2, 255, -1)

    # Bewegungsmaske (fgmask) und Morphologische Reinigung VOR Canny
    fgmask = fgbg.apply(gray, learningRate=0)
    fgmask = cv2.medianBlur(fgmask, 7)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    # Vorverarbeitung: Erode und Dilate
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Canny-Kantenerkennung auf dem vorverarbeiteten fgmask
    edges = cv2.Canny(fgmask, 60, 130)

    # Opening und Closing auf Kantenbild
    edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges_clean = cv2.morphologyEx(edges_open, cv2.MORPH_CLOSE, kernel)

    # Hybrid-Maske bilden (KNN optional)
    final_mask = ki_mask.copy()
    # final_mask = cv2.bitwise_or(final_mask, edges_clean)  # bei Bedarf wieder aktivieren

    kernel_small = np.ones((3, 3), np.uint8)

    # 1) Kleine Löcher schließen
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 2) Silhouette leicht verdicken (wenn zu fett: Zeile auskommentieren)
    final_mask = cv2.dilate(final_mask, kernel_small, iterations=1)

    # --- Konturen der Person holen und füllen ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_filled = np.zeros_like(final_mask)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [largest], -1, 255, thickness=cv2.FILLED)

    final_mask = mask_filled

    # Ausgabebild in voller Auflösung (reine Silhouette)
    out_full = np.zeros_like(frame)
    out_full[final_mask > 0] = [255, 255, 255]

    # Anzeige
    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)

    # Downscale für Matrix (falls benötigt)
    # out_small = cv2.resize(out_full, (32, 32), interpolation=cv2.INTER_AREA)
    # cv2.imshow('Hybrid Silhouette (32x32)', out_small)
    # ImagetoMatrix.drawImage(out_small)

    if cv2.waitKey(1) & 0xFF == 27:
        break

release_camera(cam)
cv2.destroyAllWindows()
