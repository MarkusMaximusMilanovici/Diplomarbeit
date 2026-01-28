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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Helligkeit/Kontrast grob anheben
    alpha_bright = 1.3  # Kontrast
    beta_bright = 20  # Helligkeit
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)

    # Für Mediapipe dieses hellere Bild nehmen
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    hand_res = hands.process(rgb)

    fgbg.apply(gray, learningRate=0.5)

    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Kalibrierung', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

print("Kalibrierung abgeschlossen! Du kannst jetzt ins Bild.")

# ====== zeitliche Glättung vorbereiten ======
prev_mask = None
prev_hand_mask = None
alpha = 0.3
alpha_hand = 0.5

# ====== Hauptloop ======
while True:
    ret, frame = get_frame(cam)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Person Segmentierung (RGB erwartet)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.35).astype(np.uint8) * 255

    # ===== Hände erkennen und separate Hand-Maske erstellen =====
    hand_res = hands.process(rgb)
    hand_mask = np.zeros_like(ki_mask)

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape
        for handLms in hand_res.multi_hand_landmarks:
            # Sammle alle Hand-Punkte
            hand_points = []
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])

            # === STRATEGIE 1: Nur Kreise (natürlicher) ===
            # Zeichne mittelgroße Kreise um jeden Landmark
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), 10, 255, -1)

            # === STRATEGIE 2: Verbinde benachbarte Punkte mit Linien ===
            # MediaPipe Hand hat 21 Landmarks in bestimmter Reihenfolge
            # Verbindungen zwischen benachbarten Fingerknochen
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Daumen
                (0, 5), (5, 6), (6, 7), (7, 8),  # Zeigefinger
                (5, 9), (9, 10), (10, 11), (11, 12),  # Mittelfinger
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ringfinger
                (13, 17), (17, 18), (18, 19), (19, 20),  # Kleiner Finger
                (0, 17)  # Handgelenk zu Handfläche
            ]

            # Zeichne dicke Linien zwischen verbundenen Punkten
            for connection in connections:
                if connection[0] < len(hand_points) and connection[1] < len(hand_points):
                    pt1 = tuple(hand_points[connection[0]])
                    pt2 = tuple(hand_points[connection[1]])
                    cv2.line(hand_mask, pt1, pt2, 255, thickness=18)

    # ===== Zeitliche Glättung NUR für die Hand-Maske =====
    if prev_hand_mask is None:
        prev_hand_mask = hand_mask.copy()
    else:
        hand_blended = cv2.addWeighted(prev_hand_mask, alpha_hand, hand_mask, 1 - alpha_hand, 0)
        _, hand_blended = cv2.threshold(hand_blended, 127, 255, cv2.THRESH_BINARY)
        prev_hand_mask = hand_blended
        hand_mask = hand_blended

    # Dilatiere und Blur für weichere Kanten
    if np.any(hand_mask > 0):
        kernel_hand = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel_hand, iterations=1)
        # Blur macht die Kanten weicher/natürlicher
        hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)
        _, hand_mask = cv2.threshold(hand_mask, 127, 255, cv2.THRESH_BINARY)

    # Hand-Maske mit KI-Maske kombinieren
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)

    # Bewegungsmaske (optional)
    fgmask = fgbg.apply(gray, learningRate=0)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    edges = cv2.Canny(fgmask, 60, 130)
    edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges_clean = cv2.morphologyEx(edges_open, cv2.MORPH_CLOSE, kernel)

    # Hybrid-Maske bilden
    final_mask = ki_mask.copy()
    # final_mask = cv2.bitwise_or(final_mask, edges_clean)

    kernel_small = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel_small, iterations=1)

    # --- Konturen der Person holen und füllen ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(final_mask)

    if len(contours) > 0:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(mask_filled, [contours_sorted[0]], -1, 255, thickness=cv2.FILLED)

        min_area = 300
        for contour in contours_sorted[1:]:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    final_mask = mask_filled

    # ===== zeitliche Glättung der gesamten Maske =====
    if prev_mask is None:
        prev_mask = final_mask.copy()
    else:
        blended = cv2.addWeighted(prev_mask, alpha, final_mask, 1 - alpha, 0)
        _, blended = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)
        prev_mask = blended
        final_mask = blended

    # Ausgabebild in voller Auflösung (reine Silhouette)
    out_full = np.zeros_like(frame)
    out_full[final_mask > 0] = [255, 255, 255]

    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)

    # out_small = cv2.resize(out_full, (32, 32), interpolation=cv2.INTER_AREA)
    # cv2.imshow('Hybrid Silhouette (32x32)', out_small)
    # ImagetoMatrix.drawImage(out_small)

    if cv2.waitKey(1) & 0xFF == 27:
        break

release_camera(cam)
cv2.destroyAllWindows()