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

    alpha_bright = 1.3
    beta_bright = 20
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)

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
alpha = 0.25  # Weniger Glättung = weniger Rauschen durch Verschmierung
alpha_hand = 0.35  # Weniger Glättung für schnellere Hand-Reaktion

# ====== Hauptloop ======
while True:
    ret, frame = get_frame(cam)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Person Segmentierung
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255  # Höherer Threshold = sauberere Maske

    # ===== Hände erkennen =====
    hand_res = hands.process(rgb)
    hand_mask = np.zeros_like(ki_mask)

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape
        for handLms in hand_res.multi_hand_landmarks:
            # Sammle alle 21 Hand-Punkte
            hand_points = []
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])

            # === Erstelle realistische Hand-Form ===

            # 1. Zeichne moderate Linien zwischen Fingerknochen
            connections = [
                # Daumen
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Zeigefinger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Mittelfinger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ringfinger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Kleiner Finger
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Handfläche
                (5, 9), (9, 13), (13, 17)
            ]

            # Moderate Liniendicke - nicht zu dick!
            for connection in connections:
                pt1 = tuple(hand_points[connection[0]])
                pt2 = tuple(hand_points[connection[1]])
                cv2.line(hand_mask, pt1, pt2, 255, thickness=9)  # Reduziert von 12 auf 9

            # 2. Moderate Kreise um Gelenke
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), 6, 255, -1)  # Erhöht von 4 auf 6

    # ===== Hand-Maske verfeinern (WENIGER aggressiv) =====
    if np.any(hand_mask > 0):
        # Moderate Dilatation für zusammenhängende Hand
        kernel_medium = np.ones((5, 5), np.uint8)  # Reduziert von (7,7) auf (5,5)
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)  # Nur 1 statt 2

        # Moderater Blur für organische Form
        hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)  # Reduziert von (15,15) auf (9,9)
        _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)  # Höherer Threshold

        # Leichte Erosion für realistische Größe
        kernel_small = np.ones((3, 3), np.uint8)  # Reduziert von (5,5) auf (3,3)
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)

    # ===== Zeitliche Glättung für Hand-Maske =====
    if prev_hand_mask is None:
        prev_hand_mask = hand_mask.copy()
    else:
        hand_blended = cv2.addWeighted(prev_hand_mask, alpha_hand, hand_mask, 1 - alpha_hand, 0)
        _, hand_blended = cv2.threshold(hand_blended, 140, 255, cv2.THRESH_BINARY)  # Höherer Threshold
        prev_hand_mask = hand_blended
        hand_mask = hand_blended

    # Hand-Maske mit KI-Maske kombinieren
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)

    # === Leichte morphologische Operationen gegen Rauschen ===
    kernel_denoise = np.ones((3, 3), np.uint8)  # Reduziert von (5,5) auf (3,3)
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)  # Nur 1 statt 2
    # KEIN MORPH_OPEN - das erzeugt Rauschen!

    # Bewegungsmaske (optional, auskommentiert)
    fgmask = fgbg.apply(gray, learningRate=0)

    # Hybrid-Maske
    final_mask = ki_mask.copy()

    # === NUR minimale Glättung gegen Rauschen ===
    # Bilateral Filter behält Kanten besser als Gaussian
    final_mask = cv2.bilateralFilter(final_mask, 5, 50, 50)
    _, final_mask = cv2.threshold(final_mask, 140, 255, cv2.THRESH_BINARY)

    # --- Konturen füllen (aber NICHT verbinden) ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(final_mask)

    if len(contours) > 0:
        # Fülle ALLE Konturen über Mindestgröße - NICHT nur die größte!
        min_area = 200
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    final_mask = mask_filled

    # ===== Zeitliche Glättung der gesamten Maske =====
    if prev_mask is None:
        prev_mask = final_mask.copy()
    else:
        blended = cv2.addWeighted(prev_mask, alpha, final_mask, 1 - alpha, 0)
        _, blended = cv2.threshold(blended, 150, 255, cv2.THRESH_BINARY)
        prev_mask = blended
        final_mask = blended

    # === Finale sanfte Kantenglättung ===
    # Sehr leichter Blur nur für Anti-Aliasing
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    _, final_mask = cv2.threshold(final_mask, 220, 255, cv2.THRESH_BINARY)  # Hoher Threshold für klare Kanten

    # Ausgabebild
    out_full = np.zeros_like(frame)
    out_full[final_mask > 0] = [255, 255, 255]

    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)

    out_small = cv2.resize(out_full, (32, 40), interpolation=cv2.INTER_AREA)
    # cv2.imshow('Hybrid Silhouette (32x32)', out_small)
    ImagetoMatrix.drawImage(out_small)

    if cv2.waitKey(1) & 0xFF == 27:
        break

release_camera(cam)
cv2.destroyAllWindows()