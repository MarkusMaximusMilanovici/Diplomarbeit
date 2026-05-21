import cv2                        # OpenCV für Bildverarbeitung und mathematische Matrix-Operationen
import numpy as np                # NumPy für schnelle numerische Berechnungen von Arrays (Bilder sind Arrays)
import mediapipe as mp            # Google MediaPipe für Machine-Learning basierte Objekterkennung (Körper & Hände)
import Software.LED_Matrix.ImagetoMatrix as ImagetoMatrix  # Eigenes Modul zur Hardwaresteuerung der LED-Matrix

# ============================================================
# Kamera-Auswahl: Raspberry Pi (PiCamera2) ODER Laptop (cv2.VideoCapture)
# ============================================================
USE_PI = False
try:
    # Versuche die spezifische Raspberry Pi Kamera Bibliothek zu laden
    from picamera2 import Picamera2

    USE_PI = True
    print("PiCamera2 gefunden – benutze Raspberry-Kamera.")
except ImportError:
    # Fallback für die Entwicklung auf normalen Laptops/PCs
    USE_PI = False
    print("PiCamera2 nicht gefunden – benutze cv2.VideoCapture.")


def init_camera():
    """ Initialisiert die Kamera je nach erkannter Hardware (Pi oder Laptop). """
    if USE_PI:
        picam2 = Picamera2()
        # Auflösung auf 640x480 (VGA) festlegen, reicht für die 32x48 Matrix völlig aus.
        config = picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        return picam2
    else:
        # Standard OpenCV Methode für allgemeine Webcams (Index 0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap


def get_frame(cam):
    """ Liest das aktuelle Bild (Frame) von der Kamera ein. """
    if USE_PI:
        frame = cam.capture_array()
        return True, frame
    else:
        return cam.read()


def release_camera(cam):
    """ Gibt die Hardwareressourcen der Kamera beim Beenden ordnungsgemäß frei. """
    if USE_PI:
        cam.stop()
    else:
        cam.release()


# ============================================================
# MediaPipe + Background-Subtractor initialisieren
# ============================================================

# Modul für die Körper-Segmentierung laden
mp_selfie = mp.solutions.selfie_segmentation
# model_selection=1 verwendet ein auf Geschwindigkeit optimiertes 'landscape'-Modell
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

# Spezielles Modell nur für die exakte Hand- und Fingererkennung
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # False = Videomodus, nutzt das Bewegungstracking vergangener Frames
    max_num_hands=2,                # Limitiert auf maximal 2 Hände für bessere Performance
    min_detection_confidence=0.5,   # Hände werden ab 50% Sicherheit als gefunden gewertet
    min_tracking_confidence=0.6     # Tracker behält die Hand bei ab 60% Sicherheit
)

# Klassischer BackgroundSubtractor als Fallback oder Zusatz (K-Nearest-Neighbors)
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

# Kamera letztendlich starten
cam = init_camera()

# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100
print("Kalibrierung: Bitte den sichtbaren Bereich verlassen. Die Kamera lernt den Hintergrund.")

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    if not ret:
        break

    # Graustufenbild wird für den klassischen Subtractor benötigt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Künstliche Bildaufhellung zur Kompensation von schlechtem Raumlicht
    alpha_bright = 1.3
    beta_bright = 20
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)

    # Umwandlung zu RGB für MediaPipe
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)

    # Modelle verarbeiten
    res = segmenter.process(rgb)
    hand_res = hands.process(rgb)

    # KNN trainieren
    fgbg.apply(gray, learningRate=0.5)

    # Text anzeigen
    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Kalibrierung', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

print("Kalibrierung abgeschlossen! Du kannst jetzt ins Bild.")

# ====== zeitliche Glättung vorbereiten ======
prev_mask = None
prev_hand_mask = None
alpha = 0.25
alpha_hand = 0.35

# ====== Hauptloop ======
while True:
    ret, frame = get_frame(cam)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Person Segmentierung
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)

    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255

    # ===== Hände erkennen =====
    hand_res = hands.process(rgb)
    hand_mask = np.zeros_like(ki_mask)

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape

        for handLms in hand_res.multi_hand_landmarks:
            hand_points = []

            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])

            # Handfläche füllen
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_pts = np.array([hand_points[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(hand_mask, [palm_pts], 255)

            # Fingerverbindungen
            connections = [
                (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),
                (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8),
                (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20)
            ]

            finger_thickness = 18

            for connection in connections:
                pt1 = tuple(hand_points[connection[0]])
                pt2 = tuple(hand_points[connection[1]])
                cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)

            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)

    # ===== Hand-Maske verfeinern =====
    if np.any(hand_mask > 0):

        kernel_medium = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)

        hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)
        _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)

        kernel_small = np.ones((3, 3), np.uint8)
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)

    # ===== Zeitliche Glättung Hand =====
    if prev_hand_mask is None:
        prev_hand_mask = hand_mask.copy()
    else:
        hand_blended = cv2.addWeighted(prev_hand_mask, alpha_hand, hand_mask, 1 - alpha_hand, 0)
        _, hand_blended = cv2.threshold(hand_blended, 140, 255, cv2.THRESH_BINARY)

        prev_hand_mask = hand_blended
        hand_mask = hand_blended

    # Hände + Körper kombinieren
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)

    # Morphologische Bereinigung
    kernel_denoise = np.ones((3, 3), np.uint8)
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)

    # Bewegungsmaske (KNN) erzeugen und aufräumen
    fgmask = fgbg.apply(gray, learningRate=0)

    # Gewichtete Kombination: KI-Maske als Basis, KNN ergänzt nur dort
    # wo auch tatsächlich Bewegung erkannt wird (weniger Rauschen als OR)
    ki_float = ki_mask.astype(np.float32) / 255.0
    knn_float = fgmask.astype(np.float32) / 255.0

    # KI bleibt voll erhalten (Gewicht 1.0), KNN wird nur mit 0.3 beigemischt
    # Dadurch werden KNN-only Bereiche nur übernommen wenn sie stark genug sind
    combined = np.clip(ki_float + knn_float * 0.2, 0, 1)
    ki_mask = (combined * 255).astype(np.uint8)
    _, ki_mask = cv2.threshold(ki_mask, 100, 255, cv2.THRESH_BINARY)

    # Bilateral Filter
    final_mask = cv2.bilateralFilter(ki_mask, 5, 50, 50)
    _, final_mask = cv2.threshold(final_mask, 140, 255, cv2.THRESH_BINARY)

    # Konturen finden
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_filled = np.zeros_like(final_mask)

    if len(contours) > 0:
        min_area = 200

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    final_mask = mask_filled

    # ===== Zeitliche Glättung =====
    if prev_mask is None:
        prev_mask = final_mask.copy()
    else:
        blended = cv2.addWeighted(prev_mask, alpha, final_mask, 1 - alpha, 0)
        _, blended = cv2.threshold(blended, 150, 255, cv2.THRESH_BINARY)

        prev_mask = blended
        final_mask = blended

    # Finale Kantenglättung
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    _, final_mask = cv2.threshold(final_mask, 220, 255, cv2.THRESH_BINARY)

    # Schwarzes Bild erzeugen
    out_full = np.zeros_like(frame)

    # Weiße Silhouette einsetzen
    out_full[final_mask > 0] = [255, 255, 255]

    # Vorschau
    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)

    # ===== Aspect Ratio Crop =====
    h, w = out_full.shape[:2]
    aspect = 32 / 48

    if w / h > aspect:
        crop_w = int(h * aspect)
        crop_h = h
    else:
        crop_w = w
        crop_h = int(w / aspect)

    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))

    # Downsampling auf LED Matrix
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)

    # Kleine Vorschau
    cv2.imshow('Hybrid Silhouette (32x32)', out_small)

    # Matrix Ausgabe
    ImagetoMatrix.drawImage(out_small)

    # ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
release_camera(cam)
cv2.destroyAllWindows()