import cv2
import numpy as np
import mediapipe as mp

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
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

cam = init_camera()

# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100
print("Kalibrierung: Bitte den sichtbaren Bereich verlassen. Die Kamera lernt den Hintergrund.")

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    if not ret:
        break

    # Kamera-Bild direkt drehen, aber noch NICHT downscalen
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

    # 1) Frame drehen (90° im Uhrzeigersinn), volle Auflösung behalten
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # --- ab hier alles in voller (gedrehter) Auflösung berechnen ---

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    median = cv2.medianBlur(blurred, 9)

    # MediaPipe Person Segmentierung (RGB erwartet)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Bewegungsmaske (fgmask) und Morphologische Reinigung VOR Canny
    fgmask = fgbg.apply(gray, learningRate=0)
    fgmask = cv2.medianBlur(fgmask, 7)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)

    # Vorverarbeitung: Erode und Dilate
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Canny-Kantenerkennung auf dem vorverarbeiteten fgmask
    edges = cv2.Canny(fgmask, 60, 130)

    # Opening und Closing auf Kantenbild
    edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges_clean = cv2.morphologyEx(edges_open, cv2.MORPH_CLOSE, kernel)

    # Hybrid-Maske bilden
    final_mask = ki_mask.copy()
    final_mask = cv2.bitwise_or(final_mask, edges_clean)

    # Ausgabebild in voller Auflösung
    out_full = np.zeros_like(frame)
    out_full[final_mask > 0] = [255, 255, 255]

    # === JETZT erst Downscalen auf 72x128 (Breite x Höhe, 9:16) ===
    out_small = cv2.resize(out_full, (32, 32), interpolation=cv2.INTER_AREA)

    cv2.imshow('Hybrid Silhouette (32x32)', out_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

release_camera(cam)
cv2.destroyAllWindows()
