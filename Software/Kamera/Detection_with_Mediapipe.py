import cv2
import numpy as np
import mediapipe as mp
import Software.LED_Matrix.ImagetoMatrix as ImagetoMatrix

# === NEU: Anzeige prüfen ===
import os
SHOW_GUI = "DISPLAY" in os.environ

# ============================================================
# Kamera-Auswahl
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
        return True, cam.capture_array()
    else:
        return cam.read()


def release_camera(cam):
    if USE_PI:
        cam.stop()
    else:
        cam.release()


# ============================================================
# MediaPipe
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

fgbg = cv2.createBackgroundSubtractorKNN(
    history=150, dist2Threshold=400, detectShadows=False
)

cam = init_camera()

# === Kalibrierung ===
calibration_frames = 100
print("Kalibrierung läuft...")

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    segmenter.process(rgb)
    hands.process(rgb)

    fgbg.apply(gray, learningRate=0.5)

    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === GUI nur wenn möglich ===
    if SHOW_GUI:
        cv2.imshow('Kalibrierung', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

print("Kalibrierung abgeschlossen!")

prev_mask = None

# === Hauptloop ===
while True:
    ret, frame = get_frame(cam)
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)

    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255

    out_full = np.zeros_like(frame)
    out_full[ki_mask > 0] = [255, 255, 255]

    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)

    # === GUI nur wenn möglich ===
    if SHOW_GUI:
        cv2.imshow('Preview', preview)

    out_small = cv2.resize(out_full, (32, 48), interpolation=cv2.INTER_AREA)

    # LED Ausgabe
    ImagetoMatrix.drawImage(out_small)

    # === ESC nur wenn GUI ===
    if SHOW_GUI:
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Cleanup
release_camera(cam)

if SHOW_GUI:
    cv2.destroyAllWindows()