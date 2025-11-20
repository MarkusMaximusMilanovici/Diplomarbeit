import cv2
import mediapipe as mp
import numpy as np

gamma = 1.3
look_up_table = np.array([
    ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
]).astype("uint8")

cap = cv2.VideoCapture(0)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

bg_frames = []
for i in range(30):
    ret, bg = cap.read()
    if not ret:
        break
    bg = cv2.LUT(bg, look_up_table)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg_frames.append(bg_gray)
background = np.median(bg_frames, axis=0).astype(np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gamma = cv2.LUT(frame, look_up_table)
    frame_mirror = cv2.flip(frame_gamma, 1)

    frame_rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
    results = segmenter.process(frame_rgb)
    mask = results.segmentation_mask
    mp_mask = (mask > 0.35).astype(np.uint8) * 255

    mp_mask = cv2.morphologyEx(mp_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    mp_mask = cv2.morphologyEx(mp_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    gray = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)
    _, classic_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    classic_mask = cv2.morphologyEx(classic_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    classic_mask = cv2.morphologyEx(classic_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    combo_mask = mp_mask.copy()
    combo_mask[(classic_mask == 255) & (mp_mask == 0)] = 128  # Halbwert (Grau): klassische Fingerergänzung

    vis = np.zeros_like(mp_mask)
    vis[combo_mask == 255] = 255    # Person
    vis[combo_mask == 128] = 130    # Finger/Bewegungsdetails (mittleres Grau)

    # --------- Binarisierung ("clampen"): Nur helle Werte bleiben, Rest wird schwarz ---------
    _, vis_binary = cv2.threshold(vis, 200, 255, cv2.THRESH_BINARY)

    cv2.imshow("MediaPipe + Classic - Binär Silhouette", vis_binary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
