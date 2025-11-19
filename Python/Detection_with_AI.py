import cv2
import mediapipe as mp
import numpy as np

# Kamera-Setup: plattformunabhängig, passt für Raspberry/Windows/Linux
cap = cv2.VideoCapture(0)

mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

def mask_to_ascii(mask, num_cols=80):
    h, w = mask.shape
    aspect_ratio = h / w
    num_rows = int(aspect_ratio * num_cols * 0.5)
    small = cv2.resize(mask, (num_cols, num_rows), interpolation=cv2.INTER_NEAREST)
    # Binarisiere: "1" für Person, "0" für Hintergrund
    lines = ["".join("1" if px > 127 else "0" for px in row) for row in small]
    return lines

def ascii_image_to_frame(lines, scale=10):
    h = len(lines) * scale
    w = len(lines[0]) * scale
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            color = (0, 0, 0) if char == "1" else (255,255,255)
            cv2.putText(img, char, (x*scale, y*scale+scale), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe erwartet RGB, von BGR konvertieren und Spiegeln (Mirror-Style)
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = segmenter.process(frame_rgb)
    mask = results.segmentation_mask

    # KI-Maske: alles > 0.45 ist Person
    person_mask = (mask > 0.45).astype(np.uint8) * 255

    # Morphologie für schöne Silhouette (optional), kann feiner oder gröber gestellt werden
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, np.ones((13,13),np.uint8))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    person_mask = cv2.GaussianBlur(person_mask, (7,7), 0)
    _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)

    # ASCII-Umwandlung (NUR "1"/"0")
    ascii_lines = mask_to_ascii(person_mask, num_cols=80)
    ascii_frame = ascii_image_to_frame(ascii_lines, scale=10)

    cv2.imshow("KI-ASCII Mirror (Person=1, BG=0)", ascii_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
