import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Einmalige Hintergrundkalibrierung (Median von 30 Frames für Robustheit)
print("Kalibrierung: Bitte den sichtbaren Bereich verlassen...")
bg_buffer = []
for i in range(30):
    ret, bg = cap.read()
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg_buffer.append(cv2.GaussianBlur(bg_gray, (11, 11), 0))

    # Fortschritt anzeigen
    cv2.putText(bg, f'Kalibrierung: {i + 1}/30', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Kalibrierung', bg)
    cv2.waitKey(10)

background_frame = np.median(bg_buffer, axis=0).astype(np.uint8)
print("Kalibrierung abgeschlossen!")


def mask_to_ascii(mask, num_cols=80):
    h, w = mask.shape
    aspect_ratio = h / w
    num_rows = int(aspect_ratio * num_cols * 0.5)
    small = cv2.resize(mask, (num_cols, num_rows), interpolation=cv2.INTER_NEAREST)
    # Binarisiere, Person ist 1, Rest 0
    lines = ["".join("1" if px > 127 else "0" for px in row) for row in small]
    return lines


def ascii_image_to_frame(lines, target_width=640, target_height=480):
    """
    Erstellt ASCII-Bild in genauer Zielgröße (640x480)
    """
    # Weißes Bild erstellen
    img = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    num_rows = len(lines)
    num_cols = len(lines[0]) if lines else 0

    if num_rows == 0 or num_cols == 0:
        return img

    # Berechne Skalierung um das Bild zu füllen
    scale_x = target_width / num_cols
    scale_y = target_height / num_rows
    scale = min(scale_x, scale_y)

    # Zentriere das ASCII-Bild
    total_width = int(num_cols * scale)
    total_height = int(num_rows * scale)
    offset_x = (target_width - total_width) // 2
    offset_y = (target_height - total_height) // 2

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == "1":  # Nur Person zeigen
                px = int(x * scale) + offset_x
                py = int(y * scale) + offset_y
                color = (0, 0, 0)  # Schwarz für Person
                cv2.putText(img, char, (px, py + int(scale)),
                            cv2.FONT_HERSHEY_PLAIN, scale / 10, color, max(1, int(scale / 8)))

    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    diff = cv2.absdiff(gray, background_frame)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologische Filterung: Silhouette füllen & das Ganze spiegeln
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((6, 6), np.uint8), iterations=1)
    mask = cv2.flip(mask, 1)

    # ASCII-Rendering in 640x480
    ascii_lines = mask_to_ascii(mask, num_cols=80)
    ascii_frame = ascii_image_to_frame(ascii_lines, target_width=640, target_height=480)

    cv2.imshow("Binary ASCII Person Mirror", ascii_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cap.release()
cv2.destroyAllWindows()