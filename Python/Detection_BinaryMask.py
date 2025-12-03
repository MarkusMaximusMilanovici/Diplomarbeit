import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Einmalige Hintergrundkalibrierung (Median von 30 Frames für Robustheit)
bg_buffer = []
for i in range(30):
    ret, bg = cap.read()
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg_buffer.append(cv2.GaussianBlur(bg_gray, (11,11), 0))
background_frame = np.median(bg_buffer, axis=0).astype(np.uint8)

def mask_to_ascii(mask, num_cols=80):
    h, w = mask.shape
    aspect_ratio = h / w
    num_rows = int(aspect_ratio * num_cols * 0.5)
    small = cv2.resize(mask, (num_cols, num_rows), interpolation=cv2.INTER_NEAREST)
    # Binarisiere, Person ist 1, Rest 0
    lines = ["".join("1" if px > 127 else "0" for px in row) for row in small]
    return lines

def ascii_image_to_frame(lines, scale=12):
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    diff = cv2.absdiff(gray, background_frame)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologische Filterung: Silhouette füllen & das Ganze spiegeln
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13,13),np.uint8))
    mask = cv2.dilate(mask, np.ones((9,9),np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((6,6),np.uint8), iterations=1)
    mask = cv2.flip(mask, 1)

    # ASCII-Rendering (NUR "1" für Person!)
    ascii_lines = mask_to_ascii(mask, num_cols=80)
    ascii_frame = ascii_image_to_frame(ascii_lines, scale=10)

    cv2.imshow("Binary ASCII Person Mirror", ascii_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
