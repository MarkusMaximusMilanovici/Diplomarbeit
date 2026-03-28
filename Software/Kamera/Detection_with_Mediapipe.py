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
    alpha_bright = 1.3  # Kontrast leicht erhöhen (30%)
    beta_bright = 20    # Generelle Helligkeit um 20 Einheiten anheben
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)

    # Umwandlung zu RGB für MediaPipe, da MediaPipe kein BGR unterstützt
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
    
    # Modelle mit dem Bild "füttern" beziehungsweise verarbeiten
    res = segmenter.process(rgb)
    hand_res = hands.process(rgb)

    # Das KNN-Modell mit hoher Lernrate an die leere Umgebung gewöhnen
    fgbg.apply(gray, learningRate=0.5)

    # Text auf dem Vorschaubild der Kalibrierungsphase anzeigen
    cv2.putText(frame, f'Kalibrierung: {i + 1}/{calibration_frames}', (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Kalibrierung', frame)
    
    # Abbrechen, wenn der Benutzer die ESC-Taste drückt (ASCII-Code 27)
    if cv2.waitKey(10) & 0xFF == 27:
        break

print("Kalibrierung abgeschlossen! Du kannst jetzt ins Bild.")

# ====== zeitliche Glättung vorbereiten ======
prev_mask = None
prev_hand_mask = None
alpha = 0.25         # Für den gesamten Körper: Weniger neues Bild, mehr altes Bild (Glättung)
alpha_hand = 0.35    # Für die Hände: Höherer Wert für eine reaktionsschnellere Hand-Interaktion

# ====== Hauptloop (Die Verarbeitung in Echtzeit) ======
while True:
    # Aktuelles Videobild im Array ablegen
    ret, frame = get_frame(cam)
    if not ret:
        break

    # Graustufenbild für Bewegungsfallbacks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Person Segmentierung (Benötigt RGB Format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    
    # Thresholding: Ab 40% Sicherheit (0.4) dass ein Pixel zum Benutzer gehört, wird er auf 'Weiß' (255) gesetzt.
    # Höherer Threshold (z.B. >0.5) erzeugt sauberere Ränder, schneidet aber Haare/Kleidung öfter ab
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255

    # ===== Hände erkennen =====
    hand_res = hands.process(rgb)      # Zweiter Schritt des KI Modells - fokussiert sich jetzt nur auf Hände
    hand_mask = np.zeros_like(ki_mask) # Leere schwarze Leinwand im Format (480, 640)

    # Prüfe ob überhaupt eine Hand oder mehrere Hände im Bild vorhanden sind
    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape  # Dimensionen des Bildes (h=480, w=640)
        for handLms in hand_res.multi_hand_landmarks:
            # Sammle alle 21 geometrischen Hand-Punkte und skaliere sie aus dem %-Bereich (0.00-1.00) in echte Pixelkoordinaten
            hand_points = []
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])

            # === Erstelle realistische massive Hand-Form ===

            # 1. Fülle die Handfläche als Polygon, um Löcher (Skelett-Look) zu vermeiden
            palm_indices = [0, 1, 5, 9, 13, 17] # Das sind die Landmark-Indizes der inneren Handfläche (ohne Finger)
            palm_pts = np.array([hand_points[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(hand_mask, [palm_pts], 255) # Weiß ausmalen

            # 2. Verbindungen der Finger und Handkanten anhand der anatomischen Knoten
            connections = [
                # Handfläche (Umrandung für weiche Kanten)
                (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),
                # Daumen
                (1, 2), (2, 3), (3, 4),
                # Zeigefinger
                (5, 6), (6, 7), (7, 8),
                # Mittelfinger
                (9, 10), (10, 11), (11, 12),
                # Ringfinger
                (13, 14), (14, 15), (15, 16),
                # Kleiner Finger
                (17, 18), (18, 19), (19, 20)
            ]

            # Etwas dickere Linien für Finger, damit sie auf der LED Matrix nachher klarer wirken
            finger_thickness = 18 
            for connection in connections:
                pt1 = tuple(hand_points[connection[0]])
                pt2 = tuple(hand_points[connection[1]])
                cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)

            # 3. Kreise um alle Gelenke für echte, bündige Abschlüsse (verhindert eckige Knickstellen)
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)

    # ===== Hand-Maske verfeinern (WENIGER aggressiv) =====
    if np.any(hand_mask > 0): # Überspringen, falls keine Hände auf dem Bild sind, spart Rechenzeit

        # Moderate Dilatation (Aufblähen): Schließt feine Lücken zwischen den Fingern
        kernel_medium = np.ones((5, 5), np.uint8) 
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)

        # Moderater Blur (Unschärfefilter): Rundet die geraden Polygon-Kanten extrem stark ab, dass sie organisch wirken
        hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)  
        _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)  
        # Durch den hohen Threshold wird der graue Unschärfebereich zu knackigen, weißen Kanten konvertiert.

        # Leichte Erosion (Schälen): Entfernt überschüssiges "Restfleisch", das durch die anfängliche Dilatation hinzugekommen ist
        kernel_small = np.ones((3, 3), np.uint8)  
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)

    # ===== Zeitliche Glättung für Hand-Maske =====
    if prev_hand_mask is None:
        prev_hand_mask = hand_mask.copy()
    else:
        # Frameblending: Die addWeighted Methode verknüpft x% der Maske des letzten Frames mit y% des aktuellen Frames
        hand_blended = cv2.addWeighted(prev_hand_mask, alpha_hand, hand_mask, 1 - alpha_hand, 0)
        _, hand_blended = cv2.threshold(hand_blended, 140, 255, cv2.THRESH_BINARY) 
        prev_hand_mask = hand_blended
        hand_mask = hand_blended

    # Danach wird die generierte KI Körper-Maske mit der Hand-Maske verschmolzen (Bitwise OR)
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)

    # === Leichte morphologische Operationen gegen Rauschen am Rande der Maske ===
    kernel_denoise = np.ones((3, 3), np.uint8)
    # MORPH_CLOSE ist "Dilatation, gefolgt von Erosion". Schließt kleine Löcher im weißen Inneren der Figur
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)
    # KEIN MORPH_OPEN - das erzeugt Rauschen in diesem spezifischen Setup!

    # Bewegungsmaske (optional, aktuell auskommentiert, dient aber dem hybriden Modell falls KI nicht greift)
    fgmask = fgbg.apply(gray, learningRate=0)

    # Kopiere in Hybrid-Maske
    final_mask = ki_mask.copy()

    # === NUR minimale Glättung gegen Rauschen ===
    # Bilateral Filter (Fenstergröße 5, SigmaColor 50, SigmaSpace 50) behält im Gegensatz zum Blur Kanten besser unversehrt bei
    final_mask = cv2.bilateralFilter(final_mask, 5, 50, 50)
    _, final_mask = cv2.threshold(final_mask, 140, 255, cv2.THRESH_BINARY)

    # --- Konturen füllen (aber NICHT verbinden) ---
    # RETR_EXTERNAL ignoriert innere Details/Löcher und betrachtet nur den extremsten äußeren Rand (Umriss)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(final_mask)

    if len(contours) > 0:
        # Fülle ALLE zusammenhängenden Konturen über Mindestgröße (filtert kleinste Störpixel heraus)
        min_area = 200
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED) # Weiß füllen (massiv)

    final_mask = mask_filled

    # ===== Zeitliche Glättung der gesamten Maske =====
    if prev_mask is None:
        prev_mask = final_mask.copy()
    else:
        # Erstelle ein gemischtes "Ghost Image" (Schatten) über die Frames hinweg (blended)
        blended = cv2.addWeighted(prev_mask, alpha, final_mask, 1 - alpha, 0)
        _, blended = cv2.threshold(blended, 150, 255, cv2.THRESH_BINARY) # Halte die Kanten scharf
        prev_mask = blended
        final_mask = blended

    # === Finale sanfte Kantenglättung (Anti-Aliasing) ===
    # Sehr leichter Blur für abgerundete Ränder des Endbildes
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    _, final_mask = cv2.threshold(final_mask, 220, 255, cv2.THRESH_BINARY)  # Hoher Threshold für glatte Kurven

    # --- Reales Videobild mit Binärmaske verschneiden ---
    # Erstellt ein komplett schwarzes Bild
    out_full = np.zeros_like(frame)
    # Und setzt jeden Pixel dort auf (255,255,255) Weiß, wo die Maske zutrifft (>0)
    out_full[final_mask > 0] = [255, 255, 255]

    # Hochauflösende Skalierungsvorschau auf dem PC für Testzwecke anzeigen
    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)

    # Variante 1: Mit getRectSubPix (einfachster und sicherster Weg, Proportionen beizubehalten)
    h, w = out_full.shape[:2]  # Höhe 480, Breite 640
    aspect = 32 / 48           # Das exakte LED Matrix Seitenverhältnis (Zwei Drittel = 0.66)
    
    # Logik für das Finden der maximalen Schnittfläche im korrekten Aspect Ratio (mittiger Crop)
    if w / h > aspect:
        crop_w = int(h * aspect)
        crop_h = h
    else:
        crop_w = w
        crop_h = int(w / aspect)

    # Extrahiert exakt die Mitte aus dem (out_full) Canvas heraus (Cropping statt Stretching)
    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))
    
    # Herunterrechnen von Gigantischer Auflösung auf mikroskopische 32x48 Pixel
    # INTER_AREA liefert weniger Moiré-Artefakte als INTER_LINEAR/CUBIC beim extremem Downsampling
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)

    # Miniaturansicht einblenden 
    cv2.imshow('Hybrid Silhouette (32x32)', out_small)
    
    # --- Matrix Code Einspeisung ---
    # Das Bild ist nun bereit und wird visuell an die nachgeschaltete Hardware-Schnittstelle geschickt
    ImagetoMatrix.drawImage(out_small)

    # Überprüfung, ob auf der Tastatur die ESC Taste gedrückt wurde (ASCII = 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Wenn die While True Schleife verlassen wird (Z. B. durch ESC), gib den Speicher frei
release_camera(cam)
cv2.destroyAllWindows()