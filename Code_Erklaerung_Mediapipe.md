# Implementierung: Bildverarbeitung mit MediaPipe und OpenCV

In diesem Unterkapitel wird die Funktionsweise und Architektur des Skripts `Detection_with_Mediapipe.py` im Detail erläutert. Das Programm ist so konzipiert, dass es den Anwender vor der Kamera erkennt, Körper und Hände vom Hintergrund isoliert und diese Informationen anschließend optimiert an das LED-Matrix-System (32x48 Pixel) weiterleitet. Der gesamte Code durchläuft verschiedene Stufen der Computer-Vision.

## 1. Importe und Dynamische Kameraauswahl

Das Skript beginnt mit dem Import der benötigten Bibliotheken und richtet die Kameraarchitektur so ein, dass es plattformunabhängig lauffähig ist.

```python
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
```

**Erklärung:**
Das System muss sowohl auf Entwicklungsrechnern als auch auf der finalen Hardware (Raspberry Pi) ohne Code-Änderungen funktionieren. Durch den `try-except`-Block wird geprüft, ob die Bibliothek `picamera2` installiert ist. Ist dies der Fall, läuft der Code nativ auf dem Pi. Schlägt der Import fehl, schaltet das Programm auf den Entwicklermodus mit einer herkömmlichen Webcam um. 

## 2. Abstraktion der Kamerafunktionen

Um den Code lesbar zu halten, sind die unterschiedlichen Kamerazugriffe in Kapselungsfunktionen ausgelagert.

```python
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
```

**Erklärung der Parameterwahl:**
Die Auflösung wurde bewusst auf **640x480 Pixel** (VGA) limitiert. Da das Ziel-Interface lediglich eine LED-Matrix mit gigantischen Pixelabstufungen (32x48) ist, wäre die Verarbeitung von HD- oder 4K-Bildern eine Ressourcenverschwendung. Die geringe Auflösung sorgt stattdessen für einen starken Performance-Vorsprung (`FPS - Frames per Second`), der wichtig ist, um eine echtzeitfähige Interaktion (Spiegeleffekt ohne Latenz) aufrechtzuerhalten. Das Format `BGR888` ist der Standard-Farbraum in OpenCV.

## 3. Modelle laden und initialisieren

Hier werden die vor-trainierten Machine-Learning Netze und OpenCV-Objekte instanziiert, bevor sie auf den Videostream angewandt werden.

```python
# ============================================================
# MediaPipe + Background-Subtractor
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
# Aus historischen Gründen oder für Bewegungen initalisiert
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)

# Kamera letztendlich starten
cam = init_camera()
```

**Erklärung der Architektur:**
MediaPipe wird als künstliche Intelligenz eingesetzt.
- **SelfieSegmentation (`model_selection=1`):** Nutzt das effiziente Modell, um Rechenleistung zu sparen. Das extrem detailreiche `model_selection=0` ist langsamer und für 32x48 Pixel nicht nötig. 
- **Hands:** Parameter wie `min_detection_confidence=0.5` und `min_tracking_confidence=0.6` wurden empirisch abgewogen. Wenn man den Wert höher stellt, verwirft das System bei der kleinsten Unschärfe die Hand. Stellt man ihn auf 0.2, kommt es zu False Positives (z. B. Falten im T-Shirt werden fälschlicherweise als Hände erkannt).

## 4. Kalibrierungsphase: Kamerahelligkeit und Umgebungsdaten lernen

Damit sich das System auf den Raum einstellen kann, liest es zu Beginn 100 Bilder statisch ein.

```python
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
```

**Erklärung der Parameterwahl:**
Ein grundlegendes Problem von billigeren Kameras (wie Raspberry Pi Modulen) in Wohnräumen ist die Unterbelichtung. Über `alpha_bright` und `beta_bright` zieht das Programm die Werte hoch, sodass Hautfarben für MediaPipe deutlich dominanter und heller zur Geltung kommen. Ein dunkles Zimmer führt sonst zum vollständigen Verlust der Segmentierungsdaten.

## 5. Vorbereitung der zeitlichen Glättung und Beginn der Hauptschleife

Um Bildflimmern (Jitter) zu vermeiden, nutzt das Programm Trägheit in Form eines "Gleitenden Durchschnitts".

```python
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Graustufenbild für Bewegungsfallbacks
```

## 6. Generierung der Körpermaske

Der Kernprozess erzeugt zunächst eine binäre Schwarz/Weiß Matrix aus den Segmentierungsdaten des Oberkörpers (Selfie). Das geschieht über die Konfidenz.

```python
    # MediaPipe Person Segmentierung (Benötigt RGB Format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    
    # Thresholding: Ab 40% Sicherheit (0.4) dass ein Pixel zum Benutzer gehört, wird er auf 'Weiß' (255) gesetzt.
    # .astype(np.uint8) wandelt die Wahrheitswerte (True/False) der Matrix in ganze Zahlen (0/1) um
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255
```

**Erklärung der Architektur:**
MediaPipe verarbeitet das Bild intern über ein TensorFlow Lite Netz. `res.segmentation_mask` enthält eine Fließkomma-Matrix (Werte zwischen 0.00 und 1.00), in der jeder Pixelwert die Wahrscheinlichkeit repräsentiert, dass dieser Punkt zu einer menschlichen Silhouette gehört. 
- **Threshold 0.4:** Ein rigoroses Standardvorgehen für saubere Masken wäre bei `> 0.5`. Hier wurde gezielt `> 0.4` gewählt, um die Akzeptanz zu lockern und abrupte Abbrüche an Schultern oder Armen zu verhindern. Dass das Bild im Gegenzug weicher bis zur Grenze von Unschärfe am Rand wird, ist beabsichtigt (es entsteht ein "Aura"-Effekt der KI-Segmentierung).

## 7. Spezifische Hand-Erkennung & Geometrisches Füllen

MediaPipe liefert für Hände keine vollflächigen Masken, sondern nur abstrakte Orientierungspunkte im Raum ("Landmarks"). Die Rekonstruktion eines Polygon-Gehäuses aus diesen Punkten auf dem blanken schwarzen Canvas ist erforderlich, ansonsten würden sie auf der LED Matrix wie ein "Skelett" wirken.

```python
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
                (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),    # Handfläche Umrandung
                (1, 2), (2, 3), (3, 4),                                # Daumen
                (5, 6), (6, 7), (7, 8),                                # Zeigefinger
                (9, 10), (10, 11), (11, 12),                           # Mittelfinger
                (13, 14), (14, 15), (15, 16),                          # Ringfinger
                (17, 18), (18, 19), (19, 20)                           # Kleiner Finger
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
```

**Erklärung der Architektur und Parameter:**
Die 21 Landmarks beschreiben das pure physikalische Skelett der Hand. Zeichnet man diese auf, erscheinen jedoch nur dünne Linien.
- **`cv2.fillPoly`:** Die Indizes (0,1,5,9,13,17) sind die Kantenpunkte der Handflächen-Basis. Hier wird ein massives Polygon gezeichnet, welches die Handmitte völlig schließt und Löcher vermeidet.
- **`cv2.line` & `cv2.circle`:** Eine Linie mit einer dicke von `18` Pixeln simuliert die ungefähren anatomischen Hautmaße der Finger zu diesem Kamera-Format (640x480). Bei eckigen Winkeln ragen Linien unbündig übereinander. Deshalb wird mit `thickness // 2` ein ausgefüllter Kreis exakt auf jedes Gelenk gezeichnet – die Finger erscheinen weich und abgerundet.

## 8. Morphologische Filterung der Masken (Dilatation/Erosion)

Eine rein geometrisch konstruierte Hand in Verbindung mit der verpixelten KI-Silhouette führt oft zu Ausfransungen und abgehakten Kanten.

```python
    # ===== Hand-Maske verfeinern (WENIGER aggressiv) =====
    if np.any(hand_mask > 0): # Überspringen, falls keine Hände auf dem Bild sind, spart Rechenzeit

        # Moderate Dilatation (Aufblähen): Schließt feine Lücken zwischen den Fingern
        kernel_medium = np.ones((5, 5), np.uint8) 
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)

        # Moderater Blur (Unschärfefilter): Rundet die geraden Polygon-Kanten extrem stark ab, dass sie organisch wirken
        hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)  
        _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)  
        # Durch den hohen Threshold wird der graue Unschärfebereich zu knackigen, weißen Kanten konvertiert.

        # Leichte Erosion (Schälen): Entfernt überschüssiges "Restfleisch", das durch die anfängliche Dilatation hinzugekommen ist (Proportionserhalt)
        kernel_small = np.ones((3, 3), np.uint8)  
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)

    # ===== Zeitliche Glättung nur für das Handgerüst =====
    if prev_hand_mask is None:
        prev_hand_mask = hand_mask.copy()
    else:
        # Die addWeighted Methode verknüpft x% der Maske des letzten Frames mit y% des aktuellen Frames -> Frameblending
        hand_blended = cv2.addWeighted(prev_hand_mask, alpha_hand, hand_mask, 1 - alpha_hand, 0)
        _, hand_blended = cv2.threshold(hand_blended, 140, 255, cv2.THRESH_BINARY) 
        prev_hand_mask = hand_blended
        hand_mask = hand_blended

    # Danach wird die generierte KI Körper-Maske mit der Hand-Maske verschmolzen (Bitwise OR)
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)
```

**Erklärung der Parameterwahl:**
In der Morphologie ist die Reihenfolge entscheidend. Die Dilatation (mit einem `5x5` Kern) vergrößert als Erstes alle weißen Segmente. Der nachfolgende **Gaussian Blur** (`9x9`) ist extrem ausgeprägt. Er zerstört die eckigen Architekturen des Polygons völlig, das Resultat ist ein unförmiger grauer Matschfleck. Durch den anschließenden **Schwellenwert** (`120`) wird der Rand dieses "Matschs" aber exakt aufgespalten und als harte Vektorkurve umgelegt. Die Erosion (`3x3`) verkleinert die leicht aufgeblähten Dimensionen wieder auf Originalgröße. Letztendlich kombiniert **Boolsches OR** beide Masken (Person sowie Hand) additiv und verlustfrei.

## 9. Hybride Kombination & Glättung der Gesamt-Silhouette (Bilateral Filter)

Jetzt, wo Körper und optimierte Hände auf dem gleichen Canvas (`ki_mask`) zu sehen sind, durchläuft auch dieser Canvas die finale Glättung.

```python
    # === Leichte morphologische Operationen gegen Rauschen am Rande der Maske ===
    kernel_denoise = np.ones((3, 3), np.uint8)
    
    # MORPH_CLOSE ist im Prinzip "Dilatation, gefolgt von Erosion". Schließt schwarze Löcher im weißen Inneren der Figur
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)
    
    # Bewegungsmaske (optional, aktuell auskommentiert, dient aber dem hybriden Modell falls KI nicht greift)
    fgmask = fgbg.apply(gray, learningRate=0)

    # Kopiere in Hybrid-Maske
    final_mask = ki_mask.copy()

    # === NUR minimale Glättung gegen Rauschen ===
    # Bilateral Filter (Fenstergröße 5, SigmaColor 50, SigmaSpace 50) behält im Gegensatz zum Blur Kanten besser unversehrt bei
    final_mask = cv2.bilateralFilter(final_mask, 5, 50, 50)
    _, final_mask = cv2.threshold(final_mask, 140, 255, cv2.THRESH_BINARY)
```

**Erklärung der Architektur:**
Auffällig ist hier die Benutzung eines **Bilateral Filters** anstelle des bisherigen Gaussian Blurs. Die Maske besitzt an dieser Stelle viele unsaubere Artefakte. Der Gaußsche Weichzeichner verwischt konsequent Kanten mit Texturen. Der *Bilateral Filter* ist aufwändiger: Er analysiert die Farbwerte. Wenn zwei Pixel sich im Wert ähnlich sind, werden sie verwaschen. Gibt es aber einen harten Kontrast (`Wert > 140`), was einem Kantensprung zwischen Hintergrund/Körper entspricht, blockiert der Filter. Dadurch "wäscht" er das Rauschen aus der Maske heraus, lässt die Ränder der Person jedoch messerscharf bestehen.

## 10. Konturen versiegeln

Wenn sich Arme über den Körper kreuzen, können unerkannte Löcher (z.B. der Zwischenraum zwischen Arm und Hüfte) innerhalb der Figur entstehen.

```python
    # --- Konturen füllen (aber NICHT verbinden) ---
    # RETR_EXTERNAL ignoriert Löcher und betrachtet nur den extremsten äußeren Rand (Umriss)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(final_mask)

    if len(contours) > 0:
        # Fülle ALLE zusammenhängenden Konturen über Mindestgröße
        min_area = 200
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED) # Weiß füllen (massiv)

    final_mask = mask_filled
```

**Erklärung der Parameterwahl:**
`cv2.findContours` mit dem Modus `RETR_EXTERNAL` zieht ein unsichtbares Band um den äußersten Rand der weißen Maske und ignoriert alles, was sich darin befindet. `thickness=FILLED (-1)` übermalt das gesamte umschlossene Innere. Falls die Segmentierung also ein "Loch" in der Brust des Anwenders gelassen hatte, ist dieses hiermit physisch und flächig korrigiert. `min_area = 200` unterdrückt kleinste Detektionsfehler (vereinzelte Störpixel) – sie werden einfach weggelassen.

## 11. Finales Maskenblending (Gesamtkörper) & Aliasing

Wie zuvor bei der Handmaske generiert das System hier ein "Trägheitsmoment" für die gesamte Maske, bevor sie ins Rendering geht.

```python
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
    # Und setzt jeden Pixel dort auf (255,255,255) Weiß, wo die Maske Pixel enthält
    out_full[final_mask > 0] = [255, 255, 255]

    # Hochauflösende Skalierungsvorschau auf dem PC für Testzwecke anzeigen
    preview = cv2.resize(out_full, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Hybrid Silhouette (gross)', preview)
```

**Erklärung der Architektur:**
Durch `cv2.addWeighted` (`alpha = 0.25`) steuert das Programm das zeitliche Ausklingen des alten Bildes. Ein generierter Rand-Pixel muss über mehrere verknüpfte Frames hinweg konstant und beständig erkannt werden, damit das Flackern aufhört. Auf großen Matrix-Flächen ist Rauschunterdrückung essenziell.

## 12. Umrechnung in Aspect Ratio & LED-Matrix Crop

Der letzte und gravierendste Abschnitt kümmert sich um die Formatkonvertierung für das LED Modul.

```python
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
```

**Erklärung der Architektur und Parameter:**
Ein wesentliches Problem der Transformation besteht darin, dass das Kamera-Ausgangsmaterial normalerweise ein Seitenverhältnis von 4:3 (640x480) aufweist, während die physikalische LED-Wand auf hochkantes Format (32x48 = 2:3) ausgelegt ist. Ein unbedachtes Reskalieren (horizontales Stauchen) würde zu einer unnatürlichen Verzerrung führen und den Benutzer unrealistisch schmal aussehen lassen. 
Stattdessen schneidet die Logik hier einen perfekten zentralen Kasten (`crop_w, crop_h`) aus. Durch `cv2.getRectSubPix` wird der berechnete, nicht verzerrte Mittenausschnitt extrahiert. Dieses Bildteil wird durch den `cv2.resize` Intervall `INTER_AREA` für eine artefaktfreie Verkleinerung (Anti-Aliasing im Downsampling auf Pixel-Ebene) interpoliert. Die Miniaturmatrix kann nun den Code in `ImagetoMatrix.drawImage()` fehlerfrei ausführen.

## 13. Speicherfreigabe und Programmbeendigung

Am Schluss des Endlosschleifen-Constructs (`while True`) muss aufgeräumt werden.

```python
# Wenn die While True Schleife verlassen wird (Z. B. durch ESC), gib den Speicher frei
release_camera(cam)
cv2.destroyAllWindows()
```

**Erklärung:**
Diese zwei Zeilen stellen sicher, dass das Kameramodul im Betriebssystem deallokiert ('released') wird (damit es nach Beenden für die reguläre Windows Kamera-App oder andere Linux-Funktionen zur Verfügung steht), und dass sämtliche von OpenCV instanziierten Grafikfenster (`cv2.imshow`) im Anwendungs-Kontext ordnungsgemäß zerstört und vom Arbeitsspeicher des Computers freigegeben werden (`cv2.destroyAllWindows`).
