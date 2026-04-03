# Implementierung: Bildverarbeitung mit MediaPipe und OpenCV

Dieses Unterkapitel detailliert die Computer-Vision-Pipeline des Skripts `Detection_with_Mediapipe.py`. Im Fokus stehen die künstliche Intelligenz (MediaPipe) zur Isolierung von Körper und Händen sowie die matrizielle Nachbearbeitung durch OpenCV. Hardware-spezifische Aufgaben (Kamera und LED-Export) werden am Ende des Kapitels gesondert behandelt.

## 1. Initialisierung der Machine-Learning-Modelle

Die Open-Source-Bibliothek *MediaPipe* übernimmt die primäre Objekterkennung.

```python
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6
)
# Optionales Fallback-Modell
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)
```

**Erklärung der Funktionen & Parameter:**
- **`SelfieSegmentation(model_selection=1)`**: Modell-ID 1 ist das auf Performanz getrimmte "Landscape"-Netz. Ein verzögerungsfreier Spiegeleffekt auf dem Matrix-Display hat oberste Priorität vor extremen Detailtiefen.
- **`Hands()`**:
  - `static_image_mode=False`: Das System nutzt historische Trackingdaten, um fließende Bewegungen statt Einzelbilder auszuwerten.
  - `min_detection_confidence=0.5`: Eine Sicherheit von 50 % reicht zur Ersterkennung aus, filtert jedoch offensichtliche False-Positives heraus.
  - `min_tracking_confidence=0.6`: Um den fortlaufenden Pfad nicht abzubrechen, benötigt das Tracking 60 % Stabilitätsschwelle.
- **`BackgroundSubtractorKNN`**: Dient als Fallback; lernt über `history=150` Frames den Hintergrund. `detectShadows=False` deaktiviert rechenintensive Schattenverfolgung.

## 2. Körpersegmentierung (KI-Maske)

Für jeden Video-Frame berechnet das Modell eine Wahrscheinlichkeitsmatrix des menschlichen Körpers.

```python
    # MediaPipe erfordert den RGB-Farbraum
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`**: Konvertiert das BGR-Kamerabild (OpenCV-Standard) ins von MediaPipe zwingend geforderte RGB-Format. Ohne diesen Schritt versagen die Modelle.
- **`segmenter.process(rgb)`**: Konstruiert eine Matrix aus Konfidenzahlen (`0.00` bis `1.00`), die angeben, mit welcher Wahrscheinlichkeit ein Pixel Körpermasse ist.
- **Schwellenwert (`> 0.4`)**: Es wird nicht der strikte Standard (`0.5`) genutzt. `0.4` ist geringfügig toleranter, womit das Abschneiden von Haaren oder Körperkonturen verhindert wird.
- **`.astype(np.uint8) * 255`**: Konvertiert den Booleschen Tensor (Wahr/Falsch) in eine klassische 8-Bit-Farbmatrix, genauer gesagt ein binäres Schwarzweiß-Bild (Weiß = `255`).

## 3. Hand-Tracking und Landmarks (Koordinaten-Extraktion)

Während der Selfie-Segmenter ungenau bei Händen ist, lokalisiert `Hands()` alle Finger exakt. MediaPipe liefert hier jedoch keine Grafik, sondern räumliche Landmarks.

```python
    hand_res = hands.process(rgb)
    hand_mask = np.zeros_like(ki_mask) # Leere schwarze Leinwand

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape  # Dimensionen des Kamerabildes berechnen
        
        for handLms in hand_res.multi_hand_landmarks:
            hand_points = []
            
            # Alle 21 Landmarks aus dem normierten Format umrechnen
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])
```

**Erklärung der Funktionen & Parameter:**
- **`hands.process(rgb)`**: Untersucht das Bild gezielt nach Händen.
- **`hand_res.multi_hand_landmarks`**: Eine Collection aus physikalischen Markierungen ("Skelettpunkten"). Existiert sie, so sind Hände im Bild.
- **`lm.x * w`, `lm.y * h`**: Die erhaltenen Landmark-Daten (`lm.x`, `lm.y`) sind Fließkommazahlen zwischen `0.0` und `1.0` (als prozentuale Bildschirmposition). Sie werden mit der echten Pixelbreite (`w`) sowie Höhe (`h`) der Auflösung multipliziert und durch `int` zu absoluten Ganzzahl-Koordinaten abgerundet.

## 4. Geometrische Konstruktion der Hände

Aus dem gesammelten Array `hand_points` wird nun physisch ein Hand-Körper auf den Matrix-Canvas gezeichnet, um Lücken in der späteren LED-Auswertung zu vermeiden.

```python
            # 1. Fülle die Handmitte als Form (Vermeidung des Skelett-Looks)
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_pts = np.array([hand_points[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(hand_mask, [palm_pts], 255)
            
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
                # 2. Zeichne Finger auf
                cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)

            # 3. Knickstellen an Gelenken abrunden
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.fillPoly`**: Basiert auf anatomischen Indizes (`0`: Handgelenk, `1,5,9,13,17`: Fingergrundgelenke) der inneren Handfläche. Dieser API-Aufruf zeichnet ein einheitlich geschlossenes Polygon, wodurch die Handmitte massiv ausgemalt (`255`) wird.
- **`cv2.line(..., thickness=18)`**: Simuliert Gliedmaßen durch Vektorlinien. Bei einer Input-Sensorbreite von 640 Pixeln entspricht eine Strichdicke (`thickness`) von `18` der proportionalen Breite menschlicher Finger und verhindert den Strichmännchen-Effekt.
- **`cv2.circle(..., thickness=-1)`**: Ein harter Linienschnitt resultiert bei angewinkelten Fingern in Störungskanten am Gelenk. Da `-1` für massive Formen steht, fungiert der mit Radius `thickness // 2` gezeichnete Gelenk-Zirkel als ein bündig abgerundetes Kurvenverbindungsstück.

## 5. Morphologische Maskenverfeinerung 

Unstrukturiertes KI-Rauschen und geometrische Ecken in den Masken werden mithilfe matrizierter OpenCV-Kernel gefiltert.

```python
    # --- Spezifische Hand-Filterung ---
    hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)
    hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)
    _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)
    hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)
    
    # --- Generelle Filterung ---
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)
    
    final_mask = cv2.bilateralFilter(ki_mask, 5, 50, 50)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.dilate` (1. Schritt)**: Ein Ausdehnungsalgorithmus. Schiebt einen (`5x5`) Kernel über dunkle Kanten. Trifft er auf ein weißes Hindernis, wird die angrenzende Schwarz-Fläche ebenfalls weiß eingefärbt. Dadurch wächst das Bildvolumen minimal, was kleine Risse in den Händen überbrückt.
- **`cv2.GaussianBlur` (2. Schritt)**: Dieser Kernel überlagert `9x9` Pixel mittels Normalverteilung, was ein sehr aggressives, graues Weichzeichnen erzeugt. Das Sigma (`0`) lässt OpenCV den Glättungsradius deterministisch aus der Kernelgröße ermitteln. 
- **`cv2.threshold(..., THRESH_BINARY)`**: Verwandelt die unscharfe Weichzeichnerwolke zurück in einen puren Binärstatus. Der Schwellenwert liegt präzise auf `"120"` angesetzt (dunkelgrau); Pixel heller als 120 werden wieder hartweiß. Das wandelt den Blur-Gradianten in organisch sanfte Außenkurven.
- **`cv2.erode` (3. Schritt)**: Die Abschälung (`3x3` Kernel). Reduziert das temporär aufgeblähte Masken-Dimensionen-Ausmaß aus Schritt 1 wieder auf realitätsgetreue Maße.
- **`cv2.bitwise_or`**: Legt das Hand-Array logisch-additiv über das Körper-Array. Unabhängig vom Layer bedeutet Weiß = Weiß und sichert einen konsistenten Hybrid-Charakter.
- **`cv2.morphologyEx(..., cv2.MORPH_CLOSE)`**: Führt zwingend in Reihenfolge eine Erweiterung gefolgt von einer Zurückstufung durch ("Closing"). Ziel von Closing-Prozessen ist das lückenlose Versiegeln fehlerhafter Schwarz-Pixel ("Salt/Pepper Noise") mitten im weißen Körperbereich.
- **`cv2.bilateralFilter(5, 50, 50)`**: Ein kantenerhaltener Spezialfilter. Anders als Gaussian Blur ignoriert er den Rand (kontrastreiche Übergänge wie zwischen Körper Weiß/Raum Schwarz). Er wischt lediglich bei extrem niedrigen Kontrastunterschieden quer durchs Körperzentrum (`5` Pixel Radius). Die Toleranzen `50 (SigmaColor)` beziehungsweise `50 (SigmaSpace)` definieren, dass Farbirritationen bis zu einem Schwellwert geglättet, aber Outlines gerettet werden.

## 6. Lückenversiegelung & Zeitliche Flimmer-Glättung

```python
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    blended = cv2.addWeighted(prev_mask, 0.25, final_mask, 0.75, 0)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.findContours`**:
  - `RETR_EXTERNAL`: Erkennt selektiv nur die äußerste physikalische Hülle der weißen Silhouette. Gefangene Innenhöhlen werden nicht evaluiert.
  - `CHAIN_APPROX_SIMPLE`: Eliminiert im resultierenden Vektor unzählige Matrix-Zwischenpunkte auf direkten Geraden, wodurch die Arbeitsspeicherlast drastisch kollabiert.
- **`cv2.drawContours`**: Mittels Zielargument `-1` verarbeitet er sämtliche Listeneinträge der Hülle aus dem Vorherigen Aufruf. `thickness=cv2.FILLED` (`-1`) malt die Umrandung nicht nach, sondern betoniert den gesamten Flächeninhalt mit massivem Binär-Weiß aus.
- **`cv2.addWeighted(src1, alpha, src2, beta, gamma)`**: Diese Formel imitiert eine mathematische Trägheit ($ \text{blended} = \text{alt} \cdot \alpha + \text{neu} \cdot \beta + \gamma $). Die Übernahme des alten Videoframes (`prev_mask`) geschieht konservativ zu 25 % (`0.25`), während die reale Realzeit auf 75 % (`0.75`) justiert wird. Das dämpft Kantenflimmern (Jittering) der Webcams massiv zur fast fehlerfreien Stabilität, das additive Offset `gamma=0` wird logischerweise ignoriert.

---

## 7. Abgrenzung: Hardware, Schnittstellen und Formatkonformen

Dieser sekundäre Code-Abschnitt dient als Rahmenhülle (`Boilerplate`), um Hardwarespezifische Funktionen zu adressieren.

### 7.1. Kamera-Initialisierung & Ausleuchtung

```python
def init_camera():
    if USE_PI:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        # ...
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    # Kalibrierungsaufhellung: 
    frame_enh = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
```
- **Umgebungslogik**: Das Skript kann flexibel für den normalen Computer (`cv2.VideoCapture`) als auch für native Raspberry Pi Boards (`Picamera2`) compiliert werden, der `BGR888`-Farbraum synchronisiert beide Architekturen auf 8-Bit.
- Die Auflösung bleibt bewusst auf `640x480` limitiert, um die neuronale Verarbeitungslast zu vermindern.
- **`cv2.convertScaleAbs`**: Berechnet $ \text{Pixel neu} = \text{Pixel alt} \cdot \alpha + \beta $. Eine Aufhellung durch 30% Kontraststeigerung (`alpha=1.3`) sowie konstante Generellaufhellung um +20 (`beta=20`).

### 7.2. Dimensionstransformation für das Matrix-LED-Format (Aspect Ratio)

Das 4:3 Webcam-Bild (640x480) auf das asymmetrische vertikale (32x48) Panel projizieren. 

```python
    aspect = 32 / 48           # Ziel-Seitenverhältnis
    
    # ... Berechnung von crop_w und crop_h
    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)
    ImagetoMatrix.drawImage(out_small)
```
- **`aspect (32/48)`**: Vordefiniertes Zielproportionenverhältnis (Quotient 0.66). Zwingend notwendig, da ein einfaches Reskalierungs-"Stretching" den Anwender gestaucht und deformiert wirken ließe.
- **`cv2.getRectSubPix`**: Kopiert punktgenau den ausgerechneten optimalen Quader (`crop_w, crop_h`) aus dem genauen Bildzentrum (`w // 2, h // 2`) heraus.
- **`cv2.resize`**: Zwingt den zugeschnittenen Quader in seine Endmaße 32x48. Das übergebene Argument `INTER_AREA` basiert mathematisch nicht auf Pixelpunkt-Näherungen, sondern auf prozentualen Flächenanteilen. Es ist die de facto standardisierte Algorithmik, um gigantische Bildstrukturen unbeschadet auf extremes *Downscaling* zu minimieren, da Formverlust und Moiré-Zeichnungen am zuverlässigsten abgewehrt werden.
