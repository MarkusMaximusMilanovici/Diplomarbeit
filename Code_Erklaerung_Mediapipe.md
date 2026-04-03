# Implementierung: Bildverarbeitung mit MediaPipe und OpenCV

Dieses Unterkapitel detailliert die Computer-Vision-Pipeline des Skripts `Detection_with_Mediapipe.py`. Im Fokus stehen die künstliche Intelligenz (MediaPipe) zur Isolierung von Körper und Händen sowie die matrizielle Nachbearbeitung durch OpenCV, in chronologischer Reihenfolge.

## 1. Kamera-Initialisierung & Hardware-Abstraktion

Zu Beginn richtet das Skript die Schnittstelle zur Kamera ein. Dieser Vorgang ist hardwareseitig abgegrenzt, um Pi-Kameras und USB-Webcams identisch ansteuern zu können.

```python
USE_PI = False
try:
    from picamera2 import Picamera2
    USE_PI = True
except ImportError:
    USE_PI = False

def init_camera():
    if USE_PI:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
        # ...
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
```

**Erklärung der Parameter:**
- Das Skript nutzt mit **`640x480`** bewusst eine sehr niedrige Auflösung, um Rechenleistung beim KI-Processing zu sparen (Echtzeit-Optimierung).
- **`BGR888`** stellt sicher, dass das Array von Beginn an im OpenCV-Standardfarbraum (Blau, Grün, Rot) aufgebaut wird.

## 2. Initialisierung der Machine-Learning-Modelle

Bevor Bilder ausgelesen werden, müssen die schweren ML-Modelle für den Arbeitsspeicher instanziiert werden.

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

**Erklärung der Parameter:**
- **`SelfieSegmentation(model_selection=1)`**: Modell-ID 1 ist das performante Landschafts-Netz. Für den Spiegeleffekt auf dem Matrix-Display hat eine hohe Framerate absolute Priorität, da Detailtiefen (Modell-ID 0) auf 32x48 Pixeln ohnehin verloren gehen.
- **`Hands(static_image_mode=False)`**: Der Parameter `False` teilt der KI mit, dass es sich um einen Videostream handelt. Die KI cacht dadurch den letzten Tracking-Pfad der Hände und liefert glattere Bewegungen.
- Die **`min_detection_confidence=0.5`** und **`min_tracking_confidence=0.6`** verlangen vom Algorithmus mindestens 50 % Wahrscheinlichkeit zum Finden der Hand und >60 % für eine flüssige Fortführung des Tracks. Dies unterminiert *False-Positives* (Z. B. Falten im Pullover).
- **`BackgroundSubtractorKNN`**: Ein Fallback für klassische Pixelunterschiede. `history=150` legt einen Speicher von 150 Frames für das Hintergrund-Lernen fest, `detectShadows=False` schont Rechenressourcen.

## 3. Kamerakalibrierung und Umgebungsanalyse

Nachdem die Modelle gestartet sind, erfasst die Kamera 100 Frames im leeren Raum, um sich auf das Licht einzustellen. Dieser Prozess ist essenziell für die Robustheit.

```python
# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    # ...
    # Zwingende Bildaufhellung:
    alpha_bright = 1.3
    beta_bright = 20
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)

    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    
    fgbg.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), learningRate=0.5)
```

**Erklärung der Parameter:**
- **`cv2.convertScaleAbs(alpha, beta)`**: Diese mathematische Funktion rechnet für jeden Pixel `$ \text{Neu} = \text{Alt} \cdot \alpha + \beta $`.
- **`alpha=1.3`** skaliert den Kontrastfaktor um 30 % nach oben, während **`beta=20`** eine absolute Addition von 20 Helligkeitspunkten aufspannt. Dies zwingt selbst bei schlechten Lichtverhältnissen Hauttöne drastisch zum Vorschein, was ein Versagen der KI im Dunkeln blockiert.
- Das Setzen der `learningRate=0.5` beim `fgbg.apply` lässt den KNN-Subtractor extrem rasch das grundlegende Rauschen des leeren Zimmers auswendig lernen.

## 4. Echtzeit-Segmentierung (Körpermaske)

Innerhalb des eigentlichen `While True` Loops wird für jeden Frame die Wahrscheinlichkeitsmatrix des menschlichen Körpers errechnet.

```python
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255
```

**Erklärung der Parameter:**
- **`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`**: Wandelt das OpenCV BGR-Bild nun zwingend in Rot-Grün-Blau um, ansonsten würde die Google-KI fehlschlagen.
- **Schwellenwert (`> 0.4`)**: MediaPipe gibt Werte von `0.0` bis `1.0` (Körper-Konfidenz) aus. Der reduzierte Schwellenwert von `0.4` (statt strikten 0.5) macht die Maskengrenzen geringfügig toleranter. Das verhindert das versehentliche Abschneiden von Haaren oder Kleidungsrändern in den Randbereichen.
- **`.astype(np.uint8) * 255`**: Konvertiert das Boolean-Array (True/False) exakt in reine 8-Bit-Farbwerte zurück (Weiß = `255`, Schwarz = `0`).

## 5. Hand-Tracking und Landmark-Konstruktion

Die Körpersilhouette von `SelfieSegmentation` ist im Handbereich zu unpräzise. `Hands()` löst dieses Problem punktgenau, spuckt anstatt Weißer Pixel allerdings nur Landmarks (Koordinaten im 3D Raum) aus. Diese müssen algorithmisch in eine visuelle Matrix konvertiert werden.

```python
    hand_res = hands.process(rgb)
    hand_mask = np.zeros_like(ki_mask)

    if hand_res.multi_hand_landmarks:
        h, w, _ = frame.shape  
        for handLms in hand_res.multi_hand_landmarks:
            hand_points = []
            
            # Alle 21 Landmarks aus dem normierten Format umrechnen
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append([cx, cy])

            # 1. Handfläche füllen
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_pts = np.array([hand_points[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(hand_mask, [palm_pts], 255)
            
            # 2. Finger zeichnen
            connections = [(0, 1), (1, 5), ..., (19, 20)] # Gelenkverknüpfungen 
            finger_thickness = 18 
            for connection in connections:
                pt1 = tuple(hand_points[connection[0]])
                pt2 = tuple(hand_points[connection[1]])
                cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)

            # 3. Gelenke abrunden
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)
```

**Erklärung der Parameter:**
- **`lm.x * w`**: Landmark-Daten in MediaPipe statten nur relative Prozentwerte (`0.0` bis `1.0`) aus. Erst durch Multiplikation mit der echten Kamerabreite `w` (640) und der Höhe `h` (480) entstehen absolute, zeichenbare Pixel-Zahlenkoordinaten (`cx, cy`).
- **`cv2.fillPoly`**: Anhand der anatomischen Indizes (`0`: Handgelenk, `5, 9, 13, 17`: Fingeransätze) wird zentral ein umschlossenes Polygon ("Palm") in reines Weiß gemalt. Das dichtet die innere Handfläche restlos ab und verhindert Lücken im finalen Bild.
- **`cv2.line(thickness=18)`**: Die Liniendicke `18` Pixel liefert proportional die exakte physiologische Breite eines menschlichen Fingers, der bei normiertem Abstand in eine analoge 640p-Kamera gehalten wird.
- **`cv2.circle(..., thickness=-1)`**: An extremen Knickstellen (z. B. eine geschlossene Faust) ragen Vektorgeraden spitz und unbündig zusammen. Durch das Setzen voll-gefüllter Kreise (`thickness=-1` = in OpenCV massiv) auf alle 21 Gelenkkoordinaten werden diese Ecken glatt und physikalisch korrekt abgerundet.

## 6. Morphologische Maskenverfeinerung 

Die geometrisch vektorisierte Hand und das raue KI-Rauschen der Hauptmaske weisen noch harte Treppenkanten auf. Faltungs-Filter-Kernel (`Kernels`) lösen dies.

```python
    # Hand-Morphologie
    hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)
    hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)
    _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)
    hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)
    
    # Körper / Hybrid-Morphologie
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)
```

**Erklärung der Parameter:**
- **`cv2.dilate` (1. Schritt)**: Ein Filterausdehnungsalgorithmus (`5x5` Kernel). Trifft der Kernel auf eine Körperkante, färbt er benachbartes Schwarz der Umgebung sofort Weiß. Das dichtet feine Risse an der Outline ab.
- **`cv2.GaussianBlur` (2. Schritt)**: Eine gewaltige Matrixglocke (`9x9`), die harte Kanten extrem in Unschärfe und weiche Grautöne "kaputtwäscht".
- **`cv2.threshold(..., THRESH_BINARY)`**: Operiert auf dem gewaschenem Grau und schneidet es exakt in der Schwellenwertmitte bei `120` in strahlendes Weiß `255` und pures Schwarz (`0`) auf. Dadurch heilen kantige Hände und Treppeneffekte des Videobildes zu organisch-menschlichen Radien.
- **`cv2.erode` (3. Schritt)**: Wirft durch die Kernabtragung (`3x3`) das überflüssige Material ausritts der anfänglichen Ausdehnung (Dilatation) restlos ab, was die Maskentexturen bei identischen Originalausmaßen aushärtet.
- **`cv2.bitwise_or`**: Addiert Hand-Kanal und Körpersilhouette via verlustfreier ODER-Verknüpfung.
- **`cv2.morphologyEx(..., cv2.MORPH_CLOSE)`**: Vollzieht ein striktes "Closing" – die Abriegelung minimalster vereinzelter "Pepper-Noise" Schwarzpixel, welche die KI als winzige Auslese-Irrtümer fälschlicherweise in der weißen Brust der Person verankert hat.

## 7. Zusammenführung & Flimmer-Glättung

```python
    final_mask = cv2.bilateralFilter(ki_mask, 5, 50, 50)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    blended = cv2.addWeighted(prev_mask, 0.25, final_mask, 0.75, 0)
```

**Erklärung der Parameter:**
- **`cv2.bilateralFilter(5, 50, 50)`**: Während `GaussianBlur` kompromisslos verschmiert, ignoriert der Bilateral-Filter absolute Kontrastwechsel (Mensch Weiß / Wand Schwarz) und wischt lediglich zart im Zentrum lokaler Bereiche (Raster-Radius `5`). Sein Toleranzwert bei `50` SigmaColor respektiert harte Schattenwürfe von Objekten, eliminiert jedoch unbedeutendes Hintergrundfeinrauschen.
- **`cv2.findContours`**: 
  - `RETR_EXTERNAL`: Umkreist physikalisch zwingend exklusiv die Außenschale des Anwenders. Hohlräume wie verschränkte Arme interessieren nicht.
  - `CHAIN_APPROX_SIMPLE`: Wirft zehntausende unnütze Positionsdaten auf der Strecke ab und abstrahiert die Vektoren nur anhand der Stützpfeiler.
- **`cv2.drawContours`**: Über `thickness=cv2.FILLED (-1)` flutet der Befehl den leeren Polygon-Inhalt komplett zuweiß (`255`), was jegliche Löcher in der Silhouette garantiert blockt.
- **`cv2.addWeighted(src1, alpha, src2, beta, gamma)`**: Ein exponentielles "Nachziehen". Bildpixel-Varianzen (Jittering) auf schwachen Webcams werden getilgt, da beständig exakt `25 %` (`alpha=0.25`) der Matrix des veralteten, vorherigen Zyklus additiv mit `75 %` des Echtzeit-Zyklus verrechnet bleiben. So erstirbt das Flimmern drastisch, da fehlerhafte Pixel stetig durch den Shadow-Beweis der Vorframes abgemildert werden.

---

## 8. Abgrenzung: Dimensionstransformation für das LED-Panel

Im Schlussakkord ist das Array fertig segmentiert – muss jedoch hardwareseitige Skalierungen durchlaufen, um per Seriell-Schnittstelle in den Screen der Micro-Controller-Matrix (32x48) gepusht zu werden.

```python
    aspect = 32 / 48           # Ziel-Seitenverhältnis Hochkant
    
    # ... Berechnung von Sub_Ratios crop_w und crop_h
    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)
    ImagetoMatrix.drawImage(out_small)
```

**Erklärung der Parameter:**
- **`aspect (32/48)`**: Da Webcams extrem verbreitert aufnehmen (`4:3`), die Matrix jedoch scharf hochkant (`2:3` bzw. `0.66`) gelötet ist, wäre schlichtes Quetschen fatal. Der Anwender würde im Spiegel langgezogen deformiert werden.
- **`cv2.getRectSubPix`**: Der Filter extrahiert das mathematisch makellose 2:3 Sub-Seitenverhältnis berechnet als Quadrat (`crop_w, crop_h`) aus der Toten Mitte der Kamera, zentriert auf `(w // 2, h // 2)`. Die lateralen Ränder des Kamerastroms blenden spurlos weg.
- **`cv2.resize`**: Der Kern-Übersetzungsschritt. Die Parameter-Deklaration `INTER_AREA` bewirkt, dass die Skalierung weg von tausenden HD Einzelpixeln in 32x48 kleine Blöcke vollkommen verlustfrei über massive prozentuale Flächenkalkulationen geschieht und nicht linear gelöscht („gesampled“) wird. Das beschützt winzige Details (etwa aufgespannte Finger von Landmark) vor unleserlichem *Moiré*-Flackern, kurz bevor der Befehl in `drawImage()` ausfeuert.
