# Implementierung: Bildverarbeitung mit MediaPipe und OpenCV

In diesem Unterkapitel wird die Funktionsweise und Architektur des Skripts `Detection_with_Mediapipe.py` strukturiert erläutert. Das Programm erkennt den Anwender vor der Kamera, löst Körper und Hände vom Hintergrund und bereitet die generierte Silhouette im korrekten Seitenverhältnis für die 32x48-LED-Matrix auf. 

## 1. Kamera-Abstraktion (init_camera, get_frame)

Der Code nutzt eine einheitliche Schnittstelle, unabhängig davon, ob das Programm auf einem Raspberry Pi (`Picamera2`) oder einem Windows-Rechner (`cv2.VideoCapture`) ausgeführt wird.

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
        return cap
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.VideoCapture(0)`**: Öffnet die primäre Standard-Webcam am Laptop.
- **`CAP_PROP_FRAME_WIDTH` / `HEIGHT` `(640, 480)`**: Setzt die Sensorauflösung. Der gewählte VGA-Standard (640x480) wird verwendet, da eine höhere Auflösung für die nachfolgende Skalierung auf 32x48 Pixel unnötige Rechenlast der neuronalen Netze bedeuten würde.
- **`BGR888`**: OpenCV verarbeitet Bildmatrizen intern nicht als standardmäßiges RGB, sondern im invertierten BGR-Farbraum (Blau, Grün, Rot, mit je 8 Bit = 255 Werte), daher wird die Pi-Kamera explizit auf dieses Format genormt.

## 2. Initialisierung der Machine-Learning-Modelle

Die von Google stammende Open-Source Bibliothek *MediaPipe* wird primär zur Objektsegmentierung genutzt.

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
fgbg = cv2.createBackgroundSubtractorKNN(history=150, dist2Threshold=400, detectShadows=False)
```

**Erklärung der Funktionen & Parameter:**
- **`SelfieSegmentation(model_selection=1)`**: Modell-ID 1 wendet ein extrem performantes ("Landscape"-) Netz an. Variante 0 wäre detaillierter, liefert aber nicht die hohe Framerate, welche für einen störungsfreien Echtzeit-Spiegeleffekt unumgänglich ist.
- **`Hands()`**:
  - **`static_image_mode=False`**: Teilt MediaPipe mit, dass ein Videosteam verarbeitet wird. Dies zwingt den Algorithmus, historische Trackerdaten von Handbewegungen vergangener Frames zu cachen und zu nutzen. Dadurch läuft es weitaus performanter.
  - **`min_detection_confidence=0.5`**: Hände müssen vom Netz mit einer Sicherheit von mindestens 50% detektiert werden, andernfalls wird die Hand ignoriert. Dadurch werden Fehlmessungen minimiert (False-Positives wie etwa Hemdfalten).
  - **`min_tracking_confidence=0.6`**: Nach erfolgreicher Detektion muss der Track von Bild zu Bild zu 60% stabil bleiben. Dies erzwingt robuste Verfolgung bei schnellen Schwüngen.
- **`createBackgroundSubtractorKNN`**: Ein optionales System für klassische Hintergrundsubtraktion, basierend auf "K-Nearest-Neighbors". `history=150` legt fest, dass der Raum-Hintergrund über 150 Frames angelernt und evaluiert wird. Parameter `detectShadows=False` spart drastisch Rechenzeit, da eine separate Eigenschattierungsanalyse erzeugt werden würde, welche im LED-Endprodukt unsichtbar bleibt.

## 3. Kamerakalibrierung und Vorkorrektur

Zur Absicherung in schwach beleuchteten Räumen lernt die Kamera über 100 Frames den Hintergrundraum und führt künstliches Aufhellen durch.

```python
    frame_enh = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
    rgb = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2RGB)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.convertScaleAbs(alpha, beta)`**: Modifiziert das Matrix-Array des Kamera-Bildes nach der Formel: $ \text{NeuerPixel} = \text{AlterPixel} \cdot \alpha + \beta $.
  - `alpha=1.3`: Führt eine Skalierung und damit 30%ige Kontraststeigerung durch.
  - `beta=20`: Verschiebt das gesamte RGB-Spektrum konstant um +20 Einheiten herauf (generelle Aufhellung). Das lässt Haut in dunklen Umgebungen stark für MediaPipe hervortreten.
- **`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`**: Konvertiert den OpenCV BGR-Farbraum in RGB, da Googles MediaPipe ausschließlich mit Rot-Grün-Blau-Mustern trainiert wurde und sonst komplett versagen würde.

## 4. Körpersegmentierung (KI-Maske)

Innerhalb der Echtzeitschleife (`while True`) wird pro Bild-Zyklus aus dem Videostream eine visuelle Binärmaske konstruiert.

```python
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255
```

**Erklärung der Funktionen & Parameter:**
- **`segmenter.process(rgb)`**: Das aufgerufene Inferenz-Modell wehrt das Bild aus und formt eine Fließkomma-Matrix. Darin ist jeder Pixel mit einem float-Wert zwischen 0.00 und 1.00 bewertet, was die Wahrscheinlichkeit darstellt, dass es sich um Körpermaterial handelt.
- **`Schwellenwert > 0.4`**: Alle Array-Werte über `0.4` werden als gültig markiert (entspricht Wahrheitsgehalt True). Der unkonventionelle Schwellenwert von `0.4` statt 0.5 wird als Abmilderung genutzt. Wenn der Schwellenwert zu strikt ist, reißt die Maske schnell an Haaren oder dünner Kleidung ab.
-  Der Operator `.astype(np.uint8) * 255` konvertiert die aus den Bedingungen resultierenden True/False Ausdrücke zurück in reine Farbsignale für Bild-Matrizen ("8 Bit Integer", `1 * 255 = 255 (Weiß)`).

## 5. Hand-Rekonstruktion

Da MediaPipe bei Händen lediglich die Koordinaten (`Landmarks`) von 21 Gelenkpunkten extrahiert, muss die fehlende Hand als Vektorgrafik auf den schwarzen Canvas gezeichnet werden, damit sie gefüllt repräsentiert wird.

```python
    # 1. Fülle die Handmitte als Form
    cv2.fillPoly(hand_mask, [palm_pts], 255)
    
    # 2. Zeichne Finger auf
    cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)
    
    # 3. Knickstellen an Gelenken abrunden
    cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.fillPoly`**: Diese Funktion schlägt abstrakte Punkte im Raum zu einem gefüllten Polygon zusammen. Hier nutzt es die 6 Knotenpunkte der Handwurzel. Wenn dieser Schritt fehlt, bleibt die Innenseite der Handfläche transparent (ein "Loch").
- **`cv2.line(thickness=18)`**: Simuliert Gliedmaßen durch Vektorstriche. Bei einer Input-Sensorbreite von 640 Pixeln entspricht eine Strichdicke (`thickness`) von `18` Pixeln verhältnisgleich etwa der durchschnittlichen physiologischen Breite menschlicher Finger in Armlänge zur Webcam.
- **`cv2.circle(..., thickness=-1)`**: Verbinden sich die groben Striche der Fingergelenke im Winkel, ragt der Startpunkt unbündig eckig über. Durch das Generieren eines vollen Kreises (`thickness=-1` heißt in OpenCV, das Konstrukt massiv füllen) exakt auf den Schnittpunkt des Gelenks werden alle Kanten abgerundet.

## 6. Morphologische Maskenverfeinerung 

Die geometrisch gebaute Vektor-Hand und die gekörnte KI-Silhouette besitzen harte Polygonkanten und Rauschen, das über Matrix-Filter optimiert werden muss.

```python
    hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)
    hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)
    _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)
    hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)
    
    ki_mask = cv2.morphologyEx(ki_mask, cv2.MORPH_CLOSE, kernel_denoise, iterations=1)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.dilate`**: Ausdehnungsalgorithmus. Schiebt einen kleinen Abtastungsschieber (`kernel 5x5`) über die Bildkanten. Trifft er auf ein weißes Hindernis, wird die angrenzende Fläche ebenfalls auf Weiß erweitert. Das bläht Lücken zu (wie Schwimmhäute) und lässt Finger breiter wirken. `iterations=1` verhindert unendliche Übergriffe durch einmalige Ausführung.
- **`cv2.GaussianBlur`**: Nutzt eine Normalverteilung (Gaußsche Glocke) in einer `9x9`-Matrix, um Farbverläufe weichzuzeichnen. Das Sigma (`0`) lässt OpenCV die radiale Unschärfe automatisch anhand der 9er Größe abrunden. Es zersetzt Polygontreppen zu grauen Wolken.
- **`cv2.threshold(..., THRESH_BINARY)`**: Verwandelt die unscharfen Graustufen wieder rigoros zurück zu scharfem Schwarz und Weiß mit einem absoluten Cut-Off-Point bei `120` (ca. Mittleres Grau). Die Endresultate sind nun perfekt glatt gebogene Kanten ohne Treppeneffekt.
- **`cv2.erode`**: Das Gegenstück; der Abschälalgorithmus. Trägt die Randpixel einer `3x3`-Kernelbreite wieder minimal ab. Dies gleicht die überblähten Volumen-Proportionen aus, die durch `dilate` hinzugefügt wurden. 
- **`cv2.morphologyEx(..., cv2.MORPH_CLOSE)`**: Führt im Kern eine sequentielle Dilatation gefolgt von Erosion aus (sogenanntes Closing). Fehlerhafte schwarze Pixel-Löcher („Salt and Pepper Noise“) innerhalb eines Körperbauteils werden dadurch versiegelt.

## 7. Zusammenführung und Kantenfilter

Zuletzt fließen die dedizierte Handerkennung und die Ganzkörpersilhouette wieder in ein gemeinsames Bild (`ki_mask`).

```python
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)
    final_mask = cv2.bilateralFilter(ki_mask, 5, 50, 50)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.bitwise_or`**: Addiert Arrays über logisches ODER. Steht bei einem von beiden Bildkanälen ein Pixel auf Weiß, bleibt es auf der resultierenden Matrix Weiß. Körper und Hand verschmelzen ohne Kollisionsprobleme.
- **`cv2.bilateralFilter(5, 50, 50)`**: Ein hochkomplexer, kantenerhaltender Filter. Ein `GaussianBlur` wischt unkontrolliert über alle Bildbereiche gleichermaßen. Der *Bilateral Filter* ignoriert harte Farbübergänge (Scharfe Kante zwischen Weißem Körper/Schwarzem Hintergrund) und glättet Störungen nur bei niedrigen Kontrastunterschieden in der Mitte des Körpers.
  - `5` bedeutet den räumlichen Abtastradius (Diameter).
  - `50` (SigmaColor): Bestimmt den Grauwertverlauf zwischen Punkten, bei dem noch geglättet wird.
  - `50` (SigmaSpace): Abstandslimit zweier physikalischer Koordinaten. 

## 8. Konturen füllen & Zeitliche Glättung

Wenn sich Arme im Videobild kreuzen, zieht MediaPipe manchmal falsche Grenzen, wodurch ein Loch zwischen den Armen und dem Bauch generiert wird. Eine Outline-Suche behebt das.

```python
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    blended = cv2.addWeighted(prev_mask, 0.25, final_mask, 0.75, 0)
```

**Erklärung der Funktionen & Parameter:**
- **`cv2.findContours`**:
  - `RETR_EXTERNAL`: Zieht im Array ein theoretisches Seil um die rein äußersten Koordinatengrenzen der Maske und vergisst vollständig alle in sich eingeschlossenen Hohlräume.
  - `CHAIN_APPROX_SIMPLE`: Komprimiert den endlos langen Konturstreifen mathematisch zu Start- und Endpunkten der Geraden (spart exponenziell Rechenzeit im Vergleich zur punktgenauen Matrixspeicherung).
- **`cv2.drawContours`**: Über das Target `-1` (alle Konturen anwenden) und `thickness=cv2.FILLED` (`-1`) malt die Engine den kompletten Raum innerhalb der Outline mit solidem Weiß (`255`) neu. Alle inneren Löcher sind somit unwiderruflich gefüllt.
- **`cv2.addWeighted(src1, alpha, src2, beta, gamma)`**: Ein Exponentiell gleitender Durchschnitt zur Simulation von zeitlicher Trägheit (Gegen Kantenflackern/Jittering/Rauschen). Verrechnet das alte Videobild (`prev_mask`) zu 25 % Gewichtigkeit (`alpha=0.25`) additiv mit dem Frame dieses Zyklus zu 75 % (`beta=0.75`). Das letzte Element definiert `Gamma` (`0`) zur additiven Helligkeitsabweichung (wird nicht angewandt). Das vermindert Sensorflimmern der Webcam extrem, erzeugt ab einem gewissen Wert aber visuelles Ghosting/Mitzieheffekte.

## 9. Dimensionstransformation für LED-Aspect

Zuletzt erfolgt das Frame-Conforming, um das 4:3 Webcam-Bild (640x480) für das asymmetrische, vertikal ausgerichtete Seitenverhältnis der Hardware-Matrix (32x48 Panel) fertigzustellen.

```python
    aspect = 32 / 48           # Ziel-Seitenverhältnis
    # ... Berechnung einer Sub-Position durch crop_w, crop_h ...
    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)
```

**Erklärung der Funktionen & Parameter:**
- **`aspect (32/48)`**: Vordefiniertes Zielproportionenverhältnis (0.66). Ein unbedachtes Stauchen (`Stretching`) der Sensor-Dimensionen an die serielle Schnittstelle ließe den Anwender lächerlich und völlig verformt aussehen.
- **`cv2.getRectSubPix`**: Extrahiert auf Subpixel-Ebene exakt die ausgerechneten Zielmaße (`crop_w, crop_h`) als Ausschnitt (`Cropping`). Dieser Ausschnitt geht genau orthogonal vom Mittelpunkt des Displays aus (`w // 2, h // 2`). Nutzer, die am Bildrand stünden, werden somit abgeschnitten, aber das zentrale Verhältnis am LED Display bleibt vollkommen unberührt.
- **`cv2.resize(..., interpolation=cv2.INTER_AREA)`**: Verkleinert das neu zugeschnittene Riesen-Array anspruchsgemäß auf seine winzigen, tatsächlichen 32x48 Endpixel. Die Modifikation `INTER_AREA` rechnet Interpolationen basierend auf Pixel-Flächenanteilen, statt wie handelsübliche Interpolationen auf linearen Punkten. Diese Technik ist maßgeblich auf aggressives und extremes Downsampling bei Matrizen gepolt, um Moiré-Zeichnungen oder den massiven Verlust von Formdetails auf Low-Resolution-Panels abzuwehren.
