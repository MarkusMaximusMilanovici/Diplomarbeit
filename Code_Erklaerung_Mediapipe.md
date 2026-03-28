# Erklärung des Moduls `Detection_with_Mediapipe.py`

In diesem Kapitel wird die Funktionsweise und Architektur des zentralen Bildverarbeitungsskripts `Detection_with_Mediapipe.py` im Detail erläutert. Das Skript ist dafür verantwortlich, den Benutzer vor der Kamera zu erkennen, den Hintergrund freizustellen (Silhouette zu extrahieren) und diese für die Darstellung auf einer 32x48-LED-Matrix aufzubereiten. Hierbei kommen Computer-Vision-Techniken mittels *OpenCV* sowie vortrainierte Machine-Learning-Modelle aus dem *MediaPipe*-Framework zum Einsatz.

## 1. Kamera-Architektur und Setup

Um eine hohe Flexibilität während der Entwicklungs- und Einsatzphase zu gewährleisten, implementiert das System eine dynamische Kameraauswahl.

```python
# ============================================================
# Kamera-Auswahl: Raspberry Pi (PiCamera2) ODER Laptop (cv2.VideoCapture)
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
```

**Erklärung der Architektur und Parameter:**
Die Architektur ist so ausgelegt, dass das Programm sowohl auf einem Entwicklungs-Laptop als auch auf der finalen Hardware (Raspberry Pi) ohne Codeänderungen ausgeführt werden kann. Über einen `try-except`-Block wird geprüft, ob die Bibliothek `picamera2` verfügbar ist. 
Die Auflösung wird bewusst auf 640x480 Pixel limitiert. Diese Auflösung stellt einen optimalen Kompromiss aus Performance und Genauigkeit dar. Einerseits reicht sie völlig aus, um Konturen für das spätere, extrem niedrig aufgelöste Matrix-Display (32x48 Pixel) zu generieren, andererseits spart sie erhebliche Rechenzeit bei der Verarbeitung durch die neuronalen Netze, wodurch eine flüssige Framerate für Interaktionen in Echtzeit erzielt wird. Das Farbformat wird als `BGR888` spezifiziert, da OpenCV standardmäßig im BGR-Farbraum arbeitet.

## 2. Initialisierung der Machine-Learning-Modelle

Für die Generierung der Silhouette werden Modelle der Google MediaPipe-Bibliothek genutzt.

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

**Erklärung der Architektur und Parameter:**
- **SelfieSegmentation (`model_selection=1`):** Das Modell `1` (`landscape` / general purpose) ist für schnelle und effiziente Inferenzen in Echtzeit optimiert. Da bei einem digitalen Spiegel-Effekt eine verzögerungsfreie Darstellung eine übergeordnete Rolle für die Usability spielt, wurde dieses schnelle Modell bevorzugt. 
- **Hands:** Dieses Modul wird ergänzend eingesetzt, da schnelle oder komplexe Handbewegungen von der generellen Körpersilhouette häufig nicht präzise genug erfasst werden. Parameter wie `min_detection_confidence=0.5` und `min_tracking_confidence=0.6` wurden empirisch abgewogen, um zu schnelle Tracking-Verluste zu vermeiden, ohne jedoch zu anfällig für Fehl-Erkennungen (False Positives) zu sein. `static_image_mode=False` wird verwendet, da es sich um einen kontinuierlichen Videostream handelt und die Einbeziehung temporaler Daten (historischer Verlauf der Punkte über die vergangenen Frames) zu einer deutlich glatteren Verfolgung führt.
- Der **BackgroundSubtractorKNN** dient als redundantes oder erweiterbares Modul, um eine klassische Differenzbilderfassung als Referenz oder Fallback bereitzustellen. 

## 3. Kamerakalibrierung und Bildaufhellung

Zu Beginn der Laufzeit wird eine kurze Kalibrierungsphase durchlaufen, um auf verschiedene Belichtungsszenarien vorbereitet zu sein.

```python
# === Kalibrierungsphase für den Hintergrund ===
calibration_frames = 100

for i in range(calibration_frames):
    ret, frame = get_frame(cam)
    
    # ...
    alpha_bright = 1.3
    beta_bright = 20
    frame_enh = cv2.convertScaleAbs(frame, alpha=alpha_bright, beta=beta_bright)
    # ...
```

**Erklärung der Architektur und Parameter:**
Umgebungsbedingungen wie z. B. weite Entfernungen oder die Raumbeleuchtung haben drastischen Einfluss auf die Segmentierungsqualität der KI-Modelle. Über die Parameter `alpha_bright=1.3` (Erhöhung des Kontrasts um 30%) und `beta_bright=20` (konstantes Aufhellen um 20) wird das Bild künstlich vorkorrigiert. Diese Werte schaffen robuste Bedingungen, weil unterbelichtete Bilder erfahrungsgemäß für Aussetzer im Neural Network sorgen. Zusätzlich lernt das System über 100 Frames hinweg in der Schleife das statische Rauschen im Raum kennen, um dieses in den iterativen Abzweigungen zu filtern.

## 4. Generierung der KI-Körpermaske

In der Hauptschleife *(Main Loop)* werden für jedes Frame die KI-Ergebnisse ausgewertet und eine erste Matrix (Binärmaske) erstellt.

```python
    # MediaPipe Person Segmentierung
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    ki_mask = (res.segmentation_mask > 0.4).astype(np.uint8) * 255
```

**Erklärung der Architektur und Parameter:**
MediaPipe verarbeitet ausschließlich Bilder im RGB-Spektrum, weshalb eine vorbereitende Farbraumkonvertierung erfolgen muss. Die Variable `res.segmentation_mask` enthält nach erfolgreichem Durchlauf für jeden Pixel einen Konfidenzwert zwischen 0 und 1. Ein entscheidender Design-Faktor liegt hier im Schwellenwert (Threshold) **`0.4`**. Ein konventioneller und strengerer Wert von 0.5 würde regelmäßig Haarsträhnen oder weite Kleidung abschneiden. Die Reduktion auf 0.4 dehnt den Akzeptanzbereich marginal aus, wodurch die Randbereiche sauberer mit in die endgültige Maske aufgenommen werden. Das dabei auftretende leichte Kanten-Rauschen wird in den späteren Filterschritten problemlos mitigiert.

## 5. Hand-Tracking und geometrische Konstruktion

Einer der technisch anspruchsvollsten Aspekte des Skripts ist die Generierung der kompakten Hände. MediaPipe liefert nur diskrete Stützpunkte (Landmarks), also ein Punktwolkenskelett ohne Raumfülle.

```python
        for handLms in hand_res.multi_hand_landmarks:
            # Sammle alle 21 Hand-Punkte in ein Array (hand_points) ...

            # 1. Fülle die Handfläche als Polygon, um Löcher zu vermeiden
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_pts = np.array([hand_points[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(hand_mask, [palm_pts], 255)

            # 2. Verbindungen der Finger und Handkanten
            finger_thickness = 18 
            for connection in connections: # ... Liste aus Punktverbindungen
                pt1 = tuple(hand_points[connection[0]])
                pt2 = tuple(hand_points[connection[1]])
                cv2.line(hand_mask, pt1, pt2, 255, thickness=finger_thickness)

            # 3. Kreise um alle Gelenke für bündige Abschlüsse
            for point in hand_points:
                cv2.circle(hand_mask, tuple(point), finger_thickness // 2, 255, -1)
```

**Erklärung der Architektur und Parameter:**
Zur Erzeugung einer massiven Hand auf dem Display erfolgen hier drei Schritte der geometrischen Interpolation:
1. **Die Handfläche (Palm):** Mit den Indizes 0, 1, 5, 9, 13 und 17 werden die Eckpunkte der Handfläche ermittelt und ein massiv gefülltes Polygon gezeichnet (`fillPoly`). Dies verhindert effektiv ungewollte Transparenzen oder "Löcher" in der Bildschirmdarstellung.
2. **Die Finger:** Zwischen allen detektierten Gelenk-Indices werden gerade Linien verlegt. Die Liniendicke von `18` Pixeln wurde gewählt, weil das im Verhältnis zu der Basisauflösung von 640x480 optisch am ehesten dem Breitenverhältnis eines natürlichen Fingers entspricht. 
3. **Gelenk-Schließung:** Ein gerader Strich zwischen zwei Punkten erzeugt bei einer Winkelung der Finger oft eine kantige Ausbuchtung am Gelenk. Durch das explizite Zeichnen eines voll ausgefüllten Kreises genau um den Gelenkknoten (`thickness // 2`) entstehen bündige, weiche Abrundungen und glatte Übergänge an den Knöcheln.

## 6. Verfeinerung durch Matrix-Morphologie

Um die geometrisch konstruierten Hände organischer und die gesamte Bildmaske nahtloser wirken zu lassen, müssen Filter-Operationen auf der Maske angewendet werden.

```python
    if np.any(hand_mask > 0):
        # Moderate Dilatation für zusammenhängende Hand
        kernel_medium = np.ones((5, 5), np.uint8) 
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=1)

        # Moderater Blur für organische Form
        hand_mask = cv2.GaussianBlur(hand_mask, (9, 9), 0)
        _, hand_mask = cv2.threshold(hand_mask, 120, 255, cv2.THRESH_BINARY)

        # Leichte Erosion für realistische Größe
        kernel_small = np.ones((3, 3), np.uint8) 
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)

    # Hand-Maske mit genereller KI-Maske verknüpfen
    ki_mask = cv2.bitwise_or(ki_mask, hand_mask)

    # ... Finale Glättung der KI-Maske gegen Rauschen ...
    final_mask = cv2.bilateralFilter(ki_mask.copy(), 5, 50, 50)
```

**Erklärung der Architektur und Parameter:**
1. **Dilatation (`5x5` Kern):** Dieser Schritt verbreitert alle weißen Flächen leicht, wodurch mikroskopische Trennungen und Lücken in Handlinien überbrückt werden.
2. **Gaussian Blur (`9x9`) & Thresholding:** Starke Winkelstrukturen werden durch den Unschärfefilter "verwässert" bzw. gerundet. Der Schwellenwert `120` verwandelt den grauen Gradienten am Maskenrand zurück ins tiefste Weiß oder Schwarz, wodurch organische Kurvenverläufe anstatt Treppenstufen entstehen.
3. **Erosion (`3x3`):** Weil die Dilatation die Hände unproportional aufgebläht hat, schält die Erosion die überschüssigen Bildpixel wieder ab – zurück bleibt eine realitätsgetreue Silhouettendarstellung.
Als letztes Mittel in Bezug auf das Rauschen operiert ein **Bilateral Filter** (`5, 50, 50`). Gegenüber einem normalen Gaußschen Matrix-Blur hat dieser den entscheidenden Vorteil, Bildrauschen und Störpixel (z.B. flimmernde Graustufen) intensiv zu glätten, während starke Kantengradienten (die Konturlinien der Figur) komplett intakt bleiben.

## 7. Zeitliche Glättung (Temporal Smoothing)

Algorithmische Unschärfe auf dem Kamerabild verursacht besonders an den Rändern einer Silhouette Instabilität ("Flimmern"). Dem wird über das Frame-Blending entgegengewirkt.

```python
    # ===== Zeitliche Glättung der gesamten Maske =====
    alpha = 0.25
    if prev_mask is None:
        prev_mask = final_mask.copy()
    else:
        blended = cv2.addWeighted(prev_mask, alpha, final_mask, 1 - alpha, 0)
        _, blended = cv2.threshold(blended, 150, 255, cv2.THRESH_BINARY)
        prev_mask = blended
        final_mask = blended
```

**Erklärung der Architektur und Parameter:**
Hier nutzt das System einen simplen Exponentiellen Glättungsansatz (Exponential Moving Average). Das aktuell generierte Videobild fließt mit nur `25%` Gewichtung (`alpha = 0.25`) in das Endbild ein, während das Bild aus der Vergangenheit zu `75%` (`1 - alpha`) dominiert. Dieses künstliche "Trägheitsmoment" schluckt Fehler-Frames in Sekundenbruchteilen oder reduziert Flimmern drastisch. Zwar tritt hierdurch ein leichtes Nachziehen (Ghosting-Effekt) der Arme ein, für den Anwendungszweck auf der LED-Matrix erwies sich `0.25` nach empirischen Tests aber als die beste Balance zwischen visueller Beruhigung und Reaktivität des Systems. (Für die reine Hand wurde im Skript zusätzlich der separate Wert `alpha_hand = 0.35` definiert, was dazu führt, dass sich Hände im Gesamtsystem ein bisschen schneller und reaktiver anfühlen als der restliche, eher statische Oberkörper).

## 8. Aspect-Ratio-Korrektur und LED-Matrix Export

Zuletzt muss das Bild des Computers auf das proprietäre Format der Zielhardware übertragen werden.

```python
    # Variante 1: Mit getRectSubPix (einfachster Weg)
    h, w = out_full.shape[:2]
    aspect = 32 / 48
    if w / h > aspect:
        crop_w = int(h * aspect)
        crop_h = h
    else:
        crop_w = w
        crop_h = int(w / aspect)

    cropped = cv2.getRectSubPix(out_full, (crop_w, crop_h), (w // 2, h // 2))
    out_small = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)

    ImagetoMatrix.drawImage(out_small)
```

**Erklärung der Architektur und Parameter:**
Ein wesentliches Problem der Transformation besteht darin, dass das Kamera-Ausgangsmaterial normalerweise ein Seitenverhältnis (Aspect Ratio) von ca. 4:3 (640x480) aufweist, während die physikalische LED-Wand auf hochkantes Format (32x48) im klassischen 2:3 Seitenverhältnis ausgelegt ist.
Ein unbedachtes Reskalieren (Stauchen) würde zu einer unnatürlichen Verzerrung führen, durch die Benutzer stark deformiert erscheinen. Die Berechnungslogik evaluiert das relative Verhältnis beider Medien zueinander und kalkuliert einen maximalen zentrierten Zuschnitt (`crop_w`, `crop_h`). Durch `cv2.getRectSubPix` wird der berechnete, nicht verzerrte Mittenausschnitt der Figur extrahiert. 
Dieses Teilbild wird anschließend durch `cv2.resize` an die Zielgröße der Matrix angepasst. Auffällig ist der Gebrauch von `cv2.INTER_AREA`. Dieser Interpolations-Algorithmus ist nicht auf Upscaling oder Performance ausgelegt, sondern erzielt signifikant die saubersten und artefakt-freiesten Ergebnisse beim massiven Herunterskalieren (Rendern auf Sub-Pixel-Level), bevor die verarbeiteten Dimensionen seriell an das Steuerungsmodul `ImagetoMatrix` gesendet werden.
