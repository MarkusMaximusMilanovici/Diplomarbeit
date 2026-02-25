## DetectionV2.py

### Erste Implementierung der Personenerkennung

`DetectionV2.py` stellt den ersten ernsthaften Entwicklungsversuch zur Umsetzung einer Personenerkennung dar.  
Dieser Code wurde nicht im finalen System verwendet, spielte jedoch eine zentrale Rolle im Entwicklungsprozess.

Er diente insbesondere:
- dem Verständnis grundlegender Bildverarbeitungsalgorithmen  
- der praktischen Auseinandersetzung mit Bewegungs- und Kantenerkennung  
- der Erkenntnis, dass eine zuverlässige Personenerkennung ohne KI-basierte Verfahren mit deutlichen Einschränkungen verbunden ist  

Das Programm ist in zwei zentrale Entwicklungsabschnitte gegliedert.

---

### Part 1 – Klassische Kantenerkennung (Edge Detection)

Im ersten Ansatz wurde eine Kantenerkennung auf Basis eines Graustufenbildes implementiert.

**Funktionsweise:**
- Umwandlung des Kamerabildes in ein Graustufenbild  
- Anwendung einer Edge-Detection zur Hervorhebung von Kanten  
- Darstellung aller im Bild vorhandenen Konturen  

**Ergebnisse und Erkenntnisse:**

Dieser Ansatz reagierte gut auf Bewegungen im Bild und stellte eine funktionierende Möglichkeit zur visuellen Strukturierung dar.

Allerdings zeigten sich folgende Nachteile:
- Die Kantenerkennung wurde auf das gesamte Bild angewendet  
- Schattenbereiche wurden ebenfalls als Kanten interpretiert  
- Viele irrelevante Bilddetails wurden verarbeitet  
- Es entstand visuelles „Rauschen“  

Für das Ziel – die Darstellung einer klaren Personensilhouette – war dieser Ansatz daher nur bedingt geeignet.

Trotzdem stellte dieser Schritt einen wichtigen Durchbruch dar, da er das grundlegende Verständnis der Bildverarbeitung vertiefte.

---

### Part 2 – Bewegungsbasierte Kantenerkennung mit morphologischen Operationen

Im zweiten Entwicklungsabschnitt wurde der Fokus stärker auf Bewegungserkennung gelegt.

**Erweiterungen:**
- Edge Detection nur bei erkannter Bewegung  
- Anwendung von „Erode“ und „Dilate“ zur Bildbereinigung  
- Reduzierung von Störpixeln  
- Verbesserung der Silhouettenbildung  

**Ergebnisse und Erkenntnisse:**

Dieser Ansatz reagierte schnell auf Bewegungen und verursachte nur geringe Verzögerungen.

Durch die zusätzlichen Bildoperationen konnte das Bild stabilisiert und Rauschen reduziert werden.

Dennoch ergaben sich weiterhin Einschränkungen:
- Die Kamera musste vollständig stabil montiert sein  
- Bereits kleine Kamerabewegungen führten zu Fehlinterpretationen  
- Ohne weiterführende Verfahren blieb die Erkennung unpräzise  

---

### Fazit

`DetectionV2.py` war ein wichtiger Meilenstein im Entwicklungsprozess.

Er zeigte deutlich, dass klassische Bildverarbeitung ohne KI-Unterstützung zwar grundsätzlich funktioniert, jedoch erhebliche Nachteile hinsichtlich Genauigkeit und Robustheit mit sich bringt.

Diese Erkenntnisse führten schließlich zur Entscheidung, modernere Verfahren für die finale Personenerkennung einzusetzen.