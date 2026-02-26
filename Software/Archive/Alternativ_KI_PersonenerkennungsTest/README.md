## Detection_with_Bodypix

### Alternativer KI-Ansatz zur Personenerkennung

`Detection_with_Bodypix` stellt einen weiteren Entwicklungsversuch zur Personenerkennung dar, diesmal unter Verwendung eines anderen KI-Modells.

Im Unterschied zu den vorherigen Ansätzen wurde hier ein Modell eingesetzt, das speziell für die Segmentierung von Personen trainiert wurde. Ziel war es, eine präzisere Trennung zwischen Person und Hintergrund zu erreichen.

**Unterschiede zu vorherigen Implementierungen:**

- Verwendung eines anders trainierten KI-Modells  
- Fokus auf Personensegmentierung  
- Verbesserte Erkennung von Körperkonturen  
- Stabilere Ergebnisse bei komplexeren Hintergründen  

---

### Ziel des Experiments

- Vergleich unterschiedlicher KI-Modelle  
- Bewertung der Genauigkeit und Reaktionsgeschwindigkeit  
- Analyse der Eignung für die Live-Darstellung auf der LED-Matrix  

---

### Erkenntnisse

Durch die Nutzung eines spezialisierten Modells konnte die Person im Bild deutlich präziser vom Hintergrund getrennt werden.

Allerdings mussten auch hier folgende Faktoren berücksichtigt werden:

- Rechenleistung des Raspberry Pi  
- Echtzeitfähigkeit der Verarbeitung  
- Stabilität bei wechselnden Lichtverhältnissen  

Dieser Ansatz half dabei, die Vor- und Nachteile verschiedener KI-Modelle besser zu verstehen und eine fundierte Entscheidung für das finale System zu treffen.
