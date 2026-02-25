## Detection_BinaryMask

### Experimenteller Ansatz basierend auf dem ASCII-Spiegel-Prinzip

`Detection_BinaryMask` entstand als experimentelle Idee, inspiriert vom Konzept eines sogenannten „ASCII-Spiegels“. Ziel dieses Ansatzes war es, eine stark vereinfachte visuelle Darstellung der erkannten Person zu erzeugen.

Anstatt Graustufen oder detaillierte Kanteninformationen darzustellen, wurde das Bild auf eine binäre Maske reduziert.

**Grundprinzip:**

- Der Hintergrund wird als „0“ dargestellt  
- Die erkannte Person wird als „1“ dargestellt  
- Es entsteht eine kontrastreiche Schwarz-Weiß-Darstellung  
- Die Person erscheint als weiße Silhouette vor dunklem Hintergrund  

Dieser Ansatz orientiert sich am Prinzip eines ASCII-Mirrors, bei dem Bildinformationen auf einfache Zeichen oder binäre Werte reduziert werden.

---

### Ziel des Experiments

- Vereinfachung der Bilddaten  
- Reduktion von Detailinformationen  
- Optimierung für eine spätere Darstellung auf einer niedrig auflösenden LED-Matrix  
- Fokus auf die reine Silhouette der Person  

---

### Erkenntnisse

Die binäre Darstellung erwies sich als grundsätzlich geeignet für eine reduzierte Visualisierung. Allerdings zeigte sich, dass:

- die Erkennung stark von Lichtverhältnissen abhängig war  
- Hintergrundstrukturen teilweise fälschlicherweise als Person erkannt wurden  
- eine präzisere Segmentierung notwendig ist  

Trotzdem stellte dieser Ansatz einen wichtigen kreativen Zwischenschritt dar und half dabei, geeignete Darstellungsformen für die LED-Matrix zu evaluieren.