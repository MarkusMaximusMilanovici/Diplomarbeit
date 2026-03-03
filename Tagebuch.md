# Projektfortschritt – Black Mirror

## 1️⃣ LED-Matrix & Hardware

### September 2025
**24.09.2025**
- Entscheidung für Raspberry Pi statt Nucleo Board
- Bauteilrecherche (LED-Module, Kamera, Raspberry Pi)

### Oktober 2025
**01.10.2025**
- LEDs bestellt (günstigere Alternative gefunden)

**08.10.2025**
- Funktionsprüfung der Bauteile
- Raspberry OS auf USB installiert
- Fehlende Komponenten identifiziert (Netzteil 5V 5A, HDMI Adapter, Kühlung)

**09.10.2025**
- Analyse der LED-Module für Anschlussmechanismus


### November 2025
**05.11.2025**
- LED-Module getestet
- Defekte LEDs repariert

**19.11.2025**
- Erster Python-Code zur Ansteuerung einer LED-Matrix
- Installation eines LED-Matrix-Python-Packages

**20.11.2025**
- 4 LED-Matrix-Module zusammengelötet
- Ansteuerungscode weiterentwickelt

**26.11.2025**
- Erstmalige Anzeige eines Bildes auf 4 LED-Modulen


### Dezember 2025
**03.12.2025**
- Downscaling von Videos implementiert
- Video erfolgreich auf LED-Matrix dargestellt

**04.12.2025**
- OS auf Debian Bookworm 64 downgraded
- Bibliotheken global installiert
- Kamera um 90° gedreht
- Skalierung auf 72x128 Pixel geplant
- Skalierung auf 32x32 Pixel für Testmodule umgesetzt
- Erste Live-Kameraanzeige auf LED-Matrix

**10.12.2025**
- Blueprint für LED-Matrix-Aufbau gezeichnet

**17.12.2025**
- Verbindungslösung für LED-Module entwickelt


### Januar 2026
**04.01.2026**
- Alle LED-Module verbunden
- Raspberry Pi schafft 3 Reihen stabil

**05.01.2026**
- Netzteil reicht nur für 4 Reihen
- Helligkeit nimmt pro Reihe ab
- Lösungsideen:
  - Level-Shifter (3.3V → 5V)
  - Signal-Repeater
  - Separate Spannungseinspeisung pro Reihe

**14.–15.01.2026**
- Pin-Header gelötet
- Verbindungsplatine entworfen und bestellt

**28.–29.01.2026**
- Handdarstellung auf LED verbessert
- Problem: Rauschen bei mehreren Personen
- Maximal zwei Hände stabil erkennbar


### Februar 2026
**11.02.2026**
- Signal (CLK & Data In) mit Oszilloskop analysiert
- Signalverzerrung ab zweitem Modul festgestellt
- Planung einer Signalverstärker-Schaltung
- Design mit Schmitt-Trigger / Buffer / 30 Ohm Widerstand
- Level-Shifter als Lösung eingeplant

**12.02.2026**
- CLK, Chip Select und Vcc pro Reihe eingespeist
- 4 Reihen stabil, ab 5. Reihe instabil
- SPI0 und SPI1 getestet → nicht erfolgreich

**26.02.2026**
- Signalverstärker-Schaltung mit OPV (OPA354) entworfen
- Simulation in Multisim / LTSpice
- OPV bestellt


## 2️⃣ Personenerfassung & Bildverarbeitung

### Oktober 2025
**15.10.2025**
- Erste Ansätze des Erkennungscodes
- Verschiedene Libraries getestet

**16.10.2025**
- Erode & Dilate getestet
- Problem: Hintergrund wird teilweise erkannt


### November 2025
**13.11.2025**
- Silhouette verbessert
- Erode/Dilate optimiert (weniger Delay, detailliertere Konturen)

**19.11.2025**
- Kombination aus Bewegungs- und Edge-Detection getestet
- ASCII-Ansatz getestet (verworfen)
- KI-Ansatz getestet → bessere Silhouette

**20.11.2025**
- CLAHE-Filter eingefügt (Kontrast)
- Entfernt, da KI RGB benötigt

**26.11.2025**
- KNN-Subtractor mit 100 Frames Kalibrierung
- Beste Pipeline gefunden:
  - Grayscale
  - Erode & Dilate
  - Canny
  - Opening & Closing


### Dezember 2025
**03.–04.12.2025**
- MediaPipe mit Raspberry eingerichtet
- Virtual Environment Probleme gelöst
- PiCamera & MediaPipe kompatibel gemacht

**18.12.2025**
- Parameteroptimierung für Personenerkennung


### Januar 2026
**07.01.2026**
- Hand-Tracking (MediaPipe Hands) integriert
- Landmarkpunkte als Kreise dargestellt
- Zeitliche Glättung implementiert
- KNN-Bug (gedrehtes Bild) behoben

**28.01.2026**
- Landmarkpunkte mit Linien verbunden
- Handdarstellung verbessert

**29.01.2026**
- Arm/Körper-Trennung verbessert
- Rauschen erhöht bei mehreren Händen


## 3️⃣ Software & Systemintegration

### September 2025
- GitHub eingerichtet
- Projekttagebuch erstellt

### November 2025
- Branch für Bildübertragung erstellt
- Branch für Videoübertragung erstellt

### Dezember 2025
- OS-Downgrade für Library-Kompatibilität
- Globale Installation der Libraries
- Downscaling-Strategie implementiert

### Februar 2026
- GitHub komplett aufgeräumt
- Struktur der Diplomarbeit vorbereitet


## 4️⃣ Diplomarbeit

### November 2025
- Beginn der schriftlichen Arbeit

### Dezember 2025
- Dokumentation der bisherigen Fortschritte

### Februar 2026
- Inhaltsverzeichnis fertiggestellt
- Erste Kapitel strukturiert
- Technische Abschnitte begonnen