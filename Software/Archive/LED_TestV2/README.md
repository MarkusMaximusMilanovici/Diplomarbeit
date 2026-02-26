## MAX7219_SPI_LED_Test.py

### Direkter LED-Matrix-Test über SPI mit dem MAX7219-Treiber

Dieses Programm dient als grundlegender Hardware-Test zur direkten Ansteuerung der LED-Matrix über die SPI-Schnittstelle des Raspberry Pi.

Im Gegensatz zu vorherigen Tests mit externen Bibliotheken erfolgt hier die Kommunikation direkt über das `spidev`-Modul. Dadurch wird ein besseres Verständnis für die Low-Level-Ansteuerung des MAX7219-LED-Treibers ermöglicht.

---

### Funktionsweise

- Öffnen der SPI-Schnittstelle (Bus 0, Device 0)
- Initialisierung des MAX7219-Treibers
- Konfiguration wichtiger Register:
  - Deaktivierung des Decode-Modus
  - Setzen der maximalen Helligkeit
  - Aktivierung aller 8 Reihen
  - Wechsel in den normalen Betriebsmodus
- Aktivierung des Display-Testmodus für 2 Sekunden
- Einschalten aller LEDs
- Zurücksetzen und Löschen der Anzeige

Das Programm sendet die Befehle gleichzeitig an alle vier kaskadierten LED-Module.

---

### Ziel des Tests

- Überprüfung der SPI-Kommunikation
- Kontrolle der korrekten Verkabelung
- Funktionstest aller LEDs
- Verständnis der MAX7219-Registerstruktur
- Sicherstellung, dass alle 4 Module korrekt synchronisiert sind

---

### Bedeutung für das Gesamtprojekt

Dieser Test stellte einen wichtigen Zwischenschritt dar, um sicherzustellen, dass die LED-Matrix zuverlässig angesteuert werden kann, bevor komplexe Bilddaten oder Live-Kamera-Feeds übertragen werden.

Er diente ausschließlich zur Hardwareverifikation und ist nicht Bestandteil der finalen Implementierung, bildet jedoch die Grundlage für die spätere visuelle Darstellung auf der Matrix.