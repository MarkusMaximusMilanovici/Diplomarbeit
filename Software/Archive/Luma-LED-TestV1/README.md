## Luma_Spidev_LEDTEST

### Test der LED-Matrix-Ansteuerung mit der Luma-Bibliothek

`Luma_Spidev_LEDTEST` wurde entwickelt, um die LED-Matrix mithilfe der Luma-Bibliothek anzusteuern.

Ziel dieses Programms war es, die grundlegende Kommunikation zwischen dem Raspberry Pi und der LED-Hardware zu testen.

**Funktionen dieses Tests:**

- Initialisierung der LED-Matrix  
- Ansteuerung über die SPI-Schnittstelle  
- Darstellung einfacher Muster oder Testbilder  
- Überprüfung der korrekten Funktion einzelner LED-Module  

Dieser Code diente ausschließlich als Hardware-Test und war ein wichtiger Schritt zur Sicherstellung, dass die Matrix korrekt angesteuert werden kann, bevor komplexere Bilddaten übertragen wurden.

Er bildet somit die Grundlage für die spätere Integration der Live-Bilddarstellung auf der LED-Matrix.