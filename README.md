# Diplomarbeit
## Detection.py 
  Eine Body Detection oder Face Detection testen -> mit Vorlagen aus Github für Die Ganzen Erkennungen -> Funktioniert hauptsächlich aber die FrontalFace         funktioniert gut, jedoch unnötig aber war gut zum testen wie die Kamera so mithält

## DetectionV2.py
### Part 1 
  Ein Code für Edge detection mit Graycode -> Reagiert gut auf bewegungen aber macht im ganzen Bild eine Edge Detection und die ganzen Schatten im Bild werden    genommen was unötig ist weil wir nur die Sihlouette brauchen. Ein guter Durchbruch für den Anfang

### Part 2
  Ein Code der nur Edge Detection auf Bewegung macht und mit Errode und Dilate-> Reagiert gut auf Bewegung ohne Viel Delay aber die Kamera muss Fest sein. Aber   ohne Edge Detection und Errode und Dilate.

## main.py
  Die Funktion der Kamera getestet -> eingeschalten und gesehen wie die Kamera mit NoIR das ganze Aufnimmt und wie gut die Auflösung ist -> basic funktionen

## Detection_with_Mediapipe
  Die Personenerfassung und die wandlung auf schwarz und weiß mit KI model Mediapipe durch Errode und Dilate, KNN Subtractor, Kantenerkennung und Opening und Closing
  
## Detection_BinaryMask
  Die Personenerfassung mit einem Ascii MIrror Algorithm, wo alles weiße eine null ist und die PErson selber weiß
  
## Detection_with_Bodypix
  Das gleiche nur diesmal wird eine andere KI verwendet, welche auf andere Dinge spezialisiert ist
  
## Luma_Spidev_LEDTEST
  Mit der Luma library die Led-Matrizen ansteuern