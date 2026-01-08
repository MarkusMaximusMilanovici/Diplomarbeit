# Diplomarbeit

## LINK für Docs
https://docs.google.com/document/d/1WH5tCRd0Api436s9G5jYEZzmVug9cEUj/edit

## Detection.py 
  Eine Body Detection oder Face Detection testen -> mit Vorlagen aus Github für Die Ganzen Erkennungen -> Funktioniert hauptsächlich aber die FrontalFace         funktioniert gut, jedoch unnötig aber war gut zum testen wie die Kamera so mithält

## DetectionV2.py
### Dieser Code war der Erste wirkliche Versuch die Personenerkennung zu machen. Dieser Code wurde nicht für das Endergebnis verwendet, aber es war sehr wichtig für das Verständnis von verschiedenen Algorithmen und für die Realisierung, dass die Personenerkennung ohne Ki nur mit Nachteile mitkommt.
#### Part 1 
  Ein Code für Edge detection mit Graycode -> Reagiert gut auf bewegungen aber macht im ganzen Bild eine Edge Detection und die ganzen Schatten im Bild werden    genommen was unötig ist weil wir nur die Sihlouette brauchen. Ein guter Durchbruch für den Anfang

#### Part 2
  Ein Code der nur Edge Detection auf Bewegung macht und mit Errode und Dilate-> Reagiert gut auf Bewegung ohne Viel Delay aber die Kamera muss Fest sein. Aber   ohne Edge Detection und Errode und Dilate.

## main.py
### Dieser Code dient nur zum Testen der Kamera mit dem RAspberry
  Die Funktion der Kamera getestet -> eingeschalten und gesehen wie die Kamera mit NoIR das ganze Aufnimmt und wie gut die Auflösung ist -> basic funktionen

## Detection_with_Mediapipe
### Dieser Code wurde für die Personenerkennung des Balck Mirrors endgültig verwendet 
  Die Personenerfassung und die wandlung auf schwarz und weiß mit KI model Mediapipe durch Errode und Dilate, KNN Subtractor, Kantenerkennung und Opening und Closing
  
## Detection_BinaryMask
### Dieser Code war nur eine Idee basierend auf dem Ascii Spiegel, d.h das nur die Persone in Ascii "1" dargestellt wird
  Die Personenerfassung mit einem Ascii Mirror Algorithm, wo alles weiße eine null ist und die Person selber weiß
  
## Detection_with_Bodypix
### Dies ist noch ein Versuch nur diesemal mit einer anderen Ki die auch anders Trainiert wurde, Personen zu erkenne
  Das gleiche nur diesmal wird eine andere KI verwendet, welche auf andere Dinge spezialisiert ist
  
## Luma_Spidev_LEDTEST
  Mit der Luma library die Led-Matrizen ansteuern
