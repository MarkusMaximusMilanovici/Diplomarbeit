import cv2
import numpy as np
import os
import sys

# Grid-Größe
width, height = 120, 80

# Hintergrund-Grid mit "0"
grid = np.full((height, width), "0")

# Kamera öffnen
cap = cv2.VideoCapture(0)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Horizontale Spiegelung
        gray = cv2.flip(gray, 1)  # 1 = horizontal spiegeln

        # Auf Grid-Größe skalieren
        small_gray = cv2.resize(gray, (width, height))

        # Threshold setzen, um die Person zu erkennen
        mask = small_gray > 100  # hellere Pixel = Person

        # Grid kopieren und Maske anwenden
        display_grid = grid.copy()
        display_grid[mask] = "#"  # Person = #

        # Cursor an Anfang setzen, Grid ausgeben
        print("\033[H", end="")
        for row in display_grid:
            print("".join(row))
        sys.stdout.flush()

except KeyboardInterrupt:
    pass

cap.release()
