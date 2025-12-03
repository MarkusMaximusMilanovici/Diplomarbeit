from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from PIL import Image

# Initialisierung mit Reset
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, block_orientation=-90, width=32, height=32)

# Device neu starten/clearen
device.clear()

# Helligkeit auf Maximum
device.contrast(255)

# Alle LEDs einschalten - RICHTIGE Größe für 32x32!
img = Image.new('1', (32, 32), color=1)
device.display(img)

print("Alle 16 Blöcke (32x32 LEDs) sind jetzt AN mit voller Helligkeit!")

while True:
    pass