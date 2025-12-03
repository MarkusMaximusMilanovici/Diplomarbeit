from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from PIL import Image

# Initialisierung: 4 Blöcke in Kaskade
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=0, rotate=0)

# Helligkeit auf Maximum
device.contrast(255)

# Alle LEDs einschalten
img = Image.new('1', (32, 8), color=1)
device.display(img)

print("Alle 16 LEDs (4 Blöcke) sind jetzt AN mit voller Helligkeit!")
