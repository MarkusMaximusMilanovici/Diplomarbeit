import spidev, time


spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

# Initialize MAX7219
spi.writebytes([0x09, 0x00] * 4)  # Decode mode = 0 (no decode)
spi.writebytes([0x0A, 0x07] * 4)  # Intensity = 7 (~mid-brightness)
spi.writebytes([0x0B, 0x07] * 4)  # Scan limit = 8 digits
spi.writebytes([0x0C, 0x01] * 4)  # Shutdown = normal operation
spi.writebytes([0x0F, 0x01] * 4)  # Display test ON
time.sleep(3)
spi.writebytes([0x0F, 0x00] * 4)  # Display test OFF
spi.writebytes([0x0C, 0x00] * 4)  # Shutdown (optional)

spi.close()


in[20]
out[20]

for i in range(0, 10):
    in[i] = 1

for i in range(10, 20):
    in[i] = 0

for i in range(0, 20):
