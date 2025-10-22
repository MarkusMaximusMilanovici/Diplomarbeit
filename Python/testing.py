import spidev, time

spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

def noop():
    spi.xfer2([0x0, 0] * 3)  # Decode 0

spi.xfer2([0x9, 0]) # Decode 0
noop()
spi.xfer2([0xA, 0x7]) # Intensity 15/32
spi.xfer2([0xB, 0x7]) # Scan limit 8 digits
spi.xfer2([0xC, 1]) # Shutdown Normal Operation
spi.xfer2([0xF, 1]) # Display Test On
time.sleep(3)
spi.xfer2([0xF, 0]) # Display Test Off
spi.xfer2([0xC, 0]) # Shutdown
spi.close()
