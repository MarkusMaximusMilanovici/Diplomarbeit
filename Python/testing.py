import spidev, time

spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

def noop():
    spi.xfer3([0x0, 0] * 3)  # Decode 0

spi.xfer3([0x9, 0]) # Decode 0
noop()
spi.xfer3([0xA, 0x7]) # Intensity 15/32
noop()
spi.xfer3([0xB, 0x7]) # Scan limit 8 digits
noop()
spi.xfer3([0xC, 1]) # Shutdown Normal Operation
noop()
spi.xfer3([0xF, 1]) # Display Test On
noop()
time.sleep(3)
spi.xfer3([0xF, 0]) # Display Test Off
noop()
spi.xfer3([0xC, 0]) # Shutdown
noop()
spi.close()
