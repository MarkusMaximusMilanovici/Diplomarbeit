import spidev, time

spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

spi.xfer3([0x9, 0] * 4) # Decode 0
spi.xfer3([0xA, 0x7] * 4) # Intensity 15/32
spi.xfer3([0xB, 0x7] * 4) # Scan limit 8 digits
spi.xfer3([0xC, 1] * 4) # Shutdown Normal Operation
spi.xfer3([0xF, 1] * 4) # Display Test On
time.sleep(3)
spi.xfer3([0xF, 0] * 4) # Display Test Off
spi.xfer3([0xC, 0] * 4) # Shutdown
spi.close()
