Code for Dartmouth's Spot robot from 25F to implement Voice I/O, mapping, and improved HRI

## Connecting to Jetson via Serial Console

### Finding the Serial Device

On macOS, the Jetson will appear as a USB serial device. To find the device:

```bash
# List all USB serial devices
ls /dev/cu.usbmodem*

```

The device will typically be named something like `/dev/cu.usbmodem14217250298003` (the number will vary).

### Connecting via Serial Console

**Method 1: Using `screen` (Recommended)**

```bash
sudo screen /dev/cu.usbmodem<DEVICE_NUMBER> 115200
```

Replace `<DEVICE_NUMBER>` with the actual device number from the listing above.

### Exiting the Serial Console

- **screen**: Press `Ctrl+A`, then `K`, then `Y` to confirm
- **minicom**: Press `Ctrl+A`, then `X` to exit
- **cu**: Type `~.` (tilde followed by period) or press `Ctrl+D`

### Notes for Team Members

- Each team member can connect to the Jetson using the same serial console method
- The device path (`/dev/cu.usbmodem*`) may be different on each MacBook, so always check the device listing first
- You may need to disconnect and reconnect the USB cable if the device doesn't appear
- The baud rate is typically `115200` for Jetson devices

### Quick Reference

```bash
# Find device
ls /dev/cu.usbmodem*

# Connect (replace with your device number)
sudo screen /dev/cu.usbmodem14217250298003 115200

 python client_mic.py --device 24 --energy-mult 1.0
python src/voice_control/client_mic.py --list-devices
```
