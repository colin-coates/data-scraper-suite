# 🖥️ Desktop Observer

A cross-platform Python tool to observe, capture, and record your desktop screen.

## Features

- **Live Streaming**: View your desktop in real-time with FPS counter
- **Screenshots**: Capture single or continuous screenshots
- **Video Recording**: Record your desktop to MP4 video files
- **Multi-Monitor Support**: Select specific monitors or capture all
- **Keyboard Controls**: Interactive controls during streaming

## Installation

```bash
cd desktop_observer
pip install -r requirements.txt
```

### System Requirements

- **Windows**: Works out of the box
- **macOS**: May need screen recording permissions (System Preferences → Security & Privacy → Screen Recording)
- **Linux**: Requires X11 display server (Wayland may have limited support)

## Usage

### Live Desktop Stream

Stream your desktop in a live window:

```bash
# Default stream (all monitors, 50% scale)
python observer.py stream

# Stream specific monitor at full size
python observer.py stream -m 1 -s 1.0

# Stream at custom FPS
python observer.py stream -f 60
```

**Controls during streaming:**
- `q` - Quit
- `s` - Save screenshot
- `f` - Toggle FPS display
- `+` - Increase window size
- `-` - Decrease window size

### Screenshot Capture

```bash
# Single screenshot (auto-named)
python observer.py screenshot

# Screenshot with custom name
python observer.py screenshot -o my_screenshot.png

# Screenshot from specific monitor
python observer.py screenshot -m 2
```

### Video Recording

```bash
# Record until 'q' pressed
python observer.py record

# Record for 30 seconds
python observer.py record -d 30

# Record with custom output and FPS
python observer.py record -o video.mp4 -f 60
```

### Continuous Screenshots

```bash
# Screenshot every 1 second
python observer.py continuous

# Screenshot every 5 seconds to custom folder
python observer.py continuous -i 5 -o ~/Desktop/captures

# Capture exactly 10 screenshots
python observer.py continuous -c 10
```

### List Monitors

```bash
python observer.py monitors
```

## Python API Usage

```python
from observer import DesktopObserver

# Create observer
observer = DesktopObserver(monitor=1, fps=30)

# List monitors
observer.list_monitors()

# Capture screenshot
observer.capture_screenshot("screenshot.png")

# Start live stream
observer.stream(scale=0.5, show_fps=True)

# Record video
observer.record(output_path="recording.mp4", duration=10)
```

## Command Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `stream` | Live desktop display | `-m` monitor, `-s` scale, `-f` fps |
| `screenshot` | Single capture | `-o` output, `-m` monitor |
| `record` | Video recording | `-o` output, `-d` duration, `-f` fps |
| `continuous` | Periodic captures | `-i` interval, `-c` count, `-o` dir |
| `monitors` | List displays | - |

## Troubleshooting

### "No module named 'mss'" or similar
```bash
pip install -r requirements.txt
```

### Black screen on macOS
Grant screen recording permission:
1. Open System Preferences → Security & Privacy → Privacy
2. Select "Screen Recording" 
3. Add your terminal app or Python

### Display issues on Linux
Ensure you're using X11:
```bash
echo $XDG_SESSION_TYPE  # Should say "x11"
```

### High CPU usage
Lower the FPS:
```bash
python observer.py stream -f 15
```

## License

MIT License - Use freely for any purpose.
