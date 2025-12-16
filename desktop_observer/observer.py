#!/usr/bin/env python3
"""
Desktop Observation Tool
========================
A cross-platform tool to observe and capture your desktop screen.

Features:
- Live desktop streaming with real-time display
- Screenshot capture (single or continuous)
- Multi-monitor support
- Configurable frame rate and quality

Usage:
    python observer.py stream          # Live desktop stream
    python observer.py screenshot      # Single screenshot
    python observer.py record          # Record to video file
    python observer.py --help          # Show all options
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np
from PIL import Image


class DesktopObserver:
    """Observes and captures desktop screen content."""

    def __init__(self, monitor: int = 0, fps: int = 30):
        """
        Initialize the desktop observer.

        Args:
            monitor: Monitor index (0 = all monitors, 1 = first, 2 = second, etc.)
            fps: Target frames per second for streaming/recording
        """
        self.sct = mss.mss()
        self.monitor = monitor
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self._running = False

    @property
    def monitors(self) -> list:
        """Get list of available monitors."""
        return self.sct.monitors

    def get_monitor_info(self) -> dict:
        """Get information about the selected monitor."""
        monitors = self.monitors
        if self.monitor == 0:
            return monitors[0]  # All monitors combined
        elif self.monitor < len(monitors):
            return monitors[self.monitor]
        else:
            print(f"Monitor {self.monitor} not found. Using primary monitor.")
            return monitors[1] if len(monitors) > 1 else monitors[0]

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the desktop.

        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        monitor = self.get_monitor_info()
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array (BGRA format from mss)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        return frame

    def capture_screenshot(self, output_path: Optional[str] = None) -> str:
        """
        Capture a single screenshot and save to file.

        Args:
            output_path: Path to save the screenshot. If None, auto-generates filename.

        Returns:
            Path to the saved screenshot
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"screenshot_{timestamp}.png"

        frame = self.capture_frame()
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.save(output_path)
        
        print(f"📸 Screenshot saved: {output_path}")
        return output_path

    def stream(self, scale: float = 1.0, show_fps: bool = True):
        """
        Stream the desktop in a live window.

        Args:
            scale: Scale factor for the display window (0.5 = half size)
            show_fps: Whether to display FPS counter on the stream

        Controls:
            - Press 'q' to quit
            - Press 's' to save a screenshot
            - Press 'f' to toggle FPS display
            - Press '+' to increase scale
            - Press '-' to decrease scale
        """
        print("🖥️  Desktop Stream Starting...")
        print("Controls:")
        print("  [q] Quit")
        print("  [s] Save screenshot")
        print("  [f] Toggle FPS display")
        print("  [+] Increase window size")
        print("  [-] Decrease window size")
        print("-" * 40)

        window_name = "Desktop Observer - Press 'q' to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self._running = True
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0

        try:
            while self._running:
                loop_start = time.time()

                # Capture frame
                frame = self.capture_frame()

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                # Apply scale
                if scale != 1.0:
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Draw FPS if enabled
                if show_fps:
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(
                        frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Stopping stream...")
                    self._running = False
                elif key == ord('s'):
                    self.capture_screenshot()
                elif key == ord('f'):
                    show_fps = not show_fps
                    print(f"FPS display: {'ON' if show_fps else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    scale = min(scale + 0.1, 2.0)
                    print(f"Scale: {scale:.1f}x")
                elif key == ord('-'):
                    scale = max(scale - 0.1, 0.1)
                    print(f"Scale: {scale:.1f}x")

                # Frame rate control
                elapsed = time.time() - loop_start
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)

        except KeyboardInterrupt:
            print("\n👋 Stream interrupted by user")
        finally:
            cv2.destroyAllWindows()

    def record(
        self,
        output_path: Optional[str] = None,
        duration: Optional[float] = None,
        codec: str = "mp4v"
    ):
        """
        Record the desktop to a video file.

        Args:
            output_path: Path for the output video file
            duration: Recording duration in seconds (None = until 'q' pressed)
            codec: Video codec (mp4v, XVID, etc.)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recording_{timestamp}.mp4"

        monitor = self.get_monitor_info()
        width = monitor["width"]
        height = monitor["height"]

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print(f"🎬 Recording started: {output_path}")
        print(f"   Resolution: {width}x{height} @ {self.fps} FPS")
        if duration:
            print(f"   Duration: {duration} seconds")
        print("   Press 'q' to stop recording")
        print("-" * 40)

        window_name = "Recording - Press 'q' to stop"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width // 2, height // 2)

        self._running = True
        start_time = time.time()
        frame_count = 0

        try:
            while self._running:
                loop_start = time.time()

                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print("\n⏱️  Duration limit reached")
                    break

                # Capture and write frame
                frame = self.capture_frame()
                out.write(frame)
                frame_count += 1

                # Show preview (scaled down)
                preview = cv2.resize(frame, (width // 2, height // 2))
                elapsed = time.time() - start_time
                cv2.putText(
                    preview, f"REC {elapsed:.1f}s | Frames: {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                cv2.imshow(window_name, preview)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Recording stopped by user")
                    break

                # Frame rate control
                elapsed = time.time() - loop_start
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)

        except KeyboardInterrupt:
            print("\n⏹️  Recording interrupted")
        finally:
            out.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n✅ Recording saved: {output_path}")
            print(f"   Total frames: {frame_count}")
            print(f"   Duration: {total_time:.1f}s")
            print(f"   Actual FPS: {frame_count / total_time:.1f}")

    def continuous_screenshots(
        self,
        output_dir: str = "screenshots",
        interval: float = 1.0,
        count: Optional[int] = None
    ):
        """
        Capture screenshots at regular intervals.

        Args:
            output_dir: Directory to save screenshots
            interval: Seconds between screenshots
            count: Number of screenshots (None = until Ctrl+C)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"📸 Continuous screenshot capture")
        print(f"   Output: {output_dir}/")
        print(f"   Interval: {interval}s")
        if count:
            print(f"   Count: {count}")
        print("   Press Ctrl+C to stop")
        print("-" * 40)

        captured = 0
        try:
            while count is None or captured < count:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = f"{output_dir}/screenshot_{timestamp}.png"
                self.capture_screenshot(output_path)
                captured += 1
                
                if count is None or captured < count:
                    time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n✅ Captured {captured} screenshots")

    def list_monitors(self):
        """Display information about available monitors."""
        print("🖥️  Available Monitors:")
        print("-" * 40)
        for i, mon in enumerate(self.monitors):
            if i == 0:
                label = "All monitors combined"
            else:
                label = f"Monitor {i}"
            print(f"  [{i}] {label}")
            print(f"      Position: ({mon['left']}, {mon['top']})")
            print(f"      Size: {mon['width']}x{mon['height']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Desktop Observation Tool - Capture and stream your desktop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python observer.py stream              # Live desktop stream
  python observer.py stream -s 0.5       # Stream at half size
  python observer.py screenshot          # Single screenshot
  python observer.py record -d 30        # Record 30 seconds
  python observer.py continuous -i 5     # Screenshot every 5 seconds
  python observer.py monitors            # List available monitors
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Live desktop stream")
    stream_parser.add_argument(
        "-m", "--monitor", type=int, default=0,
        help="Monitor index (0=all, 1=first, 2=second, etc.)"
    )
    stream_parser.add_argument(
        "-s", "--scale", type=float, default=0.5,
        help="Display scale factor (default: 0.5)"
    )
    stream_parser.add_argument(
        "-f", "--fps", type=int, default=30,
        help="Target FPS (default: 30)"
    )
    stream_parser.add_argument(
        "--no-fps", action="store_true",
        help="Hide FPS counter"
    )

    # Screenshot command
    shot_parser = subparsers.add_parser("screenshot", help="Capture screenshot")
    shot_parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path"
    )
    shot_parser.add_argument(
        "-m", "--monitor", type=int, default=0,
        help="Monitor index"
    )

    # Record command
    rec_parser = subparsers.add_parser("record", help="Record desktop to video")
    rec_parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path"
    )
    rec_parser.add_argument(
        "-d", "--duration", type=float, default=None,
        help="Recording duration in seconds"
    )
    rec_parser.add_argument(
        "-m", "--monitor", type=int, default=0,
        help="Monitor index"
    )
    rec_parser.add_argument(
        "-f", "--fps", type=int, default=30,
        help="Recording FPS"
    )

    # Continuous screenshots command
    cont_parser = subparsers.add_parser(
        "continuous", help="Continuous screenshot capture"
    )
    cont_parser.add_argument(
        "-o", "--output-dir", type=str, default="screenshots",
        help="Output directory"
    )
    cont_parser.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="Interval between screenshots (seconds)"
    )
    cont_parser.add_argument(
        "-c", "--count", type=int, default=None,
        help="Number of screenshots to capture"
    )
    cont_parser.add_argument(
        "-m", "--monitor", type=int, default=0,
        help="Monitor index"
    )

    # List monitors command
    subparsers.add_parser("monitors", help="List available monitors")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Create observer
    monitor = getattr(args, 'monitor', 0)
    fps = getattr(args, 'fps', 30)
    observer = DesktopObserver(monitor=monitor, fps=fps)

    # Execute command
    if args.command == "stream":
        observer.stream(scale=args.scale, show_fps=not args.no_fps)
    
    elif args.command == "screenshot":
        observer.capture_screenshot(args.output)
    
    elif args.command == "record":
        observer.record(output_path=args.output, duration=args.duration)
    
    elif args.command == "continuous":
        observer.continuous_screenshots(
            output_dir=args.output_dir,
            interval=args.interval,
            count=args.count
        )
    
    elif args.command == "monitors":
        observer.list_monitors()


if __name__ == "__main__":
    main()
