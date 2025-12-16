#!/usr/bin/env python3
"""
Desktop Observer GUI
====================
A simple graphical interface for desktop observation.

Usage:
    python gui_observer.py
"""

import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

try:
    import cv2
    import mss
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


class DesktopObserverGUI:
    """GUI application for desktop observation."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖥️ Desktop Observer")
        self.root.geometry("800x650")
        self.root.minsize(600, 500)

        # State
        self.sct = mss.mss()
        self.streaming = False
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_frame = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Settings
        self.selected_monitor = tk.IntVar(value=0)
        self.target_fps = tk.IntVar(value=15)
        self.show_fps = tk.BooleanVar(value=True)

        self._setup_ui()
        self._update_monitor_list()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Monitor selection
        monitor_frame = ttk.Frame(control_frame)
        monitor_frame.pack(fill=tk.X, pady=5)

        ttk.Label(monitor_frame, text="Monitor:").pack(side=tk.LEFT)
        self.monitor_combo = ttk.Combobox(
            monitor_frame, 
            textvariable=self.selected_monitor,
            state="readonly",
            width=30
        )
        self.monitor_combo.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(monitor_frame, text="FPS:").pack(side=tk.LEFT)
        fps_spin = ttk.Spinbox(
            monitor_frame,
            from_=1, to=60,
            textvariable=self.target_fps,
            width=5
        )
        fps_spin.pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(
            monitor_frame,
            text="Show FPS",
            variable=self.show_fps
        ).pack(side=tk.LEFT, padx=10)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.stream_btn = ttk.Button(
            button_frame,
            text="▶ Start Stream",
            command=self._toggle_stream
        )
        self.stream_btn.pack(side=tk.LEFT, padx=5)

        self.screenshot_btn = ttk.Button(
            button_frame,
            text="📸 Screenshot",
            command=self._take_screenshot
        )
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)

        self.record_btn = ttk.Button(
            button_frame,
            text="⏺ Record",
            command=self._toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Preview canvas
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))

        # Bind resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_monitor_list(self):
        """Update the monitor selection dropdown."""
        monitors = self.sct.monitors
        options = []
        for i, mon in enumerate(monitors):
            if i == 0:
                label = f"All Monitors ({mon['width']}x{mon['height']})"
            else:
                label = f"Monitor {i} ({mon['width']}x{mon['height']})"
            options.append(label)

        self.monitor_combo['values'] = options
        self.monitor_combo.current(0)

    def _get_monitor(self) -> dict:
        """Get the selected monitor configuration."""
        monitors = self.sct.monitors
        idx = self.selected_monitor.get()
        if idx < len(monitors):
            return monitors[idx]
        return monitors[0]

    def _capture_frame(self) -> np.ndarray:
        """Capture a frame from the selected monitor."""
        monitor = self._get_monitor()
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _toggle_stream(self):
        """Start or stop the live stream."""
        if self.streaming:
            self.streaming = False
            self.stream_btn.config(text="▶ Start Stream")
            self.status_var.set("Stream stopped")
        else:
            self.streaming = True
            self.stream_btn.config(text="⏹ Stop Stream")
            self.status_var.set("Streaming...")
            self.frame_count = 0
            self.fps_start_time = time.time()
            threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stream_loop(self):
        """Background thread for streaming frames."""
        while self.streaming:
            try:
                frame_start = time.time()

                # Capture frame
                frame = self._capture_frame()
                self.current_frame = frame

                # Calculate FPS
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()

                # Write to video if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)

                # Update preview
                self._update_preview(frame)

                # Frame rate control
                target_delay = 1.0 / self.target_fps.get()
                elapsed = time.time() - frame_start
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)

            except Exception as e:
                print(f"Stream error: {e}")
                self.streaming = False
                break

    def _update_preview(self, frame: np.ndarray):
        """Update the canvas with a new frame."""
        try:
            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width < 10 or canvas_height < 10:
                return

            # Calculate scale to fit
            frame_height, frame_width = frame.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize frame
            resized = cv2.resize(frame, (new_width, new_height))

            # Add FPS overlay if enabled
            if self.show_fps.get():
                fps_text = f"FPS: {self.current_fps:.1f}"
                if self.recording:
                    fps_text += " [REC]"
                cv2.putText(
                    resized, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            # Update canvas (must be on main thread)
            self.root.after(0, lambda: self._draw_image(photo, canvas_width, canvas_height, new_width, new_height))

        except Exception as e:
            print(f"Preview error: {e}")

    def _draw_image(self, photo, canvas_width, canvas_height, img_width, img_height):
        """Draw image on canvas (main thread)."""
        try:
            self.canvas.delete("all")
            x = (canvas_width - img_width) // 2
            y = (canvas_height - img_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.canvas._photo = photo  # Keep reference
        except Exception:
            pass

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        pass  # Preview will auto-adjust on next frame

    def _take_screenshot(self):
        """Capture and save a screenshot."""
        try:
            frame = self._capture_frame()
            
            # Ask for save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"screenshot_{timestamp}.png"
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                initialfile=default_name
            )
            
            if filepath:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.save(filepath)
                self.status_var.set(f"Screenshot saved: {Path(filepath).name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {e}")

    def _toggle_recording(self):
        """Start or stop video recording."""
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start video recording."""
        # Get save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"recording_{timestamp}.mp4"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")],
            initialfile=default_name
        )
        
        if not filepath:
            return

        try:
            monitor = self._get_monitor()
            width = monitor["width"]
            height = monitor["height"]
            fps = self.target_fps.get()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")

            self.recording = True
            self.record_btn.config(text="⏹ Stop Recording")
            self.status_var.set(f"Recording to: {Path(filepath).name}")

            # Start streaming if not already
            if not self.streaming:
                self._toggle_stream()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def _stop_recording(self):
        """Stop video recording."""
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.record_btn.config(text="⏺ Record")
        self.status_var.set("Recording saved")

    def _on_close(self):
        """Handle window close."""
        self.streaming = False
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    app = DesktopObserverGUI()
    app.run()


if __name__ == "__main__":
    main()
