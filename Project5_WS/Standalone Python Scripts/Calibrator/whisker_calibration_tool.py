import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


PointTuple = Tuple[float, float, float]


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if self.tip_window is not None or not self.text:
            return

        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#fff8cc",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=3,
        )
        label.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


@dataclass
class WhiskerCalibration:
    angle_deg: int
    color: str
    points: List[PointTuple] = field(default_factory=list)
    spline_x: Optional[UnivariateSpline] = None
    spline_y: Optional[UnivariateSpline] = None
    fitted: bool = False

    def clear(self) -> None:
        self.points.clear()
        self.spline_x = None
        self.spline_y = None
        self.fitted = False

    def add_point(self, pixel_x: float, pixel_y: float, distance_mm: float) -> None:
        self.points.append((pixel_x, pixel_y, distance_mm))

    def remove_last_point(self) -> None:
        if self.points:
            self.points.pop()
        if not self.points:
            self.spline_x = None
            self.spline_y = None
            self.fitted = False

    def fit(self) -> None:
        if len(self.points) < 3:
            raise ValueError("At least 3 points are required to fit a spline.")

        sorted_points = sorted(self.points, key=lambda p: p[2])
        distances = np.array([p[2] for p in sorted_points], dtype=float)
        pixels_x = np.array([p[0] for p in sorted_points], dtype=float)
        pixels_y = np.array([p[1] for p in sorted_points], dtype=float)

        if np.unique(distances).size != distances.size:
            raise ValueError("Distance values must be unique for spline fitting.")

        self.spline_x = UnivariateSpline(distances, pixels_x, k=3, s=0)
        self.spline_y = UnivariateSpline(distances, pixels_y, k=3, s=0)
        self.fitted = True


class CameraSelector:
    def enumerate_cameras(self) -> List[int]:
        available = []
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                available.append(index)
                cap.release()
        return available

    def connect(self, index: int) -> Tuple[Optional[cv2.VideoCapture], bool, str]:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return None, False, "Camera index not available"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w != 1920 or actual_h != 1080:
            cap.release()
            return None, False, "Camera does not support 1920×1080"

        return cap, True, "1920×1080 OK"


class CalibrationApp:
    WHISKER_ANGLES = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]
    WHISKER_COLORS = [
        "#f44336",
        "#e91e63",
        "#9c27b0",
        "#3f51b5",
        "#2196f3",
        "#00bcd4",
        "#009688",
        "#4caf50",
        "#8bc34a",
        "#ff9800",
        "#795548",
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Whisker Calibration Tool")
        self.root.geometry("1400x850")

        self.camera_selector = CameraSelector()
        self.available_cameras: List[int] = []
        self.capture: Optional[cv2.VideoCapture] = None
        self.preview_after_id: Optional[str] = None
        self.preview_active = False

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_width = 0
        self.image_height = 0
        self.image_path = ""

        self.zoom_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.dragging_pan = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.active_whisker: Optional[int] = None
        self.whiskers: Dict[int, WhiskerCalibration] = {
            angle: WhiskerCalibration(angle, self.WHISKER_COLORS[i])
            for i, angle in enumerate(self.WHISKER_ANGLES)
        }

        self.whisker_buttons: Dict[int, tk.Button] = {}
        self.canvas_image: Optional[ImageTk.PhotoImage] = None
        self.tooltips: List[ToolTip] = []

        self._build_ui()
        self._refresh_camera_list()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, padx=10, pady=10)
        left.grid(row=0, column=0, sticky="ns")

        center = tk.Frame(self.root, padx=10, pady=10)
        center.grid(row=0, column=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)

        right = tk.Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=2, sticky="ns")

        tk.Label(left, text="Camera Index").pack(anchor="w")
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(left, textvariable=self.camera_var, state="readonly", width=18)
        self.camera_combo.pack(fill="x", pady=(2, 8))

        self.connect_btn = tk.Button(left, text="Connect", command=self.connect_camera)
        self.connect_btn.pack(fill="x", pady=3)

        self.resolution_status = tk.Label(left, text="Not connected", fg="gray")
        self.resolution_status.pack(anchor="w", pady=(4, 8))

        self.snap_btn = tk.Button(left, text="Snap Image", command=self.snap_image, state="disabled")
        self.snap_btn.pack(fill="x", pady=3)

        self.load_btn = tk.Button(left, text="Load Calibration", command=self.load_calibration)
        self.load_btn.pack(fill="x", pady=3)

        self.save_btn = tk.Button(left, text="Save Calibration", command=self.save_calibration)
        self.save_btn.pack(fill="x", pady=3)

        self.path_status = tk.Label(left, text="No calibration file", wraplength=220, justify="left", fg="gray")
        self.path_status.pack(anchor="w", pady=(8, 0))

        self.camera_status = tk.Label(left, text="", wraplength=220, justify="left", fg="gray")
        self.camera_status.pack(anchor="w", pady=(8, 0))

        self.canvas = tk.Canvas(center, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        tk.Label(right, text="Whiskers", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        for angle in self.WHISKER_ANGLES:
            btn = tk.Button(
                right,
                text=f"{angle}°",
                width=10,
                bg="#b5b5b5",
                relief="raised",
                command=lambda a=angle: self.start_whisker_selection(a),
            )
            btn.pack(fill="x", pady=2)
            self.whisker_buttons[angle] = btn

        self.done_btn = tk.Button(right, text="Done", command=self.done_active_whisker)

        self.tooltips.extend(
            [
                ToolTip(self.camera_combo, "Select which detected camera index to use for calibration."),
                ToolTip(self.connect_btn, "Connect to selected camera and start live preview."),
                ToolTip(self.snap_btn, "Capture a frame for whisker point marking."),
                ToolTip(self.load_btn, "Load an existing calibration JSON and image."),
                ToolTip(self.save_btn, "Save calibration points and snapshot path to JSON."),
                ToolTip(self.path_status, "Current calibration file path or snapshot location."),
                ToolTip(self.camera_status, "Camera connection and preview status messages."),
                ToolTip(self.canvas, "Left-click: add point, Right-click: remove last point, Wheel: zoom, Middle-drag: pan."),
                ToolTip(self.done_btn, "Fit and finalize the currently active whisker."),
            ]
        )
        for angle, btn in self.whisker_buttons.items():
            self.tooltips.append(ToolTip(btn, f"Select whisker {angle} deg and begin entering distance points."))

    def _refresh_camera_list(self) -> None:
        self.available_cameras = self.camera_selector.enumerate_cameras()
        values = [f"Camera {idx}" for idx in self.available_cameras]
        self.camera_combo["values"] = values
        if values:
            self.camera_combo.current(0)
            self.camera_status.config(text="")
        else:
            self.camera_var.set("")
            self.camera_status.config(text="No cameras found (indices 0-9)", fg="red")

    def connect_camera(self) -> None:
        self._stop_preview_loop()
        self._release_camera()
        self.snap_btn.config(state="disabled")
        self.preview_active = False

        if not self.camera_var.get():
            self.camera_status.config(text="Select a camera first", fg="red")
            return

        try:
            selected_index = int(self.camera_var.get().split()[-1])
        except ValueError:
            self.camera_status.config(text="Invalid camera selection", fg="red")
            return

        cap, ok, message = self.camera_selector.connect(selected_index)
        if not ok or cap is None:
            self.capture = None
            self.resolution_status.config(text=message, fg="red")
            self.camera_status.config(text=message, fg="red")
            return

        self.capture = cap
        self.preview_active = True
        self.resolution_status.config(text=message, fg="green")
        self.camera_status.config(text=f"Connected to camera {selected_index}", fg="green")
        self.snap_btn.config(state="normal")
        self._preview_loop()

    def _preview_loop(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            return

        ok, frame = self.capture.read()
        if ok:
            self.image_bgr = frame
            self._set_image_from_bgr(frame, reset_view=False)
            self._fit_image_to_canvas()
            self.draw_scene()

        self.preview_after_id = self.root.after(66, self._preview_loop)

    def _stop_preview_loop(self) -> None:
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None

    def _release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.preview_active = False

    def snap_image(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            self.camera_status.config(text="No active camera connection", fg="red")
            return

        ok, frame = self.capture.read()
        if not ok:
            self.camera_status.config(text="Failed to capture frame", fg="red")
            return

        self._stop_preview_loop()
        self._release_camera()
        self.snap_btn.config(state="disabled")

        self.image_bgr = frame
        self._set_image_from_bgr(frame, reset_view=True)
        self._fit_image_to_canvas()
        self.image_path = str((Path.cwd() / "calibration_snapshot.png").resolve())
        cv2.imwrite(self.image_path, frame)
        self.path_status.config(text=self.image_path, fg="black")
        self.camera_status.config(text="Snapshot captured", fg="green")

        self.draw_scene()

    def _set_image_from_bgr(self, frame_bgr: np.ndarray, reset_view: bool) -> None:
        self.image_bgr = frame_bgr.copy()
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.image_rgb.shape[:2]

        if reset_view:
            self.zoom_scale = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

    def _fit_image_to_canvas(self) -> None:
        if self.image_rgb is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        sx = canvas_w / float(self.image_width)
        sy = canvas_h / float(self.image_height)
        fit_scale = max(0.1, min(10.0, min(sx, sy)))

        self.zoom_scale = fit_scale
        img_display_w = self.image_width * self.zoom_scale
        img_display_h = self.image_height * self.zoom_scale
        self.pan_x = (canvas_w - img_display_w) / 2.0
        self.pan_y = (canvas_h - img_display_h) / 2.0

    def canvas_to_image(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        img_x = (canvas_x - self.pan_x) / self.zoom_scale
        img_y = (canvas_y - self.pan_y) / self.zoom_scale
        return img_x, img_y

    def image_to_canvas(self, img_x: float, img_y: float) -> Tuple[float, float]:
        canvas_x = img_x * self.zoom_scale + self.pan_x
        canvas_y = img_y * self.zoom_scale + self.pan_y
        return canvas_x, canvas_y

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if self.image_rgb is None:
            return

        old_scale = self.zoom_scale
        if event.delta > 0:
            new_scale = min(10.0, old_scale * 1.1)
        else:
            new_scale = max(0.1, old_scale / 1.1)

        if abs(new_scale - old_scale) < 1e-12:
            return

        img_x, img_y = self.canvas_to_image(event.x, event.y)
        self.zoom_scale = new_scale
        self.pan_x = event.x - img_x * self.zoom_scale
        self.pan_y = event.y - img_y * self.zoom_scale
        self.draw_scene()

    def on_pan_start(self, event: tk.Event) -> None:
        self.dragging_pan = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_move(self, event: tk.Event) -> None:
        if not self.dragging_pan:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.draw_scene()

    def on_pan_end(self, _event: tk.Event) -> None:
        self.dragging_pan = False

    def on_left_click(self, event: tk.Event) -> None:
        if self.active_whisker is None or self.image_rgb is None:
            return

        img_x, img_y = self.canvas_to_image(event.x, event.y)
        if not (0 <= img_x <= self.image_width and 0 <= img_y <= self.image_height):
            return

        distance = simpledialog.askfloat(
            "Distance (mm)",
            f"Enter distance in mm for whisker {self.active_whisker}°:",
            parent=self.root,
        )
        if distance is None:
            return

        self.whiskers[self.active_whisker].add_point(img_x, img_y, float(distance))
        self.draw_scene()

    def on_right_click(self, _event: tk.Event) -> None:
        if self.active_whisker is None:
            return
        self.whiskers[self.active_whisker].remove_last_point()
        self._update_whisker_button_states()
        self.draw_scene()

    def start_whisker_selection(self, angle: int) -> None:
        if self.image_rgb is None:
            messagebox.showwarning("No image", "Snap or load an image before calibrating whiskers.")
            return

        self.active_whisker = angle
        self.whiskers[angle].clear()
        self.done_btn.pack(fill="x", pady=(10, 0))
        self._update_whisker_button_states()
        self.draw_scene()

    def done_active_whisker(self) -> None:
        if self.active_whisker is None:
            return

        whisker = self.whiskers[self.active_whisker]
        if len(whisker.points) < 3:
            messagebox.showwarning("Not enough points", "At least 3 points are required before fitting.")
            return

        try:
            whisker.fit()
        except Exception as exc:
            whisker.spline_x = None
            whisker.spline_y = None
            whisker.fitted = False
            messagebox.showwarning("Spline fit failed", f"Whisker {self.active_whisker}° fit failed:\n{exc}")
            return

        self.active_whisker = None
        self.done_btn.pack_forget()
        self._update_whisker_button_states()
        self.draw_scene()

    def _update_whisker_button_states(self) -> None:
        for angle, btn in self.whisker_buttons.items():
            whisker = self.whiskers[angle]
            if whisker.fitted:
                base_color = "#87d37c"
            else:
                base_color = "#b5b5b5"

            btn.config(bg=base_color)
            if self.active_whisker == angle:
                btn.config(highlightthickness=3, highlightbackground="yellow", highlightcolor="yellow")
            else:
                btn.config(highlightthickness=1, highlightbackground="#909090", highlightcolor="#909090")

    def draw_scene(self) -> None:
        self.canvas.delete("all")

        if self.image_rgb is None:
            self._update_whisker_button_states()
            return

        display_w = max(1, int(self.image_width * self.zoom_scale))
        display_h = max(1, int(self.image_height * self.zoom_scale))

        pil_img = Image.fromarray(self.image_rgb)
        if display_w != self.image_width or display_h != self.image_height:
            pil_img = pil_img.resize((display_w, display_h), resample=Image.Resampling.BILINEAR)

        self.canvas_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.pan_x, self.pan_y, image=self.canvas_image, anchor="nw")

        origin_x = self.image_width / 2.0
        origin_y = float(self.image_height)
        ox, oy = self.image_to_canvas(origin_x, origin_y)
        self.canvas.create_oval(ox - 4, oy - 4, ox + 4, oy + 4, fill="#ffd54f", outline="black", width=1)

        for whisker in self.whiskers.values():
            for px, py, _distance in whisker.points:
                cx, cy = self.image_to_canvas(px, py)
                self.canvas.create_oval(
                    cx - 4,
                    cy - 4,
                    cx + 4,
                    cy + 4,
                    fill=whisker.color,
                    outline="white",
                    width=1,
                )

            if whisker.fitted and whisker.spline_x is not None and whisker.spline_y is not None:
                d_vals = np.linspace(0.0, 500.0, 200)
                try:
                    x_vals = whisker.spline_x(d_vals)
                    y_vals = whisker.spline_y(d_vals)
                    poly = []
                    for x, y in zip(x_vals, y_vals):
                        cx, cy = self.image_to_canvas(float(x), float(y))
                        poly.extend([cx, cy])
                    if len(poly) >= 4:
                        self.canvas.create_line(*poly, fill=whisker.color, width=2, smooth=True)
                except Exception:
                    pass

        self._update_whisker_button_states()

    def save_calibration(self) -> None:
        if self.image_rgb is None:
            messagebox.showwarning("No image", "There is no snapped or loaded image to save.")
            return

        json_path = filedialog.asksaveasfilename(
            title="Save Calibration",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not json_path:
            return

        json_path_obj = Path(json_path).resolve()
        target_img_path = (json_path_obj.parent / "calibration_snapshot.png").resolve()

        if self.image_path:
            src_img = Path(self.image_path).resolve()
        else:
            src_img = (Path.cwd() / "calibration_snapshot.png").resolve()

        if src_img != target_img_path:
            cv2.imwrite(str(target_img_path), cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR))
            self.image_path = str(target_img_path)
        else:
            if not src_img.exists():
                cv2.imwrite(str(src_img), cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR))
            self.image_path = str(src_img)

        whisker_json: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        for angle in self.WHISKER_ANGLES:
            points = self.whiskers[angle].points
            whisker_json[str(angle)] = {
                "points": [
                    {
                        "pixel_x": float(px),
                        "pixel_y": float(py),
                        "distance_mm": float(dist),
                    }
                    for px, py, dist in points
                ]
            }

        payload = {
            "image_path": self.image_path,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "whiskers": whisker_json,
        }

        with open(json_path_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.path_status.config(text=str(json_path_obj), fg="black")
        messagebox.showinfo("Saved", "Calibration saved successfully.")

    def load_calibration(self) -> None:
        json_path = filedialog.askopenfilename(
            title="Load Calibration",
            filetypes=[("JSON", "*.json")],
        )
        if not json_path:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Failed to read JSON:\n{exc}")
            return

        image_path = data.get("image_path", "")
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            messagebox.showerror("Image missing", f"Image not found at:\n{image_path}\nPlease locate it manually.")
            manual_path = filedialog.askopenfilename(
                title="Locate Calibration Image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")],
            )
            if not manual_path:
                return
            image_path_obj = Path(manual_path)

        img_bgr = cv2.imread(str(image_path_obj))
        if img_bgr is None:
            messagebox.showerror("Load failed", "Failed to load selected image file.")
            return

        self._stop_preview_loop()
        self._release_camera()
        self.snap_btn.config(state="disabled")

        self.image_path = str(image_path_obj.resolve())
        self._set_image_from_bgr(img_bgr, reset_view=True)
        self._fit_image_to_canvas()

        for angle in self.WHISKER_ANGLES:
            self.whiskers[angle].clear()

        whiskers_json = data.get("whiskers", {})
        fit_errors: List[str] = []

        for angle in self.WHISKER_ANGLES:
            point_entries = whiskers_json.get(str(angle), {}).get("points", [])
            whisker = self.whiskers[angle]

            for item in point_entries:
                try:
                    px = float(item["pixel_x"])
                    py = float(item["pixel_y"])
                    dist = float(item["distance_mm"])
                except Exception:
                    continue
                whisker.add_point(px, py, dist)

            if len(whisker.points) >= 3:
                try:
                    whisker.fit()
                except Exception as exc:
                    whisker.fitted = False
                    whisker.spline_x = None
                    whisker.spline_y = None
                    fit_errors.append(f"{angle}°: {exc}")

        self.active_whisker = None
        self.done_btn.pack_forget()
        self.path_status.config(text=str(Path(json_path).resolve()), fg="black")
        self.draw_scene()

        if fit_errors:
            messagebox.showwarning(
                "Partial load",
                "Some whiskers could not be fitted:\n" + "\n".join(fit_errors),
            )

    def on_close(self) -> None:
        self._stop_preview_loop()
        self._release_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = CalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
