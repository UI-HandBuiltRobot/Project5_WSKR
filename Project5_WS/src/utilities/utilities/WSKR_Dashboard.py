#!/usr/bin/env python3
"""WSKR Dashboard — the "what is the robot thinking?" window.

A single tkinter window tiled into three panels so you can watch the whole
WSKR stack at a glance while it runs:

    Tile 1 — Live camera (``camera1/image_raw/compressed``) with a bright
             yellow overlay showing the tracked-target bbox (drawn from
             ``WSKR/tracked_bbox``, which is only published while an
             ``ApproachObject`` goal is active).
    Tile 2 — The consolidated WSKR overlay
             (``wskr_overlay/compressed``) — floor mask in the background,
             labelled whisker rays, dashed heading meridians, and a text
             strip with heading / mode / cmd_vel.
    Tile 3 — Telemetry panel: fused ``heading_to_target`` in degrees,
             ``tracking_mode`` (visual/dead_reckoning) as a colored badge,
             a top-down schematic of the robot with the whisker fan drawn
             as rays (length ∝ drive distance, color-coded red <150 mm,
             amber <400 mm, blue otherwise) plus a yellow arrow in the
             ``heading_to_target`` direction, and the latest
             ``WSKR/cmd_vel`` autopilot twist.

Refreshes at ~15 Hz. Nothing here commands motion — it's read-only.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
import tkinter as tk
from PIL import Image, ImageTk
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image as RosImage
from std_msgs.msg import Float32, Float32MultiArray, String


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def fit_to_label(frame: np.ndarray, lw: int, lh: int) -> np.ndarray:
    lw = max(lw, 1)
    lh = max(lh, 1)
    sh, sw = frame.shape[:2]
    if sw <= 0 or sh <= 0:
        return np.zeros((lh, lw, 3), dtype=np.uint8)
    scale = min(lw / sw, lh / sh)
    new_w = max(int(sw * scale), 1)
    new_h = max(int(sh * scale), 1)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((lh, lw, 3), dtype=np.uint8)
    off_x = (lw - new_w) // 2
    off_y = (lh - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


class WSKRDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__('wskr_dashboard')

        self.bridge = CvBridge()

        self._cam_lock = threading.Lock()
        self._overlay_lock = threading.Lock()

        self._cam_frame: Optional[np.ndarray] = None
        self._overlay_frame: Optional[np.ndarray] = None

        self._tracked_bbox: Optional[tuple[float, float, float, float]] = None
        self._whiskers_mm: Optional[np.ndarray] = None
        self._heading_deg: Optional[float] = None
        self._mode: str = '—'
        self._cmd_vel: Twist = Twist()

        self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self._on_camera, IMAGE_QOS,
        )
        self.create_subscription(
            CompressedImage, 'wskr_overlay/compressed', self._on_overlay, IMAGE_QOS,
        )
        self.create_subscription(
            Float32MultiArray, 'WSKR/tracked_bbox', self._on_tracked_bbox, 10
        )
        self.create_subscription(
            Float32MultiArray, 'WSKR/whisker_lengths', self._on_whiskers, 10
        )
        self.create_subscription(Float32, 'WSKR/heading_to_target', self._on_heading, 10)
        self.create_subscription(String, 'WSKR/tracking_mode', self._on_mode, 10)
        self.create_subscription(Twist, 'WSKR/cmd_vel', self._on_cmd_vel, 10)

        self.gui_window: Optional[tk.Tk] = None
        self.cam_label: Optional[tk.Label] = None
        self.overlay_label: Optional[tk.Label] = None
        self.numeric_canvas: Optional[tk.Canvas] = None
        self._gui_stop = threading.Event()

        self._start_gui()

    def _decode(self, msg: RosImage) -> Optional[np.ndarray]:
        try:
            if msg.encoding in ('mono8', '8UC1'):
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'decode failed ({msg.encoding}): {exc}')
            return None
        return ensure_bgr(frame)

    def _on_camera(self, msg: CompressedImage) -> None:
        # Display-only; half-res decode is fast and plenty for the tile.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            return
        with self._cam_lock:
            self._cam_frame = frame

    def _on_overlay(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._overlay_lock:
            self._overlay_frame = frame

    def _on_tracked_bbox(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self._tracked_bbox = None
            return
        self._tracked_bbox = (
            float(msg.data[0]), float(msg.data[1]),
            float(msg.data[2]), float(msg.data[3]),
        )

    def _on_whiskers(self, msg: Float32MultiArray) -> None:
        self._whiskers_mm = np.asarray(msg.data, dtype=np.float64)

    def _on_heading(self, msg: Float32) -> None:
        self._heading_deg = float(msg.data)

    def _on_mode(self, msg: String) -> None:
        self._mode = msg.data

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = msg

    def _start_gui(self) -> None:
        threading.Thread(target=self._gui_run, daemon=True).start()

    def _gui_run(self) -> None:
        root = tk.Tk()
        root.title('WSKR Dashboard')
        root.geometry('1400x900')
        root.configure(bg='#101010')
        self.gui_window = root

        grid = tk.Frame(root, bg='#101010')
        grid.pack(fill=tk.BOTH, expand=True)
        grid.rowconfigure(0, weight=1, uniform='row')
        grid.rowconfigure(1, weight=1, uniform='row')
        grid.columnconfigure(0, weight=1, uniform='col')
        grid.columnconfigure(1, weight=1, uniform='col')

        def make_tile(parent: tk.Frame, title: str, row: int, col: int, colspan: int = 1) -> tk.Label:
            tile = tk.Frame(parent, bg='#101010', bd=1, relief=tk.FLAT)
            tile.grid(row=row, column=col, columnspan=colspan, sticky='nsew', padx=4, pady=4)
            tk.Label(
                tile, text=title, bg='#202020', fg='#d0d0d0',
                font=('Arial', 11, 'bold'), anchor='w', padx=6,
            ).pack(side=tk.TOP, fill=tk.X)
            img_label = tk.Label(tile, bg='black')
            img_label.pack(fill=tk.BOTH, expand=True)
            return img_label

        self.cam_label = make_tile(grid, 'Target Tracking (camera1/image_raw/compressed)', 0, 0)
        self.overlay_label = make_tile(grid, 'Consolidated WSKR Overlay (wskr_overlay/compressed)', 0, 1)

        # Numeric panel spans the bottom row
        stats_tile = tk.Frame(grid, bg='#101010')
        stats_tile.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
        tk.Label(
            stats_tile, text='Telemetry', bg='#202020', fg='#d0d0d0',
            font=('Arial', 11, 'bold'), anchor='w', padx=6,
        ).pack(side=tk.TOP, fill=tk.X)
        self.numeric_canvas = tk.Canvas(stats_tile, bg='#181818', highlightthickness=0)
        self.numeric_canvas.pack(fill=tk.BOTH, expand=True)

        def refresh() -> None:
            if self._gui_stop.is_set():
                return
            self._render_camera_tile()
            self._render_overlay_tile()
            self._render_numeric_tile()
            root.after(66, refresh)  # ~15 Hz

        root.after(66, refresh)

        def on_close() -> None:
            self._gui_stop.set()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()

    def _render_image_label(self, label: tk.Label, frame: Optional[np.ndarray]) -> None:
        lw = label.winfo_width()
        lh = label.winfo_height()
        if frame is None:
            canvas = np.zeros((max(lh, 1), max(lw, 1), 3), dtype=np.uint8)
        else:
            canvas = fit_to_label(frame, lw, lh)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.config(image=photo)
        label.image = photo  # keep reference

    def _render_camera_tile(self) -> None:
        if self.cam_label is None:
            return
        with self._cam_lock:
            frame = None if self._cam_frame is None else self._cam_frame.copy()
        if frame is not None and self._tracked_bbox is not None:
            # tracked_bbox is width-normalized (publisher divides by frame
            # width); scale to this camera frame's width to draw.
            scale = float(frame.shape[1])
            xn, yn, wn, hn = self._tracked_bbox
            x1 = int(xn * scale)
            y1 = int(yn * scale)
            x2 = int((xn + wn) * scale)
            y2 = int((yn + hn) * scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                frame, 'TARGET', (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        self._render_image_label(self.cam_label, frame)

    def _render_overlay_tile(self) -> None:
        if self.overlay_label is None:
            return
        with self._overlay_lock:
            frame = None if self._overlay_frame is None else self._overlay_frame.copy()
        # Paint the latest tracked bbox directly here (not through
        # wskr_range_node's overlay compose loop) so the bbox freshness is
        # bounded only by approach_action_server's publish rate, not by the
        # range_node's single-threaded callback queue.
        if frame is not None and self._tracked_bbox is not None:
            scale = float(frame.shape[1])
            xn, yn, wn, hn = self._tracked_bbox
            x1 = int(xn * scale)
            y1 = int(yn * scale)
            x2 = int((xn + wn) * scale)
            y2 = int((yn + hn) * scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame, 'TARGET', (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )
        self._render_image_label(self.overlay_label, frame)

    @staticmethod
    def _whisker_color(mm: float) -> str:
        """Match the bar-chart thresholds: red <150, amber <400, blue otherwise."""
        if mm < 150.0:
            return '#ff6060'
        if mm < 400.0:
            return '#ffcc40'
        return '#60c0ff'

    def _draw_robot_diagram(
        self, c: tk.Canvas, x0: float, y0: float, x1: float, y1: float,
    ) -> None:
        """Draw a top-down schematic: ground line, robot body with wheels,
        whisker rays fanning from the robot's front, and a yellow arrow in
        the heading-to-target direction. Coordinates are canvas pixels,
        forward = up, +heading = left (CCW from forward)."""
        area_w = x1 - x0
        area_h = y1 - y0
        if area_w < 60 or area_h < 60:
            return

        cx = (x0 + x1) / 2.0
        # Robot body: modest footprint at the bottom of the diagram area.
        body_w = min(area_w * 0.14, 90.0)
        body_h = body_w * 0.75
        wheel_r = max(body_w * 0.11, 3.0)

        robot_bottom = y1 - 10
        robot_top = robot_bottom - body_h
        body_cx = cx
        body_cy = (robot_top + robot_bottom) / 2.0

        # Whisker origin at front-center of the body (the top edge, since
        # forward = up on screen).
        ox, oy = cx, robot_top
        max_ray_px = max(20.0, (oy - y0) - 8)

        # Ground / horizon line through the whisker origin.
        c.create_line(x0, oy, x1, oy, fill='#404040', width=1)

        # --- Whisker rays --------------------------------------------------
        whiskers = self._whiskers_mm
        scale_mm = 500.0  # wskr_range's default max_range_mm; caps the visual.
        if whiskers is not None and whiskers.size > 0:
            n = int(whiskers.size)
            for i, mm in enumerate(whiskers):
                # Whisker[0] is the -90° (right-side) ray per
                # approach_action_server's np.linspace(-90, 90, n) convention.
                theta_deg = -90.0 + i * (180.0 / max(n - 1, 1))
                theta = math.radians(theta_deg)
                length = max_ray_px * max(0.0, min(1.0, float(mm) / scale_mm))
                # Screen: forward=up, +theta=CCW → left on screen (-x).
                ex = ox - length * math.sin(theta)
                ey = oy - length * math.cos(theta)
                color = self._whisker_color(float(mm))
                c.create_line(ox, oy, ex, ey, fill=color, width=2)
                c.create_oval(
                    ex - 3, ey - 3, ex + 3, ey + 3, fill=color, outline='',
                )
                # Length label at the tip, nudged outward so it sits off the ray.
                nudge = 10
                lx = ex - nudge * math.sin(theta)
                ly = ey - nudge * math.cos(theta)
                c.create_text(
                    lx, ly, text=f'{int(mm)}',
                    fill='#b0b0b0', font=('Consolas', 8),
                )

        # --- Robot body + wheels ------------------------------------------
        c.create_rectangle(
            cx - body_w / 2, robot_top, cx + body_w / 2, robot_bottom,
            fill='#2a63d6', outline='#1d4ba3', width=1,
        )
        for wx, wy in (
            (cx - body_w / 2, robot_top),
            (cx + body_w / 2, robot_top),
            (cx - body_w / 2, robot_bottom),
            (cx + body_w / 2, robot_bottom),
        ):
            c.create_oval(
                wx - wheel_r, wy - wheel_r, wx + wheel_r, wy + wheel_r,
                fill='#101010', outline='#303030',
            )

        # Body axes: red = forward (+x, up), green = left (+y, left).
        axis_len = body_w * 0.32
        c.create_line(
            body_cx, body_cy, body_cx, body_cy - axis_len,
            fill='#ff6060', width=2, arrow=tk.LAST, arrowshape=(7, 8, 3),
        )
        c.create_line(
            body_cx, body_cy, body_cx - axis_len, body_cy,
            fill='#60ff60', width=2, arrow=tk.LAST, arrowshape=(7, 8, 3),
        )

        # --- Heading-to-target arrow --------------------------------------
        heading = self._heading_deg
        if heading is not None:
            theta = math.radians(float(heading))
            # Slightly longer than the longest whisker so the arrow reads as
            # the dominant visual cue.
            arrow_len = max_ray_px * 1.05
            ex = body_cx - arrow_len * math.sin(theta)
            ey = body_cy - arrow_len * math.cos(theta)
            # Clamp inside the diagram area so rearward targets (|heading|>90)
            # don't punch through the tile.
            ex = max(x0 + 6, min(x1 - 6, ex))
            ey = max(y0 + 6, min(y1 - 6, ey))
            arrow_color = '#ffd84a' if self._mode == 'visual' else '#ff9040'
            c.create_line(
                body_cx, body_cy, ex, ey,
                fill=arrow_color, width=5,
                arrow=tk.LAST, arrowshape=(18, 22, 8),
                capstyle=tk.ROUND,
            )

    def _render_numeric_tile(self) -> None:
        c = self.numeric_canvas
        if c is None:
            return
        c.delete('all')
        w = max(c.winfo_width(), 1)
        h = max(c.winfo_height(), 1)

        heading = self._heading_deg
        heading_txt = '—' if heading is None else f'{heading:+7.2f}°'
        heading_color = '#80ff80' if self._mode == 'visual' else '#ffaa40'
        mode_color = '#80ff80' if self._mode == 'visual' else '#ffaa40'

        y = 16
        c.create_text(
            14, y, anchor='nw', text='Heading to target',
            fill='#a0a0a0', font=('Arial', 11),
        )
        c.create_text(
            14, y + 18, anchor='nw', text=heading_txt,
            fill=heading_color, font=('Consolas', 28, 'bold'),
        )
        c.create_text(
            w - 14, y + 18, anchor='ne', text=f'mode: {self._mode}',
            fill=mode_color, font=('Consolas', 14, 'bold'),
        )

        # Top-down robot diagram: whisker fan + heading arrow
        y_diag_top = y + 66
        y_diag_bottom = h - 80
        c.create_text(
            14, y_diag_top - 18, anchor='nw', text='Whisker fan (mm) + heading arrow',
            fill='#a0a0a0', font=('Arial', 11),
        )
        self._draw_robot_diagram(c, 14, y_diag_top, w - 14, y_diag_bottom)

        # cmd_vel
        vx = self._cmd_vel.linear.x
        vy = self._cmd_vel.linear.y
        wz_deg = math.degrees(self._cmd_vel.angular.z)
        y_cmd = h - 60
        c.create_text(
            14, y_cmd, anchor='nw', text='Autopilot cmd_vel (WSKR/cmd_vel)',
            fill='#a0a0a0', font=('Arial', 11),
        )
        cmd_str = f'Vx={vx:+6.2f} m/s   Vy={vy:+6.2f} m/s   ω={wz_deg:+6.1f}°/s'
        c.create_text(
            14, y_cmd + 20, anchor='nw', text=cmd_str,
            fill='#e0e0e0', font=('Consolas', 13, 'bold'),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WSKRDashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
