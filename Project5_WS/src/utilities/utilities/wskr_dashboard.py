#!/usr/bin/env python3
"""WSKR Dashboard — combined command centre and telemetry viewer.

A single tkinter window tiled into three panels:

    Tile 1 — ArUco Approach control (top-left).
             Live camera preview with locally-detected ArUco markers drawn
             in grey (any tag) or bright yellow (the target tag).
             Controls: ArUco ID entry, Start / Cancel buttons, tag status.
             Status row: goal state transitions.
             Feedback row: live tracking_mode / heading / visual lock /
             closest-whisker distance streamed from action feedback while
             a goal is active.

    Tile 2 — Consolidated WSKR overlay (top-right).
             (``wskr_overlay/compressed``) — floor mask in the background,
             labelled whisker rays, dashed heading meridians, and a text
             strip with heading / mode / cmd_vel.

    Tile 3 — Telemetry panel (bottom, full width).
             Fused ``heading_to_target`` in degrees, ``tracking_mode``
             badge, a top-down schematic with the whisker fan (length ∝
             drive distance, colour-coded), magenta diamond markers for
             ``WSKR/target_whisker_lengths`` intercepts, a heading arrow,
             and the latest ``WSKR/cmd_vel`` autopilot twist.

Refreshes at ~15 Hz.
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
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image as RosImage
from std_msgs.msg import Float32, Float32MultiArray, String

from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.action import ApproachObject


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

ARUCO_DICT = cv2.aruco.DICT_4X4_50  # must match approach_action_server


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

        # ── shared image frames ──────────────────────────────────────────
        self._cam_lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        self._cam_frame: Optional[np.ndarray] = None
        self._overlay_frame: Optional[np.ndarray] = None

        # ── telemetry state ──────────────────────────────────────────────
        self._tracked_bbox: Optional[tuple[float, float, float, float]] = None
        self._whiskers_mm: Optional[np.ndarray] = None
        self._target_whiskers_mm: Optional[np.ndarray] = None
        self._heading_deg: Optional[float] = None
        self._mode: str = '—'
        self._cmd_vel: Twist = Twist()

        # ── ArUco approach state ─────────────────────────────────────────
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict)
        self._approach_client = ActionClient(self, ApproachObject, 'WSKR/approach_object')
        self._active_goal_handle = None

        # Action feedback cache (populated by _on_feedback, cleared on goal end)
        self._fb_mode: str = ''
        self._fb_heading: Optional[float] = None
        self._fb_locked: Optional[bool] = None
        self._fb_whiskers: list[float] = []

        # ── subscriptions ────────────────────────────────────────────────
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
        self.create_subscription(
            Float32MultiArray, 'WSKR/target_whisker_lengths', self._on_target_whiskers, 10
        )
        self.create_subscription(Float32, 'WSKR/heading_to_target', self._on_heading, 10)
        self.create_subscription(String, 'WSKR/tracking_mode', self._on_mode, 10)
        self.create_subscription(Twist, 'WSKR/cmd_vel', self._on_cmd_vel, 10)

        # ── GUI widget refs (set in _gui_run) ────────────────────────────
        self.gui_window: Optional[tk.Tk] = None
        self.cam_label: Optional[tk.Label] = None
        self.overlay_label: Optional[tk.Label] = None
        self.numeric_canvas: Optional[tk.Canvas] = None
        self.aruco_id_entry: Optional[tk.Entry] = None
        self.tag_status_label: Optional[tk.Label] = None
        self.start_btn: Optional[tk.Button] = None
        self.cancel_btn: Optional[tk.Button] = None
        self.approach_status_label: Optional[tk.Label] = None
        self.feedback_label: Optional[tk.Label] = None
        self._gui_stop = threading.Event()

        self.get_logger().info('Waiting for approach_object action server...')
        if not self._approach_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn(
                'ApproachObject action server not available — Start button will be disabled.'
            )

        self._start_gui()

    # ── ROS callbacks ────────────────────────────────────────────────────

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

    def _on_target_whiskers(self, msg: Float32MultiArray) -> None:
        self._target_whiskers_mm = np.asarray(msg.data, dtype=np.float64)

    def _on_heading(self, msg: Float32) -> None:
        self._heading_deg = float(msg.data)

    def _on_mode(self, msg: String) -> None:
        self._mode = msg.data

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = msg

    # ── GUI construction ─────────────────────────────────────────────────

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

        # ── Tile 1: ArUco Approach control ────────────────────────────────
        approach_tile = tk.Frame(grid, bg='#101010', bd=1, relief=tk.FLAT)
        approach_tile.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        # Title bar
        tk.Label(
            approach_tile, text='ArUco Approach',
            bg='#202020', fg='#d0d0d0', font=('Arial', 11, 'bold'), anchor='w', padx=6,
        ).pack(side=tk.TOP, fill=tk.X)

        # Control bar: ID entry | tag status | Start | Cancel
        ctrl = tk.Frame(approach_tile, bg='#1e1e1e', pady=4)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            ctrl, text='ArUco ID', bg='#1e1e1e', fg='white', font=('Arial', 12, 'bold'),
        ).pack(side=tk.LEFT, padx=(10, 4))

        self.aruco_id_entry = tk.Entry(
            ctrl, width=4, font=('Arial', 20, 'bold'), justify='center',
            bg='#ffffe0', fg='black', relief=tk.SUNKEN, bd=2,
        )
        self.aruco_id_entry.insert(0, '1')
        self.aruco_id_entry.pack(side=tk.LEFT, padx=4, ipady=2)

        self.tag_status_label = tk.Label(
            ctrl, text='—', bg='#1e1e1e', fg='#ffcc00',
            font=('Arial', 11, 'bold'), anchor='w', width=26,
        )
        self.tag_status_label.pack(side=tk.LEFT, padx=8)

        self.cancel_btn = tk.Button(
            ctrl, text='Cancel', command=self._on_cancel_clicked,
            bg='#c62828', fg='white', font=('Arial', 11, 'bold'), padx=10, pady=3,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.RIGHT, padx=(4, 10))

        self.start_btn = tk.Button(
            ctrl, text='Start Approach', command=self._on_start_clicked,
            bg='#2e7d32', fg='white', font=('Arial', 11, 'bold'), padx=10, pady=3,
        )
        self.start_btn.pack(side=tk.RIGHT, padx=4)

        # Bottom rows must be packed BEFORE the expanding camera label so that
        # the expand=True widget does not claim their space.

        # Live feedback row (hidden until a goal is active)
        self.feedback_label = tk.Label(
            approach_tile, text='',
            bg='#1a2a1a', fg='#80ff80', font=('Consolas', 10), anchor='w',
        )
        self.feedback_label.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 3))

        # Goal state status row
        self.approach_status_label = tk.Label(
            approach_tile, text='Status: Ready',
            bg='#2a2a2a', fg='#80d0ff', font=('Arial', 10), anchor='w',
        )
        self.approach_status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 1))

        # Camera preview (expands to fill remaining space)
        self.cam_label = tk.Label(approach_tile, bg='black')
        self.cam_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Tile 2: Consolidated WSKR overlay ────────────────────────────
        def make_tile(title: str, row: int, col: int) -> tk.Label:
            tile = tk.Frame(grid, bg='#101010', bd=1, relief=tk.FLAT)
            tile.grid(row=row, column=col, sticky='nsew', padx=4, pady=4)
            tk.Label(
                tile, text=title, bg='#202020', fg='#d0d0d0',
                font=('Arial', 11, 'bold'), anchor='w', padx=6,
            ).pack(side=tk.TOP, fill=tk.X)
            img_label = tk.Label(tile, bg='black')
            img_label.pack(fill=tk.BOTH, expand=True)
            return img_label

        self.overlay_label = make_tile('Consolidated WSKR Overlay (wskr_overlay/compressed)', 0, 1)

        # ── Tile 3: Telemetry (spans full bottom row) ─────────────────────
        stats_tile = tk.Frame(grid, bg='#101010')
        stats_tile.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
        tk.Label(
            stats_tile, text='Telemetry', bg='#202020', fg='#d0d0d0',
            font=('Arial', 11, 'bold'), anchor='w', padx=6,
        ).pack(side=tk.TOP, fill=tk.X)
        self.numeric_canvas = tk.Canvas(stats_tile, bg='#181818', highlightthickness=0)
        self.numeric_canvas.pack(fill=tk.BOTH, expand=True)

        # ── refresh loop ──────────────────────────────────────────────────
        def refresh() -> None:
            if self._gui_stop.is_set():
                return
            self._render_camera_tile()
            self._render_overlay_tile()
            self._render_numeric_tile()
            self._update_feedback_label()
            root.after(66, refresh)  # ~15 Hz

        root.after(66, refresh)

        def on_close() -> None:
            self._gui_stop.set()
            rclpy.shutdown()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()

    # ── render helpers ────────────────────────────────────────────────────

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

        target_id = self._target_aruco_id()
        tag_text, tag_color = '(enter a valid ID)', '#ffaa00'

        if frame is not None:
            # Detect all ArUco markers and draw them on the frame.
            corners, ids, _ = self._aruco_detector.detectMarkers(frame)
            seen_ids: list[int] = []
            target_bbox_xyxy: Optional[tuple[int, int, int, int]] = None

            if ids is not None:
                # Only highlight the target tag locally when the server is not
                # already tracking (no active tracked_bbox).  Once a goal is
                # active the server's cyan TRACKING box is the authoritative
                # indicator; drawing a second yellow box on the same tag is
                # redundant and confusing.
                server_tracking = self._tracked_bbox is not None
                for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
                    mid = int(marker_id)
                    seen_ids.append(mid)
                    pts = marker_corners[0]
                    x_min = int(np.min(pts[:, 0]))
                    y_min = int(np.min(pts[:, 1]))
                    x_max = int(np.max(pts[:, 0]))
                    y_max = int(np.max(pts[:, 1]))
                    is_target = (target_id is not None and mid == target_id)
                    # Highlight the target only when the server isn't tracking yet.
                    highlight = is_target and not server_tracking
                    color = (0, 255, 255) if highlight else (120, 120, 120)
                    thickness = 3 if highlight else 1
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
                    label_txt = f'ID:{mid}' + (' [TARGET]' if highlight else '')
                    cv2.putText(
                        frame, label_txt, (x_min, max(0, y_min - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                    if is_target:
                        target_bbox_xyxy = (x_min, y_min, x_max, y_max)

            # Tag status text for the control bar label
            if target_id is None:
                tag_text, tag_color = '(enter a valid ID)', '#ffaa00'
            elif target_bbox_xyxy is None:
                seen_txt = ', '.join(str(i) for i in seen_ids) if seen_ids else 'none'
                tag_text = f'ID {target_id} not visible (seen: {seen_txt})'
                tag_color = '#ff6060'
            else:
                cx = (target_bbox_xyxy[0] + target_bbox_xyxy[2]) // 2
                tag_text = f'ID {target_id} visible  x={cx}'
                tag_color = '#80ff80'

            # Also draw the server-tracked bbox when a goal is active.
            if self._tracked_bbox is not None:
                scale = float(frame.shape[1])
                xn, yn, wn, hn = self._tracked_bbox
                x1 = int(xn * scale)
                y1 = int(yn * scale)
                x2 = int((xn + wn) * scale)
                y2 = int((yn + hn) * scale)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(
                    frame, 'TRACKING', (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

        if self.tag_status_label is not None:
            self.tag_status_label.config(text=tag_text, fg=tag_color)

        self._render_image_label(self.cam_label, frame)

    def _render_overlay_tile(self) -> None:
        if self.overlay_label is None:
            return
        with self._overlay_lock:
            frame = None if self._overlay_frame is None else self._overlay_frame.copy()
        # Paint the latest tracked bbox here (not via the range node's compose
        # loop) so freshness is bounded by approach_action_server's publish rate.
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

    def _update_feedback_label(self) -> None:
        if self.feedback_label is None:
            return
        if not self._fb_mode:
            self.feedback_label.config(text='')
            return
        mode = self._fb_mode
        hdg = f'{self._fb_heading:+.1f}°' if self._fb_heading is not None else '—'
        lock_txt = 'visual LOCK' if self._fb_locked else 'no visual'
        lock_color = '#80ff80' if self._fb_locked else '#ffaa00'
        whisker_txt = ''
        if self._fb_whiskers:
            whisker_txt = f'   closest: {int(min(self._fb_whiskers))} mm'
        self.feedback_label.config(
            text=f'mode={mode}   hdg={hdg}   {lock_txt}{whisker_txt}',
            fg=lock_color,
        )

    # ── approach action methods ───────────────────────────────────────────

    def _target_aruco_id(self) -> Optional[int]:
        if self.aruco_id_entry is None:
            return None
        try:
            return int(self.aruco_id_entry.get().strip())
        except ValueError:
            return None

    def _set_approach_status(self, text: str, color: str = '#80d0ff') -> None:
        if self.gui_window is None or self.approach_status_label is None:
            return
        self.gui_window.after(
            0, lambda: self.approach_status_label.config(text=text, fg=color)
        )

    def _reset_approach_buttons(self) -> None:
        if self.gui_window is None:
            return
        def _apply() -> None:
            if self.start_btn is not None:
                self.start_btn.config(state=tk.NORMAL)
            if self.cancel_btn is not None:
                self.cancel_btn.config(state=tk.DISABLED)
        self.gui_window.after(0, _apply)

    def _on_start_clicked(self) -> None:
        target_id = self._target_aruco_id()
        if target_id is None:
            self._set_approach_status('Status: Invalid ArUco ID', '#ff6060')
            return
        goal = ApproachObject.Goal()
        goal.target_type = ApproachObject.Goal.TARGET_BOX
        goal.object_id = target_id
        goal.selected_obj = ImgDetectionData()
        self._set_approach_status(
            f'Status: Dispatching approach for ID {target_id}…', '#ffcc00'
        )
        if self.start_btn is not None:
            self.start_btn.config(state=tk.DISABLED)
        if self.cancel_btn is not None:
            self.cancel_btn.config(state=tk.NORMAL)

        def _dispatch() -> None:
            future = self._approach_client.send_goal_async(
                goal, feedback_callback=self._on_feedback,
            )
            future.add_done_callback(self._on_goal_response)

        threading.Thread(target=_dispatch, daemon=True).start()

    def _on_cancel_clicked(self) -> None:
        handle = self._active_goal_handle
        if handle is None:
            self._set_approach_status('Status: No active goal to cancel', '#ffaa00')
            return
        self._set_approach_status('Status: Cancel requested…', '#ffaa00')
        threading.Thread(target=handle.cancel_goal_async, daemon=True).start()

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._set_approach_status('Status: Goal rejected by server', '#ff6060')
            self._reset_approach_buttons()
            return
        self._active_goal_handle = handle
        self._set_approach_status('Status: Goal accepted — approaching…', '#ffcc00')
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        self._active_goal_handle = None
        self._fb_mode = ''
        self._fb_heading = None
        self._fb_locked = None
        self._fb_whiskers = []
        self._reset_approach_buttons()
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self._set_approach_status(f'Status: Result error — {exc}', '#ff6060')
            return
        if result.movement_success and result.proximity_success:
            self._set_approach_status(
                f'Status: Success — {result.movement_message}', '#80ff80'
            )
        elif result.movement_success:
            self._set_approach_status(
                f'Status: Partial — {result.movement_message}', '#ffaa00'
            )
        else:
            self._set_approach_status(
                f'Status: Failed — {result.movement_message}', '#ff6060'
            )

    def _on_feedback(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        self._fb_mode = fb.tracking_mode or '—'
        self._fb_heading = float(fb.heading_to_target_deg)
        self._fb_locked = bool(fb.visually_tracked)
        self._fb_whiskers = list(fb.whisker_lengths)

    # ── telemetry tile ────────────────────────────────────────────────────

    @staticmethod
    def _whisker_color(mm: float) -> str:
        if mm < 150.0:
            return '#ff6060'
        if mm < 400.0:
            return '#ffcc40'
        return '#60c0ff'

    @staticmethod
    def _draw_target_marker(c: tk.Canvas, x: float, y: float) -> None:
        r = 5.0
        c.create_polygon(
            x, y - r, x + r, y, x, y + r, x - r, y,
            fill='#ff40d0', outline='#ffffff', width=1,
        )

    def _draw_robot_diagram(
        self, c: tk.Canvas, x0: float, y0: float, x1: float, y1: float,
    ) -> None:
        area_w = x1 - x0
        area_h = y1 - y0
        if area_w < 60 or area_h < 60:
            return

        cx = (x0 + x1) / 2.0
        body_h = min(area_w * 0.14, 90.0)
        body_w = body_h * 0.75
        wheel_r = max(body_h * 0.11, 3.0)

        robot_bottom = y1 - 10
        robot_top = robot_bottom - body_h
        body_cx = cx
        body_cy = (robot_top + robot_bottom) / 2.0

        ox, oy = cx, robot_top
        max_ray_px = max(20.0, (oy - y0) - 8)

        whiskers = self._whiskers_mm
        target_whiskers = self._target_whiskers_mm
        scale_mm = 500.0
        if whiskers is not None and whiskers.size > 0:
            n = int(whiskers.size)
            for i, mm in enumerate(whiskers):
                theta_deg = -90.0 + i * (180.0 / max(n - 1, 1))
                theta = math.radians(theta_deg)
                length = max_ray_px * max(0.0, min(1.0, float(mm) / scale_mm))
                sin_t = math.sin(theta)
                cos_t = math.cos(theta)
                ex = ox - length * sin_t
                ey = oy - length * cos_t
                color = self._whisker_color(float(mm))
                c.create_line(ox, oy, ex, ey, fill=color, width=2)
                c.create_oval(ex - 3, ey - 3, ex + 3, ey + 3, fill=color, outline='')
                nudge = 10
                c.create_text(
                    ex - nudge * sin_t, ey - nudge * cos_t,
                    text=f'{int(mm)}', fill='#b0b0b0', font=('Consolas', 8),
                )
                if target_whiskers is not None and target_whiskers.size == n:
                    tmm = float(target_whiskers[i])
                    if tmm < scale_mm - 1.0:
                        t_len = max_ray_px * max(0.0, min(1.0, tmm / scale_mm))
                        self._draw_target_marker(
                            c, ox - t_len * sin_t, oy - t_len * cos_t
                        )

        c.create_rectangle(
            cx - body_w / 2, robot_top, cx + body_w / 2, robot_bottom,
            fill='#2a63d6', outline='#1d4ba3', width=1,
        )
        for wx, wy in (
            (cx - body_w / 2, robot_top), (cx + body_w / 2, robot_top),
            (cx - body_w / 2, robot_bottom), (cx + body_w / 2, robot_bottom),
        ):
            c.create_oval(
                wx - wheel_r, wy - wheel_r, wx + wheel_r, wy + wheel_r,
                fill='#101010', outline='#303030',
            )

        axis_len = body_w * 0.32
        c.create_line(
            body_cx, body_cy, body_cx, body_cy - axis_len,
            fill='#ff6060', width=2, arrow=tk.LAST, arrowshape=(7, 8, 3),
        )
        c.create_line(
            body_cx, body_cy, body_cx - axis_len, body_cy,
            fill='#60ff60', width=2, arrow=tk.LAST, arrowshape=(7, 8, 3),
        )

        heading = self._heading_deg
        if heading is not None:
            theta = math.radians(float(heading))
            arrow_len = max_ray_px * 1.05
            ex = max(x0 + 6, min(x1 - 6, body_cx - arrow_len * math.sin(theta)))
            ey = max(y0 + 6, min(y1 - 6, body_cy - arrow_len * math.cos(theta)))
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
        c.create_text(14, y, anchor='nw', text='Heading to target',
                      fill='#a0a0a0', font=('Arial', 11))
        c.create_text(14, y + 18, anchor='nw', text=heading_txt,
                      fill=heading_color, font=('Consolas', 28, 'bold'))
        c.create_text(w - 14, y + 18, anchor='ne', text=f'mode: {self._mode}',
                      fill=mode_color, font=('Consolas', 14, 'bold'))

        y_diag_top = y + 66
        y_diag_bottom = h - 80
        c.create_text(14, y_diag_top - 18, anchor='nw',
                      text='Whisker fan (mm) + heading arrow',
                      fill='#a0a0a0', font=('Arial', 11))
        self._draw_robot_diagram(c, 14, y_diag_top, w - 14, y_diag_bottom)

        vx = self._cmd_vel.linear.x
        vy = self._cmd_vel.linear.y
        wz_deg = math.degrees(self._cmd_vel.angular.z)
        y_cmd = h - 60
        c.create_text(14, y_cmd, anchor='nw', text='Autopilot cmd_vel (WSKR/cmd_vel)',
                      fill='#a0a0a0', font=('Arial', 11))
        c.create_text(
            14, y_cmd + 20, anchor='nw',
            text=f'Vx={vx:+6.2f} m/s   Vy={vy:+6.2f} m/s   ω={wz_deg:+6.1f}°/s',
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
