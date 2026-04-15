#!/usr/bin/env python3
"""Start ArUco Approach — the "go drive at this tag" GUI.

A simple tkinter window with three things you care about:
    1. A live preview of the main camera (camera1/image_raw/compressed).
       Detected ArUco tags are drawn in grey; the one matching the ID you
       typed in is drawn in bright yellow and labelled ``[TARGET]``.
    2. A big, editable ArUco ID field.
    3. A green "Start Approach" button that sends the ``WSKR/approach_object``
       action. The red "Cancel" button aborts an in-flight approach.

Use this to verify the WSKR stack end-to-end: launch everything with
``wskr.launch.py``, open this GUI, point the camera at a DICT_4X4_50 tag,
type the tag's ID, and click Start. The robot should drive up to it.
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
import tkinter as tk
from PIL import Image, ImageTk
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

from robot_interfaces.msg import ImgDetectionData
from wskr.action import ApproachObject


ARUCO_DICT = cv2.aruco.DICT_4X4_50  # must match approach_action_server


class StartArucoApproachNode(Node):
    def __init__(self) -> None:
        super().__init__('start_aruco_approach')

        self.bridge = CvBridge()

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict)

        self.camera_sub = self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self._camera_callback, IMAGE_QOS,
        )
        self.heading_sub = self.create_subscription(
            Float32, 'WSKR/heading_to_target', self._on_heading, 10
        )
        self.mode_sub = self.create_subscription(
            String, 'WSKR/tracking_mode', self._on_mode, 10
        )

        self.approach_client = ActionClient(self, ApproachObject, 'WSKR/approach_object')

        self._latest_heading: Optional[float] = None
        self._latest_mode: str = 'unknown'
        self._active_goal_handle = None

        self.gui_window: Optional[tk.Tk] = None
        self.preview_label: Optional[tk.Label] = None
        self.aruco_id_entry: Optional[tk.Entry] = None
        self.status_label: Optional[tk.Label] = None
        self.tag_status_label: Optional[tk.Label] = None
        self.telemetry_label: Optional[tk.Label] = None
        self.start_btn: Optional[tk.Button] = None
        self.cancel_btn: Optional[tk.Button] = None
        self._gui_stop = threading.Event()

        self._visible_ids: list[int] = []
        self._target_bbox_xyxy: Optional[tuple[int, int, int, int]] = None

        self.get_logger().info('Waiting for approach_object action server...')
        while not self.approach_client.wait_for_server(timeout_sec=1.0) and rclpy.ok():
            self.get_logger().info('...still waiting for ApproachObject action server')

        self._start_gui()

    def _camera_callback(self, msg: CompressedImage) -> None:
        # Half-res decode is plenty for local ArUco detection + preview.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            self.get_logger().warn('Failed to decode compressed camera frame.')
            return
        with self._frame_lock:
            self._latest_frame = frame

    def _on_heading(self, msg: Float32) -> None:
        self._latest_heading = float(msg.data)

    def _on_mode(self, msg: String) -> None:
        self._latest_mode = msg.data

    def _start_gui(self) -> None:
        t = threading.Thread(target=self._gui_run, daemon=True)
        t.start()

    def _gui_run(self) -> None:
        root = tk.Tk()
        root.title('Start ArUco Approach')
        root.geometry('1000x800')
        self.gui_window = root

        # Top: prominent ArUco ID input + controls
        top = tk.Frame(root, bg='#1e1e1e', pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top, text='ArUco ID', bg='#1e1e1e', fg='white', font=('Arial', 20, 'bold')
        ).pack(side=tk.LEFT, padx=(14, 6))

        self.aruco_id_entry = tk.Entry(
            top, width=5, font=('Arial', 28, 'bold'), justify='center',
            bg='#ffffe0', fg='black', relief=tk.SUNKEN, bd=3,
        )
        self.aruco_id_entry.insert(0, '1')
        self.aruco_id_entry.pack(side=tk.LEFT, padx=6, ipady=4)

        self.tag_status_label = tk.Label(
            top, text='—', bg='#1e1e1e', fg='#ffcc00', font=('Arial', 14, 'bold'),
            width=22, anchor='w',
        )
        self.tag_status_label.pack(side=tk.LEFT, padx=12)

        self.start_btn = tk.Button(
            top, text='Start Approach', command=self._on_start_clicked,
            bg='#2e7d32', fg='white', font=('Arial', 14, 'bold'), padx=16, pady=6,
            activebackground='#1b5e20',
        )
        self.start_btn.pack(side=tk.RIGHT, padx=(4, 14))

        self.cancel_btn = tk.Button(
            top, text='Cancel', command=self._on_cancel_clicked,
            bg='#c62828', fg='white', font=('Arial', 12, 'bold'), padx=12, pady=6,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.RIGHT, padx=4)

        # Middle: live preview
        preview_frame = tk.Frame(root, bg='black')
        preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.preview_label = tk.Label(preview_frame, bg='black')
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Bottom: status
        bottom = tk.Frame(root, bg='#2a2a2a')
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(
            bottom, text='Status: Ready', bg='#2a2a2a', fg='#80d0ff',
            font=('Arial', 11), anchor='w',
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=4)
        self.telemetry_label = tk.Label(
            bottom, text='mode=— heading=—', bg='#2a2a2a', fg='#c0c0c0',
            font=('Consolas', 11), anchor='e',
        )
        self.telemetry_label.pack(side=tk.RIGHT, padx=8, pady=4)

        def refresh_preview() -> None:
            if self._gui_stop.is_set():
                return
            self._update_preview()
            self._update_telemetry_text()
            root.after(50, refresh_preview)  # ~20 Hz

        root.after(50, refresh_preview)

        def on_close() -> None:
            self._gui_stop.set()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()

    def _target_id(self) -> Optional[int]:
        if self.aruco_id_entry is None:
            return None
        try:
            return int(self.aruco_id_entry.get().strip())
        except ValueError:
            return None

    def _update_preview(self) -> None:
        if self.preview_label is None:
            return
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return

        target_id = self._target_id()
        corners, ids, _ = self._aruco_detector.detectMarkers(frame)

        seen_ids: list[int] = []
        target_bbox: Optional[tuple[int, int, int, int]] = None
        if ids is not None:
            flat = ids.flatten().tolist()
            for marker_corners, marker_id in zip(corners, flat):
                mid = int(marker_id)
                seen_ids.append(mid)
                pts = marker_corners[0]
                x_min = int(np.min(pts[:, 0]))
                y_min = int(np.min(pts[:, 1]))
                x_max = int(np.max(pts[:, 0]))
                y_max = int(np.max(pts[:, 1]))
                is_target = (target_id is not None and mid == target_id)
                color = (0, 255, 255) if is_target else (120, 120, 120)
                thickness = 4 if is_target else 1
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
                label = f'ID:{mid}' + (' [TARGET]' if is_target else '')
                cv2.putText(
                    frame, label, (x_min, max(0, y_min - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )
                if is_target:
                    target_bbox = (x_min, y_min, x_max, y_max)

        self._visible_ids = seen_ids
        self._target_bbox_xyxy = target_bbox

        if target_id is None:
            tag_text, tag_color = '(enter a valid ID)', '#ffaa00'
        elif target_bbox is None:
            seen_txt = ', '.join(str(i) for i in seen_ids) if seen_ids else 'none'
            tag_text, tag_color = f'ID {target_id} NOT visible (seen: {seen_txt})', '#ff6060'
        else:
            x1, y1, x2, y2 = target_bbox
            cx = (x1 + x2) // 2
            tag_text, tag_color = f'ID {target_id} visible @ x={cx}', '#80ff80'
        if self.tag_status_label is not None:
            self.tag_status_label.config(text=tag_text, fg=tag_color)

        # Fit to the preview label area.
        lw = max(self.preview_label.winfo_width(), 1)
        lh = max(self.preview_label.winfo_height(), 1)
        sh, sw = frame.shape[:2]
        scale = min(lw / sw, lh / sh) if (sw > 0 and sh > 0) else 1.0
        new_w = max(int(sw * scale), 1)
        new_h = max(int(sh * scale), 1)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((lh, lw, 3), dtype=np.uint8)
        off_x = (lw - new_w) // 2
        off_y = (lh - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.config(image=photo)
        self.preview_label.image = photo  # keep reference

    def _update_telemetry_text(self) -> None:
        if self.telemetry_label is None:
            return
        heading = '—' if self._latest_heading is None else f'{self._latest_heading:+.1f}°'
        self.telemetry_label.config(text=f'mode={self._latest_mode}  heading={heading}')

    def _set_status(self, text: str, color: str = '#80d0ff') -> None:
        if self.gui_window is None or self.status_label is None:
            return
        self.gui_window.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _on_start_clicked(self) -> None:
        target_id = self._target_id()
        if target_id is None:
            self._set_status('Status: Invalid ArUco ID', '#ff6060')
            return

        goal = ApproachObject.Goal()
        goal.target_type = ApproachObject.Goal.TARGET_BOX
        goal.object_id = target_id
        goal.selected_obj = ImgDetectionData()

        self._set_status(f'Status: Dispatching approach for ArUco ID {target_id}...', '#ffcc00')
        if self.start_btn is not None:
            self.start_btn.config(state=tk.DISABLED)
        if self.cancel_btn is not None:
            self.cancel_btn.config(state=tk.NORMAL)

        def dispatch() -> None:
            send_future = self.approach_client.send_goal_async(goal)
            send_future.add_done_callback(self._on_goal_response)

        threading.Thread(target=dispatch, daemon=True).start()

    def _on_cancel_clicked(self) -> None:
        handle = self._active_goal_handle
        if handle is None:
            self._set_status('Status: No active goal to cancel', '#ffaa00')
            return
        self._set_status('Status: Cancel requested', '#ffaa00')
        threading.Thread(target=handle.cancel_goal_async, daemon=True).start()

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._set_status('Status: Goal rejected by server', '#ff6060')
            self._reset_buttons()
            return
        self._active_goal_handle = handle
        self._set_status('Status: Goal accepted — approaching...', '#ffcc00')
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        self._active_goal_handle = None
        self._reset_buttons()
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self._set_status(f'Status: Result error: {exc}', '#ff6060')
            return
        if result.movement_success and result.proximity_success:
            self._set_status(f'Status: Success — {result.movement_message}', '#80ff80')
        elif result.movement_success:
            self._set_status(f'Status: Partial success — {result.movement_message}', '#ffaa00')
        else:
            self._set_status(f'Status: Failed — {result.movement_message}', '#ff6060')

    def _reset_buttons(self) -> None:
        if self.gui_window is None:
            return
        def apply() -> None:
            if self.start_btn is not None:
                self.start_btn.config(state=tk.NORMAL)
            if self.cancel_btn is not None:
                self.cancel_btn.config(state=tk.DISABLED)
        self.gui_window.after(0, apply)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StartArucoApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
