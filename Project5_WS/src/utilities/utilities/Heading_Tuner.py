#!/usr/bin/env python3
"""Heading Tuner — "why are my meridian lines crooked?"

The heading-from-bbox calculation assumes a specific camera tilt and optical
centre. If those are off, the dashed heading meridians drawn on the WSKR
overlay won't line up with real-world verticals. This GUI lets you slide
the ``y_offset`` parameter until they do.

What it does:
    - Shows a live preview of ``wskr_overlay/compressed`` so you can see
      the meridians move as you drag the slider.
    - Pushes the new ``y_offset`` to both ``WSKR_range`` and
      ``WSKR_approach_action`` via the standard ROS SetParameters service.
    - "Save YAML" writes the current value back to the source-tree
      ``config/lens_params.yaml`` so it survives the next launch.

This tool is the standard calibration step after moving/changing the
camera mount.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import tkinter as tk
import yaml
from PIL import Image, ImageTk
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from WSKR.lens_model import LensParams, compute_heading_rad, project_meridian_normalized


MERIDIAN_DEGS = (-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90)


def _draw_dashed_polyline(
    img: np.ndarray,
    pts: list,
    color: tuple,
    thickness: int,
) -> None:
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

HEADING_OVERLAY_NODE = 'WSKR_range'
APPROACH_NODE = 'WSKR_approach_action'

LENS_NODES = (HEADING_OVERLAY_NODE, APPROACH_NODE)

# Keys that live in lens_params.yaml. Editing is only exposed for y_offset,
# but the YAML rewriter preserves the other keys.
PERSISTED_KEYS = (
    'image_width', 'image_height', 'x_min', 'x_max',
    'cy', 'hfov_deg', 'tilt_deg', 'y_offset',
)


def _source_config_path_from_install_share() -> Optional[Path]:
    """Best-effort resolve of the source lens_params.yaml from the install share.

    Typical layout: <ws>/install/wskr/share/wskr/config/lens_params.yaml
                    <ws>/src/WSKR/config/lens_params.yaml
    """
    try:
        share = Path(get_package_share_directory('wskr'))
    except Exception:  # noqa: BLE001
        return None
    parts = list(share.parts)
    if 'install' not in parts:
        return None
    ws_root = Path(*parts[: parts.index('install')])
    candidate = ws_root / 'src' / 'WSKR' / 'config' / 'lens_params.yaml'
    return candidate if candidate.exists() else None


class HeadingTunerNode(Node):
    def __init__(self) -> None:
        super().__init__('heading_tuner')

        self.declare_parameter('source_config_path', '')

        self.bridge = CvBridge()

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # Diagnostic: subscribe to the raw camera feed and draw meridians
        # locally so the user can see them against a clean image (no floor
        # mask, whiskers, or readout strip from wskr_overlay).
        self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self._on_camera, IMAGE_QOS
        )

        self._param_clients = {
            n: self.create_client(SetParameters, f'/{n}/set_parameters')
            for n in LENS_NODES
        }

        self._gui_stop = threading.Event()
        self._current_y_offset = 0.0

        # Cache the non-y_offset lens params once at startup so the click-probe
        # heading calculation matches what the running nodes compute. y_offset
        # is overlaid live from the slider. If the YAML can't be found we fall
        # back to LensParams() defaults.
        self._baseline_lens_params: LensParams = self._load_baseline_lens_params()

        # Preview letterbox transform, written by the refresh loop and read by
        # the click handler. Both run on the GUI thread so no lock needed.
        # Tuple fields: (scale, new_w, new_h, off_x, off_y, image_w, image_h)
        self._preview_transform: Optional[Tuple[float, int, int, int, int, int, int]] = None

        # Last click-probe result — rendered as a cv2 overlay directly onto the
        # canvas so it's immune to Tkinter layout issues.
        self._probe_text: str = ''

        self._config_path = self._resolve_source_config_path()
        if self._config_path is None:
            self.get_logger().warn(
                'Could not resolve source lens_params.yaml path. Save button will '
                'fall back to install share (changes will not survive a rebuild). '
                "Set the 'source_config_path' param to override."
            )

        threading.Thread(target=self._run_gui, daemon=True).start()
        self.get_logger().info('Heading tuner ready.')

    def _resolve_source_config_path(self) -> Optional[Path]:
        override = str(self.get_parameter('source_config_path').value).strip()
        if override:
            return Path(override)
        return _source_config_path_from_install_share()

    def _load_baseline_lens_params(self) -> LensParams:
        """Read lens_params.yaml from the install share and build a LensParams
        from the first tuned-node section. Returns defaults if the file or
        section isn't present (the click-probe then uses LensParams() values).
        """
        defaults = LensParams()
        try:
            share = Path(get_package_share_directory('wskr'))
        except Exception:  # noqa: BLE001
            return defaults
        path = share / 'config' / 'lens_params.yaml'
        if not path.exists():
            return defaults
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Could not parse lens_params.yaml: {exc}')
            return defaults

        for key in LENS_NODES:
            section = (data.get(key) or {}).get('ros__parameters') or {}
            if section:
                return LensParams(
                    x_min=float(section.get('x_min', defaults.x_min)),
                    x_max=float(section.get('x_max', defaults.x_max)),
                    cy=float(section.get('cy', defaults.cy)),
                    hfov_deg=float(section.get('hfov_deg', defaults.hfov_deg)),
                    tilt_deg=float(section.get('tilt_deg', defaults.tilt_deg)),
                    y_offset=float(section.get('y_offset', defaults.y_offset)),
                )
        return defaults

    def _on_camera(self, msg: CompressedImage) -> None:
        # Half-res decode is plenty for an interactive preview.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            self.get_logger().warn('camera decode failed: cv2.imdecode returned None')
            return
        with self._frame_lock:
            self._latest_frame = frame

    def _draw_meridians(self, frame: np.ndarray) -> None:
        """Draw dashed heading meridians on ``frame`` using the CURRENT slider
        value for y_offset (so meridian positions update live as the user
        drags). Other lens params come from the baseline loaded at startup."""
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return
        base = self._baseline_lens_params
        params = LensParams(
            x_min=base.x_min, x_max=base.x_max, cy=base.cy,
            hfov_deg=base.hfov_deg, tilt_deg=base.tilt_deg,
            y_offset=float(self._current_y_offset),
        )
        aspect = float(h) / float(w)
        for deg in MERIDIAN_DEGS:
            pts_norm = project_meridian_normalized(deg, params, aspect=aspect)
            if len(pts_norm) < 2:
                continue
            pts = [(int(u * w), int(v * w)) for (u, v) in pts_norm]
            _draw_dashed_polyline(frame, pts, color=(0, 0, 255), thickness=2)
            mid = pts[len(pts) // 2]
            cv2.putText(
                frame, f'{deg}', (mid[0] + 4, mid[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
            )

    # ---- parameter pushing ------------------------------------------------

    def _send_y_offset(self, value: float) -> None:
        """Fire SetParameters on every lens-aware node; ignore failures (GUI stays alive)."""
        param = Parameter()
        param.name = 'y_offset'
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
        param.value = pv

        req = SetParameters.Request()
        req.parameters = [param]

        for node_name, client in self._param_clients.items():
            if not client.service_is_ready():
                continue
            future = client.call_async(req)
            future.add_done_callback(
                lambda f, n=node_name: self._on_set_done(f, n)
            )

    def _on_set_done(self, future, node_name: str) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'SetParameters on {node_name} failed: {exc}')
            return
        if resp is None:
            return
        for r in resp.results:
            if not r.successful:
                self.get_logger().warn(
                    f'{node_name} rejected y_offset: {r.reason}'
                )

    # ---- persistence ------------------------------------------------------

    def save_yaml(self, y_offset: float) -> tuple[bool, str]:
        """Write y_offset into every lens-aware node section of lens_params.yaml."""
        path = self._config_path
        if path is None:
            return False, 'No config path resolved (set source_config_path param)'
        if not path.exists():
            return False, f'Config file not found: {path}'

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            return False, f'Read failed: {exc}'

        changed = False
        for node_key in LENS_NODES:
            section = data.setdefault(node_key, {})
            params = section.setdefault('ros__parameters', {})
            params['y_offset'] = float(y_offset)
            changed = True

        if not changed:
            return False, 'No matching sections found in YAML'

        try:
            with open(path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception as exc:  # noqa: BLE001
            return False, f'Write failed: {exc}'

        return True, str(path)

    # ---- GUI --------------------------------------------------------------

    def _run_gui(self) -> None:
        BG = '#1e1e1e'
        PANEL_BG = '#2a2a2a'
        FG = 'white'
        ACCENT = '#80d0ff'
        VALUE_FG = '#ffcc00'

        root = tk.Tk()
        root.title('Heading Tuner')
        root.configure(bg=BG)
        root.geometry('1000x760')

        # Top control row
        top = tk.Frame(root, bg=BG, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top, text='y_offset', bg=BG, fg=ACCENT,
            font=('Arial', 14, 'bold'),
        ).pack(side=tk.LEFT, padx=(14, 8))

        # y_offset is in width-normalized image units. The old pixel slider
        # range [-1500, 500] corresponds to approximately [-0.78, 0.26].
        y_var = tk.DoubleVar(value=0.0)
        slider = tk.Scale(
            top, variable=y_var, from_=-0.80, to=0.30, resolution=0.001,
            orient=tk.HORIZONTAL, length=500, bg=PANEL_BG, fg=FG,
            troughcolor='#555', activebackground='#5599ff',
            highlightthickness=0, showvalue=False,
        )
        slider.pack(side=tk.LEFT, padx=4)

        value_label_var = tk.StringVar(value='0.000')
        tk.Label(
            top, textvariable=value_label_var, bg=BG, fg=VALUE_FG,
            font=('Consolas', 14, 'bold'), width=8,
        ).pack(side=tk.LEFT, padx=8)

        status_var = tk.StringVar(value='Status: Ready')

        def apply_value(_evt=None):
            val = float(y_var.get())
            self._current_y_offset = val
            value_label_var.set(f'{val:+.3f}')
            self._send_y_offset(val)

        slider.config(command=lambda _v: apply_value())

        def do_reset():
            y_var.set(0.0)
            apply_value()

        def do_save():
            ok, info = self.save_yaml(self._current_y_offset)
            if ok:
                status_var.set(f'Status: Saved y_offset={self._current_y_offset:+.4f} to {info}')
            else:
                status_var.set(f'Status: Save failed — {info}')

        tk.Button(
            top, text='Reset', command=do_reset,
            bg='#555', fg='white', font=('Arial', 11, 'bold'),
            padx=10, pady=4,
        ).pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(
            top, text='Save', command=do_save,
            bg='#2e7d32', fg='white', font=('Arial', 11, 'bold'),
            padx=14, pady=4, activebackground='#1b5e20',
        ).pack(side=tk.LEFT, padx=4)

        # Status bar — only bottom widget; packed before preview so it keeps its space.
        tk.Label(
            root, textvariable=status_var, bg='#222', fg='#80d0ff',
            font=('Arial', 10), anchor='w',
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

        # Preview — fills all remaining space.
        preview = tk.Frame(root, bg='black')
        preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        preview_label = tk.Label(preview, bg='black', cursor='crosshair')
        preview_label.pack(fill=tk.BOTH, expand=True)

        def on_preview_click(event):
            xf = self._preview_transform
            if xf is None:
                return
            scale, new_w, new_h, off_x, off_y, image_w, image_h = xf
            dx = event.x - off_x
            dy = event.y - off_y
            if dx < 0 or dx >= new_w or dy < 0 or dy >= new_h or scale <= 0:
                self._probe_text = ''
                return
            img_x = dx / scale
            img_y = dy / scale
            # Lens model is width-normalized on BOTH axes — divide by image
            # width, NOT height. This is the same convention compute_heading_rad
            # uses internally (keeps aspect isotropic).
            u_norm = img_x / float(image_w)
            v_norm = img_y / float(image_w)
            base = self._baseline_lens_params
            live_params = LensParams(
                x_min=base.x_min, x_max=base.x_max, cy=base.cy,
                hfov_deg=base.hfov_deg, tilt_deg=base.tilt_deg,
                y_offset=float(self._current_y_offset),
            )
            heading_deg = math.degrees(compute_heading_rad(u_norm, v_norm, live_params))
            self._probe_text = (
                f'({int(round(img_x))}, {int(round(img_y))})  heading {heading_deg:+.2f}\xb0'
            )

        preview_label.bind('<Button-1>', on_preview_click)

        def refresh():
            if self._gui_stop.is_set():
                return
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            # Draw meridians on the raw frame (at full-frame resolution) so
            # cv2.putText and line thickness look consistent regardless of
            # how the preview is letterboxed into the widget.
            if frame is not None:
                self._draw_meridians(frame)

            lw = max(preview_label.winfo_width(), 1)
            lh = max(preview_label.winfo_height(), 1)
            if frame is not None and lw > 1 and lh > 1:
                sh, sw = frame.shape[:2]
                scale = min(lw / sw, lh / sh)
                new_w = max(int(sw * scale), 1)
                new_h = max(int(sh * scale), 1)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((lh, lw, 3), dtype=np.uint8)
                off_x = (lw - new_w) // 2
                off_y = (lh - new_h) // 2
                canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

                # Render click-probe result as an overlay pinned to the bottom-left
                # of the canvas — immune to Tkinter layout/resize issues.
                if self._probe_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fscale, thick = 0.65, 2
                    (tw, th), bl = cv2.getTextSize(self._probe_text, font, fscale, thick)
                    tx, ty = 10, lh - 10
                    cv2.rectangle(
                        canvas,
                        (tx - 4, ty - th - 4), (tx + tw + 4, ty + bl + 2),
                        (0, 0, 0), -1,
                    )
                    cv2.putText(
                        canvas, self._probe_text, (tx, ty),
                        font, fscale, (0, 204, 255), thick, cv2.LINE_AA,
                    )

                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                preview_label.config(image=photo)
                preview_label.image = photo  # keep reference
                # Record how this frame was mapped into the widget so click
                # coords can be inverted into image pixels.
                self._preview_transform = (scale, new_w, new_h, off_x, off_y, sw, sh)

            root.after(50, refresh)

        root.after(50, refresh)

        def on_close():
            self._gui_stop.set()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeadingTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
