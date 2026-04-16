"""Floor Tuner — "which pixels count as floor?"

The floor finder has about a dozen knobs (blur size, color distance
threshold, morph kernel, etc.). Getting them right is the difference
between clean whiskers and noisy ones. This GUI lets you twiddle them
live and see the mask update immediately.

What it does:
    - Runs its own ``Floor`` instance in-process (no restart needed).
    - Shows the raw frame next to a green-tinted overlay of the mask.
    - Sliders for every tunable parameter.
    - "Apply to /WSKR node" — pushes the current values into the running
      ``WSKR_floor`` node via SetParameters, so downstream (whiskers,
      autopilot) sees the change.
    - "Save YAML" — writes the current settings to
      ``config/floor_params.yaml`` so they survive the next launch.

Do this once whenever the lighting or floor material changes.
"""

import threading
import yaml
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import tkinter as tk
from PIL import Image, ImageTk
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from wskr.find_floor import Floor


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# Parameters that are stored as plain integers (not ×0.01 fractions)
_INT_PARAMS = {
    'blur_kernel_size',
    'morph_kernel_size',
    'val_range',
    'color_distance_threshold',
    'gradient_threshold',
    'highlight_thresh',
    'resize_width',
    'resize_height',
}

# Parameters whose slider int value must be divided by 100 to get the real float
_FRAC_PARAMS = {'bottom_sample_fraction', 'bottom_sample_height_fraction'}

# Full-resolution reference used by the single "Resolution scale" slider.
# Matches the default gstreamer_camera pipeline (1920x1080). Each integer
# divisor N produces resize_width = 1920 // N and resize_height = 1080 // N,
# which are always whole numbers — snapping the slider to integer divisors
# is what guarantees integer downsampled pixel counts on both axes.
_BASE_RES_W = 1920
_BASE_RES_H = 1080

# (display label, param key, slider min, slider max, resolution, is_int)
_SLIDER_CONFIGS = [
    ('Blur kernel size',            'blur_kernel_size',              1,   51,   2, True),
    ('Bottom sample fraction',      'bottom_sample_fraction',        0,  100,   1, False),
    ('Bottom sample height frac',   'bottom_sample_height_fraction', 0,  100,   1, False),
    ('Morph kernel size',           'morph_kernel_size',             0,   51,   1, True),
    ('Brightness Distance Threshold', 'val_range',                   0,  255,   1, True),
    ('Colour distance threshold',   'color_distance_threshold',      0,  255,   1, True),
    ('Gradient threshold',          'gradient_threshold',            0, 1000,   1, True),
    ('Highlight threshold',         'highlight_thresh',              0,  255,   1, True),
]

# Student-facing explanations shown as tooltips when hovering a slider row.
_TOOLTIPS: dict[str, str] = {
    'resolution_divisor': (
        'How much the camera frame is shrunk before floor detection runs. '
        'A divisor of N means the frame becomes (1920 / N) × (1080 / N) pixels. '
        'Smaller resolutions (larger divisor) are faster but lose fine detail. '
        'Start at 3 (→ 640×360). Only integer divisors are allowed so both '
        'width and height stay whole numbers.'
    ),
    'blur_kernel_size': (
        'How much to soften the image before the floor test. Bigger values '
        'hide sensor noise and small specks but also blur real edges between '
        'floor and obstacles. Must be an odd number. 9 is a good start.'
    ),
    'bottom_sample_fraction': (
        'How wide the "this is definitely floor" reference patch is, as a '
        'fraction of the image width. 0.50 means the middle 50% of the '
        'bottom of the frame. Make it smaller if the patch catches '
        'non-floor things (feet, wheels, wall at the edges).'
    ),
    'bottom_sample_height_fraction': (
        'How tall the reference patch is, as a fraction of image height. '
        '0.25 = the bottom quarter of the frame. Make it smaller if the '
        'patch reaches up far enough to include walls or furniture.'
    ),
    'morph_kernel_size': (
        'Size of the open/close cleanup step applied to the mask. Bigger '
        'values erase small specks and fill small holes. Must be odd. '
        'If mask has pepper noise, increase; if real thin obstacles are '
        'getting filled in, decrease.'
    ),
    'val_range': (
        'How far a pixel\'s BRIGHTNESS can differ from the reference patch '
        'brightness (0–255 units) before the pixel is rejected as not-floor. '
        'Higher = more permissive (more of the image counts as floor). '
        'Raise this if shadows on the floor are being cut out.'
    ),
    'color_distance_threshold': (
        'How different a pixel\'s COLOUR can be from the reference patch '
        'colour before it\'s rejected as not-floor. Higher = more permissive. '
        'Raise if a slightly different-colored floor area is being excluded; '
        'lower if colored objects are being accepted as floor.'
    ),
    'gradient_threshold': (
        'Edges stronger than this value are treated as obstacle boundaries '
        'and cut the floor mask there. Higher = ignores more edges (more '
        'floor accepted). Lower = cuts the mask at weaker edges, which is '
        'useful if obstacles blend in colour-wise.'
    ),
    'highlight_thresh': (
        'Pixels brighter than this (0–255) are treated as blown-out glare '
        'or reflections and excluded from the floor. Reduce if a glossy '
        'floor is losing its highlights to this threshold.'
    ),
}

# Preview canvas size per pane (two panes side by side)
_PANE_W, _PANE_H = 640, 360


class _Tooltip:
    """A tiny hover-tooltip helper for Tk widgets.

    Tkinter has no built-in tooltip widget; this creates a borderless
    Toplevel on ``<Enter>`` and destroys it on ``<Leave>`` or any click.
    Positioned just below the widget, wrapped to a fixed width.
    """

    def __init__(self, widget: tk.Widget, text: str, wraplength: int = 380) -> None:
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip: Optional[tk.Toplevel] = None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _show(self, _event=None) -> None:
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f'+{x}+{y}')
        tk.Label(
            tip, text=self.text, justify='left',
            background='#2a2a2a', foreground='#f0f0f0',
            relief=tk.SOLID, borderwidth=1,
            wraplength=self.wraplength,
            font=('Arial', 9), padx=6, pady=4,
        ).pack()
        self.tip = tip

    def _hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class FloorTunerNode(Node):
    def __init__(self) -> None:
        super().__init__('floor_tuner')

        self.bridge = CvBridge()
        self.floor = Floor()
        self.floor.enable_floor_mask(True)
        self.lock = threading.Lock()

        # Raw slider int values (fractions stored as int × 100).
        # ``resolution_divisor`` is the single knob that drives resize_width /
        # resize_height via integer floor-division of _BASE_RES_W/H — keeping
        # them locked to the same scale factor so downsampled dimensions are
        # always whole numbers on both axes.
        self._values: dict[str, int] = {
            'resolution_divisor':              3,    # 1920/3 × 1080/3 = 640×360
            'resize_width':                  640,    # derived from divisor
            'resize_height':                 360,    # derived from divisor
            'blur_kernel_size':                9,
            'bottom_sample_fraction':         50,   # 0.50
            'bottom_sample_height_fraction':  25,   # 0.25
            'morph_kernel_size':               5,
            'val_range':                      40,
            'color_distance_threshold':       20,
            'gradient_threshold':             14,
            'highlight_thresh':              230,
        }

        self._apply_to_floor()

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_mask:  Optional[np.ndarray] = None

        self.camera_sub = self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self._camera_callback, IMAGE_QOS)

        # Persistent SetParameters client, re-used across Apply button presses.
        self._wskr_param_client = self.create_client(
            SetParameters, '/WSKR/set_parameters',
        )

        self._canvas_label: Optional[tk.Label] = None
        self._status_var:   Optional[tk.StringVar] = None

        gui_thread = threading.Thread(target=self._gui_run, daemon=True)
        gui_thread.start()

        self.get_logger().info(
            'Floor tuner ready. Subscribed to camera1/image_raw/compressed.')

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _real(self, key: str) -> float:
        """Return real-world value for a slider key."""
        v = self._values[key]
        return v / 100.0 if key in _FRAC_PARAMS else float(v)

    def _recompute_resize_dims(self) -> None:
        """Refresh resize_width/resize_height from the current divisor.

        Called whenever the resolution-scale slider changes. Uses floor
        division so the result is always a pair of whole numbers.
        """
        div = max(1, int(self._values['resolution_divisor']))
        self._values['resize_width'] = _BASE_RES_W // div
        self._values['resize_height'] = _BASE_RES_H // div

    def _apply_to_floor(self) -> None:
        """Push all current slider values into self.floor (call under self.lock)."""
        f = self.floor
        self._recompute_resize_dims()
        f.set_resize_dimensions(
            self._values['resize_width'], self._values['resize_height'])
        bk = self._values['blur_kernel_size']
        if bk % 2 == 0:
            bk = max(3, bk + 1)
        f.set_blur_kernel_size(bk)
        f.set_bottom_sample_size(
            self._real('bottom_sample_fraction'),
            self._real('bottom_sample_height_fraction'))
        f.set_morph_kernel_size(self._values['morph_kernel_size'])
        f.set_val_range(self._values['val_range'])
        f.set_color_distance_threshold(self._values['color_distance_threshold'])
        f.set_gradient_threshold(self._values['gradient_threshold'])
        f.set_highlight_thresh(self._values['highlight_thresh'])

    # ------------------------------------------------------------------ #
    # ROS callback                                                         #
    # ------------------------------------------------------------------ #

    def _camera_callback(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            self.get_logger().error('Frame decode error: cv2.imdecode returned None')
            return

        with self.lock:
            self.floor.find_floor(frame)
            mask = self.floor.get_floor_mask()
            self.latest_frame = frame
            self.latest_mask = mask.copy() if mask is not None else None

    # ------------------------------------------------------------------ #
    # GUI                                                                  #
    # ------------------------------------------------------------------ #

    def _gui_run(self) -> None:
        root = tk.Tk()
        root.title('WSKR Floor Mask Tuner')
        root.geometry('1700x900')
        root.minsize(1300, 700)

        # ── left panel: sliders ─────────────────────────────────────────
        left = tk.Frame(root, bd=2, relief=tk.GROOVE)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        tk.Label(left, text='Floor Parameters',
                 font=('Arial', 11, 'bold')).pack(pady=(4, 8))

        quick_actions = tk.Frame(left)
        quick_actions.pack(fill=tk.X, padx=4, pady=(0, 8))
        tk.Button(
            quick_actions,
            text='Save YAML',
            command=self._save_yaml,
            bg='lightgreen',
            font=('Arial', 10, 'bold'),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            quick_actions,
            text='Apply to /WSKR node',
            command=self._apply_to_wskr_node,
            bg='lightyellow',
            font=('Arial', 10, 'bold'),
        ).pack(side=tk.LEFT, padx=2)

        slider_vars: dict[str, tk.IntVar] = {}

        # ── Resolution scale (single slider controlling both axes) ──────
        # Rendered first because everything else operates on the downsampled
        # frame, so students should see / understand this knob before the rest.
        res_row = tk.Frame(left)
        res_row.pack(fill=tk.X, padx=4, pady=2)

        res_var = tk.IntVar(value=self._values['resolution_divisor'])
        slider_vars['resolution_divisor'] = res_var

        res_val_label = tk.Label(res_row, width=10, anchor='e')
        res_val_label.pack(side=tk.RIGHT)
        tk.Label(res_row, text='Resolution scale', width=28, anchor='w').pack(side=tk.LEFT)

        def _on_res_change(*_) -> None:
            # Slider is constrained to integer divisors 1..10, so both
            # _BASE_RES_W // div and _BASE_RES_H // div are guaranteed integers.
            div = max(1, int(res_var.get()))
            self._values['resolution_divisor'] = div
            with self.lock:
                self._apply_to_floor()
            w = self._values['resize_width']
            h = self._values['resize_height']
            res_val_label.config(text=f'1/{div}  {w}×{h}')

        tk.Scale(
            res_row, variable=res_var, from_=1, to=10,
            orient=tk.HORIZONTAL, resolution=1,
            length=230, showvalue=False, command=_on_res_change,
        ).pack(side=tk.LEFT)
        _on_res_change()

        _Tooltip(res_row, _TOOLTIPS['resolution_divisor'])

        # ── Other floor parameters ──────────────────────────────────────
        for label, key, lo, hi, res, _ in _SLIDER_CONFIGS:
            row = tk.Frame(left)
            row.pack(fill=tk.X, padx=4, pady=2)

            var = tk.IntVar(value=self._values[key])
            slider_vars[key] = var

            val_label = tk.Label(row, width=6, anchor='e')
            val_label.pack(side=tk.RIGHT)

            tk.Label(row, text=label, width=28, anchor='w').pack(side=tk.LEFT)

            def _make_cb(k, vl, lo_=lo):
                def cb(*_):
                    raw = slider_vars[k].get()
                    # Gaussian blur requires an odd kernel size; snap to nearest odd.
                    # Morph kernel has no such requirement — even sizes are valid.
                    if k == 'blur_kernel_size' and raw % 2 == 0:
                        raw = max(lo_, raw + 1)
                        slider_vars[k].set(raw)
                    self._values[k] = raw
                    real = self._real(k)
                    vl.config(text=f'{real:.2f}' if k in _FRAC_PARAMS else str(int(real)))
                    with self.lock:
                        self._apply_to_floor()
                return cb

            cb = _make_cb(key, val_label)
            tk.Scale(
                row, variable=var, from_=lo, to=hi,
                orient=tk.HORIZONTAL, resolution=res,
                length=230, showvalue=False, command=cb,
            ).pack(side=tk.LEFT)
            cb()  # initialise value label

            tip_text = _TOOLTIPS.get(key)
            if tip_text:
                _Tooltip(row, tip_text)

        # ── right panel: live preview canvas ────────────────────────────
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        tk.Label(right, text='camera1/image_raw/compressed  |  mask overlay',
                 font=('Arial', 10)).pack()

        self._canvas_label = tk.Label(right, bg='black')
        self._canvas_label.pack(fill=tk.BOTH, expand=True)

        # ── bottom bar ───────────────────────────────────────────────────
        bar = tk.Frame(root)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        self._status_var = tk.StringVar(value='Ready')
        tk.Label(bar, textvariable=self._status_var,
                 fg='blue', anchor='w').pack(side=tk.LEFT, padx=4)

        tk.Button(
            bar, text='Apply to /WSKR node',
            command=self._apply_to_wskr_node,
            bg='lightyellow', font=('Arial', 10, 'bold'),
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            bar, text='Save YAML',
            command=self._save_yaml,
            bg='lightgreen', font=('Arial', 10, 'bold'),
        ).pack(side=tk.RIGHT, padx=4)

        root.after(100, self._refresh_preview)

        def on_close() -> None:
            rclpy.shutdown()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()

    def _refresh_preview(self) -> None:
        """Update the canvas at ~10 Hz from the Tkinter event loop."""
        with self.lock:
            frame = self.latest_frame
            mask  = self.latest_mask

        if frame is not None:
            cam = cv2.resize(frame, (_PANE_W, _PANE_H))

            if mask is not None:
                m = cv2.resize(mask, (_PANE_W, _PANE_H),
                               interpolation=cv2.INTER_NEAREST)
                overlay = cam.copy().astype(np.float32)
                alpha = (m > 127).astype(np.float32)[:, :, np.newaxis]
                green = np.zeros_like(overlay)
                green[:, :, 1] = 200.0
                overlay = (overlay * (1.0 - 0.45 * alpha)
                           + green * 0.45 * alpha).astype(np.uint8)
                combined = np.hstack([cam, overlay])
            else:
                combined = np.hstack([cam, np.zeros_like(cam)])

            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._canvas_label.config(image=photo)
            self._canvas_label.image = photo  # keep reference

        self._canvas_label.after(100, self._refresh_preview)

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def _apply_to_wskr_node(self) -> None:
        """Push current values to the running /WSKR node via SetParameters.
        Reuses the persistent client created in __init__."""
        if not self._wskr_param_client.wait_for_service(timeout_sec=1.0):
            self._status_var.set('Error: /WSKR node not reachable')
            return

        params = []
        for key, raw in self._values.items():
            # resolution_divisor is a local-only UI knob; the /WSKR node knows
            # resize_width / resize_height, which were already derived from it.
            if key == 'resolution_divisor':
                continue
            real = self._real(key)
            pv = ParameterValue()
            if key in _INT_PARAMS:
                pv.type = ParameterType.PARAMETER_INTEGER
                pv.integer_value = int(real)
            else:
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = float(real)
            params.append(Parameter(name=key, value=pv))

        req = SetParameters.Request(parameters=params)
        future = self._wskr_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.done() and future.result() is not None:
            self._status_var.set('Applied to /WSKR node ✓')
        else:
            self._status_var.set('Error: SetParameters call failed')

    def _save_yaml(self) -> None:
        """Overwrite floor_params.yaml with current slider values."""
        try:
            share = get_package_share_directory('wskr')
            yaml_path = Path(share) / 'config' / 'floor_params.yaml'

            data = {
                'WSKR': {
                    'ros__parameters': {
                        'resize_width':                   int(self._real('resize_width')),
                        'resize_height':                  int(self._real('resize_height')),
                        'blur_kernel_size':               int(self._real('blur_kernel_size')),
                        'bottom_sample_fraction':         self._real('bottom_sample_fraction'),
                        'bottom_sample_height_fraction':  self._real('bottom_sample_height_fraction'),
                        'morph_kernel_size':              int(self._real('morph_kernel_size')),
                        'val_range':                      int(self._real('val_range')),
                        'color_distance_threshold':       int(self._real('color_distance_threshold')),
                        'gradient_threshold':             int(self._real('gradient_threshold')),
                        'highlight_thresh':               int(self._real('highlight_thresh')),
                    }
                }
            }
            yaml_path.write_text(
                yaml.dump(data, default_flow_style=False, sort_keys=False))
            self._status_var.set(f'Saved → {yaml_path}')
        except Exception as exc:
            self._status_var.set(f'Save failed: {exc}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FloorTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
