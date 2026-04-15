"""WSKR Whisker Range Node — "how far can I drive in each direction?"

Reads the black-and-white floor mask and walks a set of pre-calibrated
"whisker" rays outward from the robot's feet (11 of them by default, fanned
across the forward hemisphere). Each whisker stops at the first non-floor
pixel it hits. The distance along that ray — in millimeters — is the
drive-distance estimate for that direction.

Topics:
    subscribes   WSKR/floor_mask           — binary floor mask (mono8).
    subscribes   WSKR/heading_to_target    — fused heading (for readout only).
    subscribes   WSKR/tracking_mode        — current fusion mode.
    subscribes   WSKR/cmd_vel              — last autopilot twist (readout).
    publishes    WSKR/whisker_lengths      — 11 floats in millimetres.
    publishes    wskr_overlay/compressed   — JPEG diagnostic overlay with
                                             whiskers, meridians, and readout.

The whisker lengths feed the autopilot MLP. The overlay is a diagnostic
published only when something (a dashboard) is actually subscribed.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Float32MultiArray, String

from .lens_model import LensParams, project_meridian_normalized


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

MERIDIAN_DEGS = (-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90)

LENS_PARAM_NAMES = ('x_min', 'x_max', 'cy', 'hfov_deg', 'tilt_deg', 'y_offset')

READOUT_STRIP_HEIGHT = 60


def _draw_dashed_polyline(
    img: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw every other segment of a polyline to approximate a dashed curve.

    Input pts are already sampled densely enough (every 2° in phi) that
    skipping alternate segments yields a visually uniform dash pattern.
    """
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


class WSKRRangeNode(Node):
    def __init__(self) -> None:
        super().__init__('WSKR_range')

        self.bridge = CvBridge()

        default_cal = str(Path(get_package_share_directory('wskr')) / 'config' / 'FirstCal.json')
        self.declare_parameter('calibration_file', default_cal)
        self.declare_parameter('max_range_mm', 500.0)
        self.declare_parameter('sample_step_mm', 1.0)

        # Lens params for meridian projection (normalized; shared with
        # WSKR_approach_action via config/lens_params.yaml).
        _lp = LensParams()
        self.declare_parameter('x_min', _lp.x_min)
        self.declare_parameter('x_max', _lp.x_max)
        self.declare_parameter('cy', _lp.cy)
        self.declare_parameter('hfov_deg', _lp.hfov_deg)
        self.declare_parameter('tilt_deg', _lp.tilt_deg)
        self.declare_parameter('y_offset', _lp.y_offset)

        cal_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.max_range_mm = float(self.get_parameter('max_range_mm').value)
        self.sample_step_mm = float(self.get_parameter('sample_step_mm').value)

        self._params_lock = threading.Lock()
        self._params = self._read_lens_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.whisker_models, self.whisker_labels = self._load_calibration(cal_file)

        # Latest values from downstream topics (used only for the readout strip).
        self._latest_heading_deg: Optional[float] = None
        self._latest_mode: str = '—'
        self._latest_cmd: Optional[Twist] = None

        self.mask_sub = self.create_subscription(
            Image, 'WSKR/floor_mask', self.mask_callback, IMAGE_QOS
        )
        self.create_subscription(Float32, 'WSKR/heading_to_target', self._on_heading, 10)
        self.create_subscription(String, 'WSKR/tracking_mode', self._on_mode, 10)
        self.create_subscription(Twist, 'WSKR/cmd_vel', self._on_cmd_vel, 10)
        # NOTE: tracked_bbox is intentionally NOT drawn here. Routing it
        # through this node (single-threaded, mask-callback-gated) makes the
        # overlay's bbox lag the live camera feed by several hundred ms.
        # Consumers that want a bbox overlay should subscribe to
        # WSKR/tracked_bbox directly and composite it on top of wskr_overlay.

        # Overlay is diagnostic-only (dashboards), so published as JPEG to keep
        # DDS fanout cheap. Composition is skipped entirely when no subscribers
        # are connected.
        self.overlay_pub = self.create_publisher(
            CompressedImage, 'wskr_overlay/compressed', IMAGE_QOS,
        )
        self.lengths_pub = self.create_publisher(Float32MultiArray, 'WSKR/whisker_lengths', 10)

        self.declare_parameter('overlay_jpeg_quality', 70)
        self._jpeg_quality = int(self.get_parameter('overlay_jpeg_quality').value)

        self.get_logger().info(
            f'WSKR_range ready with {len(self.whisker_labels)} whiskers. '
            'Subscribed to WSKR/floor_mask; publishing wskr_overlay/compressed '
            '(lazy) and WSKR/whisker_lengths.'
        )

    # -------- calibration / params ----------------------------------------

    def _load_calibration(
        self, file_path: str
    ) -> Tuple[Dict[str, Tuple[np.ndarray, CubicSpline, CubicSpline]], List[str]]:
        """Load whisker calibration and normalize pixel coords by the
        calibration reference width. Splines return width-normalized coords
        regardless of the pixel dimensions of the incoming mask at runtime.
        """
        with open(file_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)

        whiskers = data.get('whiskers', {})
        if not whiskers:
            raise RuntimeError(f'No whiskers found in calibration: {file_path}')

        ref_w = float(data.get('image_width', 1920))
        if ref_w <= 0.0:
            raise RuntimeError(f'Invalid calibration image_width in {file_path}')

        labels = sorted(whiskers.keys(), key=lambda x: float(x))
        models: Dict[str, Tuple[np.ndarray, CubicSpline, CubicSpline]] = {}

        for label in labels:
            points = whiskers[label]['points']
            mm = np.array([float(p['distance_mm']) for p in points], dtype=np.float64)
            # Normalize pixel coords by the calibration reference width
            # (both u and v divided by width to keep aspect isotropic).
            px = np.array([float(p['pixel_x']) / ref_w for p in points], dtype=np.float64)
            py = np.array([float(p['pixel_y']) / ref_w for p in points], dtype=np.float64)

            order = np.argsort(mm)
            mm = mm[order]
            px = px[order]
            py = py[order]

            x_spline = CubicSpline(mm, px, bc_type='natural')
            y_spline = CubicSpline(mm, py, bc_type='natural')
            models[label] = (mm, x_spline, y_spline)

        return models, labels

    def _read_lens_params(self) -> LensParams:
        gp = self.get_parameter
        return LensParams(
            x_min=float(gp('x_min').value),
            x_max=float(gp('x_max').value),
            cy=float(gp('cy').value),
            hfov_deg=float(gp('hfov_deg').value),
            tilt_deg=float(gp('tilt_deg').value),
            y_offset=float(gp('y_offset').value),
        )

    def _on_set_parameters(self, params) -> SetParametersResult:
        relevant = [p for p in params if p.name in LENS_PARAM_NAMES]
        if not relevant:
            return SetParametersResult(successful=True)

        proposed = {p.name: p.value for p in relevant}
        with self._params_lock:
            current = self._params
            merged = LensParams(
                x_min=float(proposed.get('x_min', current.x_min)),
                x_max=float(proposed.get('x_max', current.x_max)),
                cy=float(proposed.get('cy', current.cy)),
                hfov_deg=float(proposed.get('hfov_deg', current.hfov_deg)),
                tilt_deg=float(proposed.get('tilt_deg', current.tilt_deg)),
                y_offset=float(proposed.get('y_offset', current.y_offset)),
            )
            if merged.x_max <= merged.x_min:
                return SetParametersResult(successful=False, reason='x_max must be > x_min')
            self._params = merged
        return SetParametersResult(successful=True)

    # -------- side-channel subscribers ------------------------------------

    def _on_heading(self, msg: Float32) -> None:
        self._latest_heading_deg = float(msg.data)

    def _on_mode(self, msg: String) -> None:
        self._latest_mode = msg.data or '—'

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd = msg

    # -------- main pipeline -----------------------------------------------

    def mask_callback(self, msg: Image) -> None:
        """Walk each whisker ray along the floor mask and publish hit distances.

        Called once per floor-mask frame. For each whisker the node samples
        points along the calibrated curve in 1 mm steps until it either
        leaves the image or lands on a non-floor pixel; the stopping distance
        becomes that whisker's length. The overlay is only composed when
        someone is subscribed to ``wskr_overlay/compressed``.
        """
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to convert floor mask image: {exc}')
            return

        if mask.ndim != 2:
            self.get_logger().warn('Expected mono floor mask. Skipping frame.')
            return

        h, w = mask.shape
        px_scale = float(w)

        # Whisker impingement computation must happen every frame — the approach
        # server consumes WSKR/whisker_lengths for proximity checks. We do this
        # unconditionally; it's cheap (pure numpy + mask indexing).
        sample_mm = np.arange(
            0.0, self.max_range_mm + self.sample_step_mm, self.sample_step_mm, dtype=np.float64
        )
        lengths: List[float] = []
        # Cache hit data so we can re-use it when drawing the overlay below.
        per_whisker: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]], float]] = []

        for label in self.whisker_labels:
            _, x_spline, y_spline = self.whisker_models[label]
            x_vals = x_spline(sample_mm) * px_scale
            y_vals = y_spline(sample_mm) * px_scale
            pts_int = np.rint(np.stack([x_vals, y_vals], axis=1)).astype(np.int32)

            hit_distance = self.max_range_mm
            hit_point: Optional[Tuple[int, int]] = None
            valid_polyline: List[Tuple[int, int]] = []
            for i, (x, y) in enumerate(pts_int):
                if x < 0 or x >= w or y < 0 or y >= h:
                    hit_distance = float(sample_mm[i])
                    break
                valid_polyline.append((int(x), int(y)))
                if mask[y, x] == 0:
                    hit_distance = float(sample_mm[i])
                    hit_point = (int(x), int(y))
                    break

            lengths.append(hit_distance)
            per_whisker.append((valid_polyline, hit_point, hit_distance))

        lengths_msg = Float32MultiArray()
        lengths_msg.data = [float(v) for v in lengths]
        self.lengths_pub.publish(lengths_msg)

        # Lazy gate: only compose + encode + publish the overlay when someone
        # is actually subscribed. Overlay is diagnostic-only and ~500 KB/frame
        # of drawing + JPEG encoding is not worth doing for /dev/null.
        if self.overlay_pub.get_subscription_count() == 0:
            return

        with self._params_lock:
            params = self._params

        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        aspect = float(h) / px_scale if px_scale > 0.0 else 9.0 / 16.0

        # 1) Dashed meridians (under whiskers so labels stay on top).
        for deg in MERIDIAN_DEGS:
            pts_norm = project_meridian_normalized(deg, params, aspect=aspect)
            if len(pts_norm) < 2:
                continue
            pts = [(int(u * px_scale), int(v * px_scale)) for (u, v) in pts_norm]
            _draw_dashed_polyline(overlay, pts, color=(0, 0, 255), thickness=2)
            mid = pts[len(pts) // 2]
            cv2.putText(
                overlay, f'{deg}', (mid[0] + 4, mid[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
            )

        # 2) Whiskers + hit markers + labels (re-using hits from the impingement
        #    pass so we don't walk the splines twice).
        for label, (valid_polyline, hit_point, hit_distance) in zip(self.whisker_labels, per_whisker):
            if len(valid_polyline) > 1:
                cv2.polylines(
                    overlay,
                    [np.array(valid_polyline, dtype=np.int32)],
                    isClosed=False,
                    color=(255, 200, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            if hit_point is not None:
                cv2.circle(overlay, hit_point, 6, (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    f'{int(round(hit_distance))} mm',
                    (hit_point[0] + 8, max(hit_point[1] - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            _, x_spline, y_spline = self.whisker_models[label]
            lx_n = float(x_spline(300.0))
            ly_n = float(y_spline(300.0))
            lx = int(np.clip(round(lx_n * px_scale), 0, w - 1))
            ly = int(np.clip(round(ly_n * px_scale), 0, h - 1))
            cv2.putText(
                overlay,
                label,
                (lx + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 3) Compose with readout strip, then JPEG-encode and publish.
        composed = self._compose_with_readout(overlay)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        ok, jpeg_buf = cv2.imencode('.jpg', composed, encode_params)
        if not ok:
            self.get_logger().warn('Overlay JPEG encode failed.')
            return

        overlay_msg = CompressedImage()
        overlay_msg.header = msg.header
        overlay_msg.format = 'jpeg'
        overlay_msg.data = jpeg_buf.tobytes()
        self.overlay_pub.publish(overlay_msg)

    def _compose_with_readout(self, frame: np.ndarray) -> np.ndarray:
        """Stack a text strip under the overlay showing heading, mode, cmd_vel."""
        h, w = frame.shape[:2]
        canvas = np.zeros((h + READOUT_STRIP_HEIGHT, w, 3), dtype=np.uint8)
        canvas[:h, :, :] = frame
        # strip stays black

        heading_text = (
            f'Heading: {self._latest_heading_deg:+.1f}°'
            if self._latest_heading_deg is not None
            else 'Heading: —'
        )
        mode_text = f'Mode: {self._latest_mode}'
        if self._latest_cmd is not None:
            cmd_text = (
                f'cmd  vx={self._latest_cmd.linear.x:+.2f}  '
                f'vy={self._latest_cmd.linear.y:+.2f}  '
                f'\u03c9={self._latest_cmd.angular.z:+.2f}'
            )
        else:
            cmd_text = 'cmd  —'

        y0 = h + 22
        cv2.putText(
            canvas, heading_text, (8, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, mode_text, (180, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 160), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, cmd_text, (8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
        )
        return canvas


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WSKRRangeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
