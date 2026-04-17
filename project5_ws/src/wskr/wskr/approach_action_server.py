"""WSKR Approach Action Server — "drive up to this specific thing."

This is the supervisor of the approach behavior. A client (typically a GUI
like ``Start_Aruco_Approach``) sends an ``ApproachObject`` action goal
naming an ArUco tag ID (or a toy bbox seeded by an external detector).
This node then:

    1. Runs vision every frame — ArUco detection for TARGET_BOX goals or a
       CSRT template tracker for TARGET_TOY. Publishes a *visual observation*
       of the target's heading whenever detection succeeds.
    2. Enables the autopilot (``wskr_autopilot``) for the duration of the
       goal. The autopilot owns ``WSKR/cmd_vel`` and consumes the fused
       heading + whiskers itself.
    3. Watches whisker drive-distances + the fused heading to decide when
       the target has been reached, aborts on timeout / loss / reacquisition
       failure, and disables the autopilot when the goal ends.

Topics:
    subscribes  camera1/image_raw/compressed     — main camera (JPEG).
    subscribes  WSKR/whisker_lengths             — 11 floats (drive distances).
    subscribes  WSKR/heading_to_target           — fused heading from DR node.
    subscribes  WSKR/tracking_mode               — ``visual`` or ``dead_reckoning``.
    publishes   WSKR/heading_to_target/visual_obs — heading from bbox (when seen).
    publishes   WSKR/tracked_bbox                — width-normalized (x,y,w,h).
    publishes   WSKR/autopilot/enable            — latched Bool gating the autopilot.
    publishes   WSKR/cmd_vel                     — zero Twist on goal end (safety stop).
Action:
    server of   WSKR/approach_object             — ApproachObject.action.
"""
import math
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from robot_interfaces.action import ApproachObject

from .lens_model import LensParams, compute_heading_rad


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

LENS_PARAM_NAMES = ('x_min', 'x_max', 'cy', 'hfov_deg', 'tilt_deg', 'y_offset')


class WSKRApproachActionServer(Node):
    def __init__(self) -> None:
        super().__init__('wskr_approach_action')

        self.bridge = CvBridge()

        self.declare_parameter('aruco_id', 1)
        self.declare_parameter('approach_timeout_sec', 20.0)
        self.declare_parameter('proximity_success_mm', 100.0)
        self.declare_parameter('target_lost_timeout_sec', 7.0)
        self.declare_parameter('reacquire_threshold', 0.55)
        self.declare_parameter('reacquire_failure_deg', 30.0)
        # Number of consecutive frames without a valid detection before the
        # reacquire-failure abort fires. Needs to be large enough to ride out
        # transient motion blur / occlusion; too small and moving the marker
        # aborts the goal.
        self.declare_parameter('reacquire_failure_frames', 10)
        # Extra downsample applied inside _try_track_box before detectMarkers.
        # Now that image_callback decodes JPEGs at 1/2 resolution (via
        # IMREAD_REDUCED_COLOR_2), the frame is already small enough that
        # an additional downsample hurts detection without saving much time.
        # Set <1.0 only if frames are larger than expected.
        self.declare_parameter('aruco_detect_scale', 1.0)
        # Log a warning whenever image_callback takes longer than this.
        self.declare_parameter('slow_frame_warn_ms', 150.0)

        # Lens params (normalized, for in-process heading computation).
        _lp = LensParams()
        self.declare_parameter('x_min', _lp.x_min)
        self.declare_parameter('x_max', _lp.x_max)
        self.declare_parameter('cy', _lp.cy)
        self.declare_parameter('hfov_deg', _lp.hfov_deg)
        self.declare_parameter('tilt_deg', _lp.tilt_deg)
        self.declare_parameter('y_offset', _lp.y_offset)

        self.aruco_id = int(self.get_parameter('aruco_id').value)
        self.approach_timeout_sec = float(self.get_parameter('approach_timeout_sec').value)
        self.proximity_success_mm = float(self.get_parameter('proximity_success_mm').value)
        self.target_lost_timeout_sec = float(self.get_parameter('target_lost_timeout_sec').value)
        self.reacquire_threshold = float(self.get_parameter('reacquire_threshold').value)
        self.reacquire_failure_deg = float(self.get_parameter('reacquire_failure_deg').value)
        self.reacquire_failure_frames = int(self.get_parameter('reacquire_failure_frames').value)
        self.aruco_detect_scale = float(self.get_parameter('aruco_detect_scale').value)
        self.slow_frame_warn_ms = float(self.get_parameter('slow_frame_warn_ms').value)

        self._lens_lock = threading.Lock()
        self._lens_params = self._read_lens_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        cb_group = ReentrantCallbackGroup()

        self.image_sub = self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self.image_callback, IMAGE_QOS,
            callback_group=cb_group,
        )
        self.whisker_sub = self.create_subscription(
            Float32MultiArray, 'WSKR/whisker_lengths', self.whisker_callback, 10, callback_group=cb_group
        )
        self.target_whisker_sub = self.create_subscription(
            Float32MultiArray, 'WSKR/target_whisker_lengths', self.target_whisker_callback, 10, callback_group=cb_group
        )

        # Visual observations only — the dead_reckoning_node owns the fused topic.
        self.visual_obs_pub = self.create_publisher(
            Float32, 'WSKR/heading_to_target/visual_obs', 10
        )
        # Latched enable to gate the autopilot for the duration of a goal.
        autopilot_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.autopilot_enable_pub = self.create_publisher(
            Bool, 'WSKR/autopilot/enable', autopilot_qos
        )
        # Safety-stop publisher: zero Twist when a goal ends, in case the
        # autopilot is slow to react to the disable flag.
        self.cmd_pub = self.create_publisher(Twist, 'WSKR/cmd_vel', 10)
        self.tracked_bbox_pub = self.create_publisher(Float32MultiArray, 'WSKR/tracked_bbox', 10)
        self._publish_autopilot_enable(False)

        self.fused_heading_sub = self.create_subscription(
            Float32, 'WSKR/heading_to_target', self._on_fused_heading, 10, callback_group=cb_group
        )
        self.tracking_mode_sub = self.create_subscription(
            String, 'WSKR/tracking_mode', self._on_tracking_mode, 10, callback_group=cb_group
        )

        self.action_server = ActionServer(
            self,
            ApproachObject,
            'WSKR/approach_object',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=cb_group,
        )

        self.lock = threading.Lock()
        self.active_goal_handle = None
        self.goal_target_type = None
        self.goal_object_id = -1
        self.goal_selected_obj = None

        self.latest_whiskers: Optional[np.ndarray] = None
        self.latest_target_whiskers: Optional[np.ndarray] = None
        self.last_heading_deg = 0.0  # mirror of WSKR/heading_to_target (fused)
        self.tracking_mode = 'visual'  # mirror of WSKR/tracking_mode
        self.visually_tracked = True  # bbox present on most recent frame
        self.last_tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frames_since_valid_track = 0
        self.target_lost_threshold_frames = 3

        self.pending_toy_bbox: Optional[Tuple[float, float, float, float]] = None
        self.tracker = None
        self.last_frame = None
        self.lost_since: Optional[float] = None
        self.lost_template: Optional[np.ndarray] = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        self.get_logger().info('Approach action server ready on WSKR/approach_object.')

    def goal_callback(self, goal_request: ApproachObject.Goal) -> GoalResponse:
        """Reject new goals while one is active; reject unknown target types."""
        with self.lock:
            if self.active_goal_handle is not None:
                return GoalResponse.REJECT
        if goal_request.target_type not in (ApproachObject.Goal.TARGET_TOY, ApproachObject.Goal.TARGET_BOX):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _create_csrt(self):
        # Stock defaults with segmentation disabled (segfault trigger on OpenCV 4.10)
        # and a slightly stricter PSR so track loss is reported sooner, handing off
        # to template-match re-acquisition.
        params = cv2.TrackerCSRT_Params()
        params.use_segmentation = False
        params.psr_threshold = 0.06
        return cv2.TrackerCSRT_create(params)

    def _pad_bbox(self, bbox, frame_shape, pad_frac: float = 0.15):
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        pad_w = w * pad_frac
        pad_h = h * pad_frac
        nx = max(0.0, x - pad_w)
        ny = max(0.0, y - pad_h)
        nw = min(fw - nx, w + 2.0 * pad_w)
        nh = min(fh - ny, h + 2.0 * pad_h)
        return (nx, ny, nw, nh)

    def _extract_bbox_from_selected_obj(self, selected_obj, object_id: int, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
        n = len(selected_obj.x)
        if n == 0:
            return None

        idx = 0
        if n > 1 and selected_obj.detection_ids:
            for i, det_id in enumerate(selected_obj.detection_ids):
                try:
                    if int(det_id) == object_id:
                        idx = i
                        break
                except ValueError:
                    pass

        cx = float(selected_obj.x[idx])
        cy = float(selected_obj.y[idx])
        bw = float(selected_obj.width[idx])
        bh = float(selected_obj.height[idx])

        if cx <= 1.0 and cy <= 1.0 and bw <= 1.0 and bh <= 1.0:
            cx *= w
            cy *= h
            bw *= w
            bh *= h

        x0 = max(0.0, cx - bw / 2.0)
        y0 = max(0.0, cy - bh / 2.0)
        bw = max(2.0, min(bw, w - x0))
        bh = max(2.0, min(bh, h - y0))
        return (x0, y0, bw, bh)

    def whisker_callback(self, msg: Float32MultiArray) -> None:
        """Cache the latest 11-whisker floor-distance array for the control loop."""
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != 11:
            return
        self.latest_whiskers = data

    def target_whisker_callback(self, msg: Float32MultiArray) -> None:
        """Cache the latest 11-whisker target-bbox distance array for proximity check."""
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != 11:
            return
        self.latest_target_whiskers = data

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
        # Live-reconfigure lens params. Any other parameter passes through.
        lens_updates = [p for p in params if p.name in LENS_PARAM_NAMES]
        if lens_updates:
            proposed = {p.name: p.value for p in lens_updates}
            with self._lens_lock:
                current = self._lens_params
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
                self._lens_params = merged

        return SetParametersResult(successful=True)

    def _publish_autopilot_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.autopilot_enable_pub.publish(msg)

    def _compute_and_publish_heading(self, u_norm: float, v_norm: float) -> None:
        """Synchronous heading computation via the in-process lens model.
        Inputs are in width-normalized image coords (u_px / W, v_px / W).
        """
        with self._lens_lock:
            params = self._lens_params
        heading_deg = math.degrees(compute_heading_rad(float(u_norm), float(v_norm), params))
        self._publish_visual_obs(heading_deg)

    def _check_bbox_impinges_whiskers(self, heading_deg: float, whiskers: Optional[np.ndarray], threshold_mm: float = 250.0) -> bool:
        """
        Check if object at heading_deg impinges on whiskers within threshold_mm.
        Maps heading angle to closest whisker and checks its length.
        """
        if whiskers is None or len(whiskers) == 0:
            return False
        
        # Whiskers arranged in fan from -90 to +90 degrees (11 whiskers typical)
        num_whiskers = len(whiskers)
        whisker_angles = np.linspace(-90.0, 90.0, num_whiskers)
        
        # Find closest whisker to object heading
        idx = np.argmin(np.abs(whisker_angles - heading_deg))
        closest_whisker_mm = float(whiskers[idx])
        
        return closest_whisker_mm <= threshold_mm

    def _publish_visual_obs(self, heading_deg: float) -> None:
        msg = Float32()
        msg.data = float(heading_deg)
        self.visual_obs_pub.publish(msg)

    def _on_fused_heading(self, msg: Float32) -> None:
        self.last_heading_deg = float(msg.data)

    def _on_tracking_mode(self, msg: String) -> None:
        self.tracking_mode = msg.data

    def _cache_template(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> None:
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        crop = frame[y:y + h, x:x + w]
        if crop.size > 0:
            self.lost_template = crop.copy()

    def _template_reacquire(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        if self.lost_template is None:
            return None
        th, tw = self.lost_template.shape[:2]
        fh, fw = frame.shape[:2]
        if tw >= fw or th >= fh:
            return None
        result = cv2.matchTemplate(frame, self.lost_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.reacquire_threshold:
            return None
        return (float(max_loc[0]), float(max_loc[1]), float(tw), float(th))

    def _try_track_toy(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Tracker is seeded in execute_callback; here we only update / re-acquire.
        if self.tracker is None:
            return None

        ok, bbox = self.tracker.update(frame)
        if ok:
            self.lost_since = None
            self._cache_template(frame, bbox)
            x, y, w, h = bbox
            return (int(x), int(y), int(w), int(h))

        # Track lost: try to re-acquire via normalized cross-correlation on the cached crop.
        if self.lost_since is None:
            self.lost_since = time.time()

        new_bbox = self._template_reacquire(frame)
        if new_bbox is not None:
            seed = tuple(int(v) for v in new_bbox)
            self.tracker = self._create_csrt()
            self.tracker.init(frame, seed)
            self.lost_since = None
            self._cache_template(frame, seed)
            x, y, w, h = new_bbox
            return (int(x), int(y), int(w), int(h))

        return None

    def _try_track_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        scale = self.aruco_detect_scale
        if 0.0 < scale < 1.0:
            det_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            inv_scale = 1.0 / scale
        else:
            det_frame = frame
            inv_scale = 1.0

        corners, ids, _ = self.aruco_detector.detectMarkers(det_frame)
        target_id = self.goal_object_id if self.goal_object_id >= 0 else self.aruco_id

        if ids is None:
            # Tag not in this frame. Start the lost_since clock if it isn't
            # running yet; target_lost_timeout_sec will abort the goal if
            # this persists long enough. Dropped frames in between don't
            # reset it — only a successful detection does (below).
            if self.lost_since is None:
                self.lost_since = time.time()
            return None

        ids = ids.flatten().tolist()
        for i, marker_id in enumerate(ids):
            if int(marker_id) != target_id:
                continue
            # Successful reacquisition of the target tag.
            self.lost_since = None
            pts = corners[i][0] * inv_scale
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            return (x_min, y_min, x_max - x_min, y_max - y_min)

        # Other markers visible but not the target — count as lost.
        if self.lost_since is None:
            self.lost_since = time.time()
        return None

    def image_callback(self, msg: CompressedImage) -> None:
        """Per-frame vision: detect target, publish bbox and visual_obs heading."""
        if self.active_goal_handle is None:
            return

        t_start = time.perf_counter()
        # IMREAD_REDUCED_COLOR_2: libjpeg-turbo decodes directly at half
        # resolution by skipping high-frequency DCT coefficients. Roughly
        # 2-4x faster than a full decode. All downstream logic uses
        # width-normalized coords so the smaller frame is fine.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            return
        t_decode = time.perf_counter()

        self.last_frame = frame

        # Always attempt detection — the dead_reckoning_node owns mode hysteresis.
        tracked_bbox = None
        if self.goal_target_type == ApproachObject.Goal.TARGET_TOY:
            tracked_bbox = self._try_track_toy(frame)
        elif self.goal_target_type == ApproachObject.Goal.TARGET_BOX:
            tracked_bbox = self._try_track_box(frame)
        t_detect = time.perf_counter()

        if tracked_bbox is None:
            self.frames_since_valid_track += 1
            self.visually_tracked = False
            self._maybe_warn_slow(t_start, t_decode, t_detect, None)
            return

        # Valid tracking frame. tracked_bbox is in frame-pixel coords; normalize
        # by the frame WIDTH so downstream consumers stay resolution-agnostic.
        self.frames_since_valid_track = 0
        self.last_tracked_bbox = tracked_bbox

        frame_w = float(frame.shape[1]) if frame.shape[1] > 0 else 1.0
        x, y, bw, bh = tracked_bbox
        x_n = x / frame_w
        y_n = y / frame_w
        w_n = bw / frame_w
        h_n = bh / frame_w

        bbox_msg = Float32MultiArray()
        bbox_msg.data = [float(x_n), float(y_n), float(w_n), float(h_n)]
        self.tracked_bbox_pub.publish(bbox_msg)

        self.visually_tracked = True
        center_u_n = x_n + w_n / 2.0
        center_v_n = y_n + h_n / 2.0
        self._compute_and_publish_heading(center_u_n, center_v_n)

        self._maybe_warn_slow(t_start, t_decode, t_detect, time.perf_counter())

    def _maybe_warn_slow(self, t_start, t_decode, t_detect, t_end) -> None:
        t_last = t_end if t_end is not None else t_detect
        total_ms = 1000.0 * (t_last - t_start)
        if total_ms < self.slow_frame_warn_ms:
            return
        decode_ms = 1000.0 * (t_decode - t_start)
        detect_ms = 1000.0 * (t_detect - t_decode)
        tail_ms = 1000.0 * (t_last - t_detect) if t_end is not None else 0.0
        self.get_logger().warn(
            f'image_callback slow: total={total_ms:.0f}ms '
            f'(decode={decode_ms:.0f}ms detect={detect_ms:.0f}ms publish={tail_ms:.0f}ms)'
        )

    def execute_callback(self, goal_handle):
        """Run the approach loop for one action goal until success, abort, or cancel."""
        with self.lock:
            self.active_goal_handle = goal_handle
            self.goal_target_type = goal_handle.request.target_type
            self.goal_object_id = int(goal_handle.request.object_id)
            self.goal_selected_obj = goal_handle.request.selected_obj
            self.visually_tracked = True
            self.tracker = None
            self.pending_toy_bbox = None
            self.frames_since_valid_track = 0
            self.last_tracked_bbox = None
            self.lost_since = None
            self.lost_template = None

        # Hand control of WSKR/cmd_vel to the autopilot for the duration
        # of this goal. Autopilot drops its own state on enable so it
        # starts cleanly each episode.
        self._publish_autopilot_enable(True)

        if self.goal_target_type == ApproachObject.Goal.TARGET_TOY:
            w = int(self.goal_selected_obj.image_width) if self.goal_selected_obj.image_width else 0
            h = int(self.goal_selected_obj.image_height) if self.goal_selected_obj.image_height else 0
            if w <= 0 or h <= 0:
                if self.last_frame is not None:
                    h, w = self.last_frame.shape[:2]
            if w > 0 and h > 0:
                self.pending_toy_bbox = self._extract_bbox_from_selected_obj(
                    self.goal_selected_obj, self.goal_object_id, w, h
                )
                self.get_logger().info(
                    f'Seeded CSRT pending bbox={self.pending_toy_bbox} '
                    f'from selected_obj image={w}x{h}'
                )
            else:
                self.get_logger().warn('No image dimensions available to seed CSRT bbox')

            # Seed CSRT immediately on the freshest available frame so init and
            # first update run on the same frame content.
            wait_start = time.time()
            while self.last_frame is None and (time.time() - wait_start) < 1.0:
                time.sleep(0.02)
            if self.last_frame is not None and self.pending_toy_bbox is not None:
                padded = self._pad_bbox(self.pending_toy_bbox, self.last_frame.shape, pad_frac=0.15)
                seed = tuple(int(v) for v in padded)
                with self.lock:
                    self.tracker = self._create_csrt()
                    self.tracker.init(self.last_frame, seed)
                    self._cache_template(self.last_frame, seed)
                self.get_logger().info(f'CSRT initialized on frozen frame with padded bbox={seed}')
            else:
                self.get_logger().warn('Could not seed CSRT: no frame or no pending bbox')

        start = time.time()
        result = ApproachObject.Result()
        result.movement_success = True
        result.proximity_success = False
        result.movement_message = 'Approach timed out'

        while rclpy.ok() and self.active_goal_handle is goal_handle:
            if goal_handle.is_cancel_requested:
                result.movement_success = False
                result.movement_message = 'Goal canceled'
                goal_handle.canceled()
                break

            feedback = ApproachObject.Feedback()
            feedback.tracking_mode = self.tracking_mode
            feedback.heading_to_target_deg = float(self.last_heading_deg)
            feedback.visually_tracked = bool(self.visually_tracked)
            feedback.whisker_lengths = self.latest_whiskers.tolist() if self.latest_whiskers is not None else []
            goal_handle.publish_feedback(feedback)

            # Reacquisition failure: DR says we should be staring at the target (inside
            # the ±reacquire_failure_deg cone) but vision has no bbox on the current frame.
            # Reacquisition-failure abort: only meaningful for TARGET_TOY.
            # CSRT can "successfully" lock onto the wrong object after drift,
            # so when DR says we should be looking at the target and vision
            # has been silent for N frames, we give up. For TARGET_BOX
            # (ArUco) detection is stateless — either the marker ID is
            # visible or not — so we rely on target_lost_timeout_sec and the
            # automatic reacquisition in dead_reckoning_node when a fresh
            # visual_obs arrives inside the reacquire cone. No early abort.
            if (
                self.goal_target_type == ApproachObject.Goal.TARGET_TOY
                and self.tracking_mode == 'dead_reckoning'
                and self.frames_since_valid_track >= self.reacquire_failure_frames
                and abs(self.last_heading_deg) <= self.reacquire_failure_deg
            ):
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = (
                    f'Reacquisition failed inside ±{self.reacquire_failure_deg:.0f}° cone '
                    f'({self.frames_since_valid_track} frames without detection)'
                )
                self.get_logger().warn(
                    f'Aborting approach: {result.movement_message}'
                )
                goal_handle.abort()
                break

            # Target lost: abort only after reacquisition has failed for target_lost_timeout_sec.
            if self.lost_since is not None and (time.time() - self.lost_since) > self.target_lost_timeout_sec:
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = f'Target lost: reacquisition failed for >{self.target_lost_timeout_sec:.1f}s'
                goal_handle.abort()
                break

            # Success when the target bbox reaches within proximity_success_mm.
            # Uses target_whisker_lengths (rays that stop at the first bbox pixel)
            # rather than floor whiskers, so only the tracked object triggers
            # success — not incidental floor obstacles at the same distance.
            # Requires a fresh valid track so a stale bbox can't falsely succeed.
            if (
                self.last_tracked_bbox is not None
                and self.frames_since_valid_track == 0
                and self.latest_target_whiskers is not None
                and float(np.min(self.latest_target_whiskers)) < self.proximity_success_mm
            ):
                closest_mm = float(np.min(self.latest_target_whiskers))
                result.proximity_success = True
                result.movement_message = (
                    f'Target bbox within {closest_mm:.0f} mm '
                    f'(threshold {self.proximity_success_mm:.0f} mm)'
                )
                goal_handle.succeed()
                break

            time.sleep(0.05)

        with self.lock:
            if self.active_goal_handle is goal_handle:
                self.active_goal_handle = None
                self.goal_target_type = None
                self.goal_object_id = -1
                self.goal_selected_obj = None
                self.tracker = None
                self.pending_toy_bbox = None
                self.lost_since = None
                self.lost_template = None

        # Hand control back: autopilot stops publishing cmd_vel; safety stop
        # in case it's still mid-tick or the next message is delayed.
        self._publish_autopilot_enable(False)
        self.cmd_pub.publish(Twist())
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WSKRApproachActionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
