"""ROS 2 autopilot node consuming the IL MLP trained by train_mlp.py.

Loads a model_*.json (schema_version >= 2, normalization == "saturation")
and drives geometry_msgs/Twist on WSKR/cmd_vel at a fixed rate.

Inputs are cached from async subscriptions; inference runs on a timer so the
publish cadence is decoupled from sensor jitter. If any required input is
stale, the node publishes a zero Twist rather than feeding stale features
into the network.

Drop-in replacement for the inline control loop in
approach_action_server.control_loop. Wire WSKR/autopilot/enable from the
action server's start/stop callbacks if you want goal-gated activation.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


# Default model filename. Override via the `model_filename` ROS parameter
# to swap policies without editing code. Models live in the installed
# share folder at <share/wskr>/models/; everything else (input dim, output
# layout, normalization scales, history depth, activation, action ordering)
# is read from the model JSON itself at startup.
DEFAULT_MODEL_FILENAME = "model_heading_002.json"

WHISKER_COUNT = 11
STATE_DIM = 23  # 11 whiskers + 11 bbox + 1 heading


class _LatestCache:
    """Latest-value cache with monotonic-clock arrival timestamps."""

    def __init__(self, clock) -> None:
        self._clock = clock
        self._values: Dict[str, object] = {}
        self._stamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def put(self, key: str, value: object) -> None:
        now = self._clock.now().nanoseconds * 1e-9
        with self._lock:
            self._values[key] = value
            self._stamps[key] = now

    def get(self, key: str) -> Optional[object]:
        with self._lock:
            return self._values.get(key)

    def age_s(self, key: str) -> Optional[float]:
        with self._lock:
            ts = self._stamps.get(key)
        if ts is None:
            return None
        return self._clock.now().nanoseconds * 1e-9 - ts


class WskrAutopilot(Node):
    def __init__(self) -> None:
        super().__init__("wskr_autopilot")

        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("input_freshness_s", 0.5)
        self.declare_parameter("max_linear_mps", 0.40)
        self.declare_parameter("max_angular_rps", math.radians(40.0))
        self.declare_parameter(
            "publish_zero_when_disabled",
            True,
            ParameterDescriptor(description="If false, simply stops publishing while disabled."),
        )
        self.declare_parameter(
            "model_filename",
            DEFAULT_MODEL_FILENAME,
            ParameterDescriptor(
                description=(
                    "Filename of the model JSON inside <share/wskr>/models/. "
                    "If an absolute path is given, it is used as-is."
                )
            ),
        )

        self._load_model(self._resolve_model_path())

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.input_freshness_s = float(self.get_parameter("input_freshness_s").value)
        self.max_linear_mps = float(self.get_parameter("max_linear_mps").value)
        self.max_angular_rps = float(self.get_parameter("max_angular_rps").value)
        self.publish_zero_when_disabled = bool(self.get_parameter("publish_zero_when_disabled").value)

        self.cache = _LatestCache(self.get_clock())
        self.enabled = True  # default-on; flipped by /WSKR/autopilot/enable if wired

        # State and action histories matching the IL feature layout
        # [s0, a0, s1, a1, ..., a_{N-2}, s_{N-1}].
        self.state_history: Deque[np.ndarray] = deque(maxlen=self.memory_steps)
        self.action_history: Deque[np.ndarray] = deque(maxlen=max(1, self.memory_steps - 1))
        self.last_action_physical = np.zeros(self.action_dim, dtype=np.float64)

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(Float32MultiArray, "WSKR/whisker_lengths", self._on_whiskers, 10)
        self.create_subscription(Float32MultiArray, "WSKR/target_whisker_lengths", self._on_target_whiskers, 10)
        self.create_subscription(Float32, "WSKR/heading_to_target", self._on_heading, 10)
        self.create_subscription(String, "WSKR/tracking_mode", self._on_tracking_mode, 10)
        self.create_subscription(Bool, "WSKR/autopilot/enable", self._on_enable, latched)

        self.cmd_pub = self.create_publisher(Twist, "WSKR/cmd_vel", 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, "WSKR/autopilot/debug", 10)
        self.status_pub = self.create_publisher(String, "WSKR/autopilot/status", 10)

        period = 1.0 / max(1e-3, self.control_rate_hz)
        self.create_timer(period, self._on_tick)

        self.get_logger().info(
            f"wskr_autopilot up: mode={self.mode}, memory_steps={self.memory_steps}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"control_rate={self.control_rate_hz}Hz"
        )

    # ------------------------------------------------------------------ model

    def _resolve_model_path(self) -> Path:
        filename = str(self.get_parameter("model_filename").value)
        candidate = Path(filename)
        if candidate.is_absolute():
            return candidate
        share = Path(get_package_share_directory("wskr"))
        return share / "models" / filename

    def _load_model(self, path: Path) -> None:
        self.get_logger().info(f"Loading autopilot model: {path}")
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)

        if int(blob.get("schema_version", 0)) < 2:
            raise RuntimeError(
                f"Model at {path} is schema_version<2; retrain with current train_mlp.py "
                f"(saturation normalization with x_scale/y_scale baked in)."
            )
        if str(blob.get("normalization", "")) != "saturation":
            raise RuntimeError(
                f"Model at {path} uses normalization={blob.get('normalization')!r}; "
                f"this node only supports 'saturation'."
            )

        self.mode = str(blob["mode"])
        self.memory_steps = int(blob["memory_steps"])
        self.input_dim = int(blob["input_dim"])
        self.output_dim = int(blob["output_dim"])
        self.activation = str(blob["activation"])
        self.output_layout = list(blob["output_layout"])
        self.action_dim = len(self.output_layout)

        if self.action_dim != self.output_dim:
            raise RuntimeError(
                f"output_layout length ({self.action_dim}) != output_dim ({self.output_dim})"
            )

        self.weights = [np.asarray(w, dtype=np.float64) for w in blob["weights"]]
        self.biases = [np.asarray(b, dtype=np.float64) for b in blob["biases"]]
        self.x_scale = np.asarray(blob["x_scale"], dtype=np.float64).reshape(-1)
        self.y_scale = np.asarray(blob["y_scale"], dtype=np.float64).reshape(-1)

        if self.x_scale.size != self.input_dim:
            raise RuntimeError(
                f"x_scale size {self.x_scale.size} != input_dim {self.input_dim}"
            )
        if self.y_scale.size != self.output_dim:
            raise RuntimeError(
                f"y_scale size {self.y_scale.size} != output_dim {self.output_dim}"
            )

        # Past-action slice order in the input vector (between consecutive
        # state slices). For heading_drive: [drive_speed, rotation_rate].
        input_signals = blob.get("input_signals", {})
        self.past_action_order: List[str] = list(input_signals.get("past_action_slice_order", []))
        if self.memory_steps > 1 and len(self.past_action_order) != self.action_dim:
            raise RuntimeError(
                f"past_action_slice_order ({self.past_action_order}) does not match "
                f"action dim {self.action_dim}"
            )

        # Map output_layout index -> position in the action-history slice.
        # We always store actions in past_action_order. When emitting, we read
        # output_layout in its native order and reorder before storing.
        self.output_to_history_idx = (
            [self.past_action_order.index(name) for name in self.output_layout]
            if self.past_action_order
            else list(range(self.action_dim))
        )

    # -------------------------------------------------------------- callbacks

    def _on_whiskers(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != WHISKER_COUNT:
            self.get_logger().warn_once(
                f"WSKR/whisker_lengths len={len(msg.data)}, expected {WHISKER_COUNT}"
            )
            return
        self.cache.put("whiskers_mm", np.asarray(msg.data, dtype=np.float64))

    def _on_target_whiskers(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != WHISKER_COUNT:
            self.get_logger().warn_once(
                f"WSKR/target_whisker_lengths len={len(msg.data)}, expected {WHISKER_COUNT}"
            )
            return
        self.cache.put("target_whiskers_mm", np.asarray(msg.data, dtype=np.float64))

    def _on_heading(self, msg: Float32) -> None:
        self.cache.put("heading_deg", float(msg.data))

    def _on_tracking_mode(self, msg: String) -> None:
        self.cache.put("tracking_mode", str(msg.data))

    def _on_enable(self, msg: Bool) -> None:
        self.enabled = bool(msg.data)
        if not self.enabled:
            # Drop history so a re-enable starts cleanly without stale actions.
            self.state_history.clear()
            self.action_history.clear()
            self.last_action_physical = np.zeros(self.action_dim, dtype=np.float64)

    # --------------------------------------------------------------- control

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())

    def _on_tick(self) -> None:
        if not self.enabled:
            if self.publish_zero_when_disabled:
                self._publish_zero()
            self._publish_status("idle")
            return

        # Required inputs: both whisker channels + heading. Tracking mode is
        # informational; absence does not gate inference (model was trained
        # without it as a feature).
        required = ("whiskers_mm", "target_whiskers_mm", "heading_deg")
        for key in required:
            age = self.cache.age_s(key)
            if age is None or age > self.input_freshness_s:
                self._publish_zero()
                self._publish_status("stale_inputs")
                # Drop history when we go stale so the next live tick rebuilds
                # from fresh observations rather than mixing old + new state.
                self.state_history.clear()
                self.action_history.clear()
                self.last_action_physical = np.zeros(self.action_dim, dtype=np.float64)
                return

        whiskers_mm = self.cache.get("whiskers_mm")
        target_whiskers_mm = self.cache.get("target_whiskers_mm")
        heading_deg = float(self.cache.get("heading_deg"))

        # Convert mm -> m to match training units; the saturation x_scale then
        # divides by WHISKER_MAX_M (0.50 m) to land in [0, 1].
        #
        # Ordering fix: wskr_range_node publishes whiskers sorted by label
        # ascending [-90 … +90] (right-to-left), but the model was trained with
        # whisker[0] = +90° (LEFT) … whisker[10] = -90° (RIGHT), i.e. left-to-right.
        # Reverse both arrays here so the feature vector matches training layout.
        # (model JSON: "angles_deg_left_to_right": [90, 60, 45, 30, 15, 0, -15, -30, -45, -60, -90])
        whiskers_m = np.asarray(whiskers_mm, dtype=np.float64)[::-1] / 1000.0
        target_whiskers_m = np.asarray(target_whiskers_mm, dtype=np.float64)[::-1] / 1000.0

        state = np.concatenate(
            [whiskers_m, target_whiskers_m, np.array([heading_deg], dtype=np.float64)]
        )
        if state.size != STATE_DIM:
            self.get_logger().error(f"state dim {state.size} != {STATE_DIM}")
            self._publish_zero()
            return

        # First fresh tick: pre-fill history with copies of the live state and
        # zero-padded actions (matches HeadlessSimulator._reset_episode_state).
        if not self.state_history:
            for _ in range(self.memory_steps - 1):
                self.state_history.append(state.copy())
                self.action_history.append(np.zeros(self.action_dim, dtype=np.float64))
            self.state_history.append(state.copy())
        else:
            self.state_history.append(state)

        feature_vec = self._build_feature_vector()
        if feature_vec.size != self.input_dim:
            self.get_logger().error(
                f"feature vec size {feature_vec.size} != input_dim {self.input_dim}"
            )
            self._publish_zero()
            return

        x_norm = feature_vec / np.where(np.abs(self.x_scale) < 1e-6, 1.0, self.x_scale)
        y_norm = self._predict(x_norm.reshape(1, -1)).reshape(-1)
        y_phys = y_norm * np.where(np.abs(self.y_scale) < 1e-6, 1.0, self.y_scale)

        twist = self._physical_to_twist(y_phys)
        self.cmd_pub.publish(twist)
        self._publish_status("running")
        self._publish_debug(y_phys)

        # Record the action we just emitted into history (in past_action_order),
        # so the next tick's feature vector reflects what the policy actually did.
        action_in_history_order = np.zeros(self.action_dim, dtype=np.float64)
        for out_idx, hist_idx in enumerate(self.output_to_history_idx):
            action_in_history_order[hist_idx] = y_phys[out_idx]
        self.action_history.append(action_in_history_order)
        self.last_action_physical = y_phys

    def _build_feature_vector(self) -> np.ndarray:
        states = list(self.state_history)
        actions = list(self.action_history)[: self.memory_steps - 1]
        parts: List[np.ndarray] = []
        for i, s in enumerate(states):
            parts.append(s)
            if i < len(actions):
                parts.append(actions[i])
        return np.concatenate(parts).astype(np.float64)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            if self.activation == "tanh":
                a = np.tanh(z)
            elif self.activation == "leaky_relu":
                a = np.where(z > 0.0, z, 0.01 * z)
            else:
                a = np.maximum(0.0, z)
        return a @ self.weights[-1] + self.biases[-1]

    def _physical_to_twist(self, y_phys: np.ndarray) -> Twist:
        """Map model output (named by output_layout) onto a Twist.

        rotation_rate is in deg/s coming out of the model; convert to rad/s
        for angular.z. Linear components are in m/s and pass through.
        """
        named = {name: float(y_phys[i]) for i, name in enumerate(self.output_layout)}
        twist = Twist()
        twist.linear.x = self._clamp(
            named.get("drive_speed", named.get("vx", 0.0)),
            -self.max_linear_mps,
            self.max_linear_mps,
        )
        twist.linear.y = self._clamp(
            named.get("vy", 0.0),
            -self.max_linear_mps,
            self.max_linear_mps,
        )
        rotation_dps = named.get("rotation_rate", 0.0)
        twist.angular.z = self._clamp(
            math.radians(rotation_dps),
            -self.max_angular_rps,
            self.max_angular_rps,
        )
        return twist

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def _publish_debug(self, y_phys: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = [float(v) for v in y_phys]
        self.debug_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WskrAutopilot()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
