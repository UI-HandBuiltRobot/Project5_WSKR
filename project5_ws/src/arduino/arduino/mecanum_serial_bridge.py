"""Mecanum Serial Bridge — the "translator" between ROS and the Arduino.

ROS talks in ``geometry_msgs/Twist`` (metres per second, radians per second)
and ``nav_msgs/Odometry``. The Arduino talks in short CSV lines over USB
serial at 115200 baud (centimetres per second, degrees per second). This
node converts between the two at ~20 Hz.

What it sends to the Arduino (one line per tick):
    V,<vx_cm_s>,<vy_cm_s>,<omega_deg_s>\\n
    X\\n                    (on ``WSKR/stop`` or shutdown — hard stop)

What it reads from the Arduino (streamed continuously):
    O,<dyaw_deg>,<dx_cm>,<dy_cm>,<dt_ms>\\n    (body-frame odometry deltas)
    any other text                              (forwarded as status strings)

The node accumulates the ``O`` deltas into global pose (x, y, yaw) and
publishes that as ``nav_msgs/Odometry`` on ``/odom``. It also broadcasts the
``odom -> base_link`` TF, which the dead_reckoning fuser needs to integrate
the yaw while vision is lost.

Topics:
    subscribes  WSKR/cmd_vel    — Twist from the autopilot.
    subscribes  WSKR/stop       — Empty, triggers an immediate ``X`` command.
    publishes   /odom           — nav_msgs/Odometry from wheel encoders.
    publishes   arduino/status  — String, for anything non-odometry the Arduino says.

Parameters: ``port``, ``baud``, ``odom_frame``, ``base_frame``, ``publish_tf``,
``cmd_rate_hz``, ``cmd_timeout_s`` (if no cmd_vel for this long, stream zeros),
``stop_on_timeout``.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String
from tf2_ros import TransformBroadcaster

try:
    import serial  # pyserial
except ImportError as exc:  # pragma: no cover
    raise ImportError('pyserial is required (apt: python3-serial, pip: pyserial).') from exc


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class MecanumSerialBridge(Node):
    def __init__(self) -> None:
        super().__init__('mecanum_serial_bridge')

        self.declare_parameter('port', '/dev/ttyACM1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('cmd_rate_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('stop_on_timeout', True)
        # Scalar multiplier applied to the whole cmd_vel (linear + angular)
        # before it is sent to the Arduino. Writable at runtime via
        # `ros2 param set` or the dashboard's "Inject" button so the robot
        # can be slowed down during careful testing without a restart.
        self.declare_parameter('drive_speed_multiplier', 1.0)

        self.port = self.get_parameter('port').value
        self.baud = int(self.get_parameter('baud').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.cmd_rate_hz = float(self.get_parameter('cmd_rate_hz').value)
        self.cmd_timeout_s = float(self.get_parameter('cmd_timeout_s').value)
        self.stop_on_timeout = bool(self.get_parameter('stop_on_timeout').value)
        self.drive_speed_multiplier = self._clamp_multiplier(
            float(self.get_parameter('drive_speed_multiplier').value)
        )
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._serial_lock = threading.Lock()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self.get_logger().info(f'Opened {self.port} @ {self.baud} baud.')

        self._cmd_lock = threading.Lock()
        self._latest_twist: Optional[Twist] = None
        self._latest_twist_stamp = self.get_clock().now()

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.twist_sub = self.create_subscription(Twist, 'WSKR/cmd_vel', self._on_twist, 10)
        self.stop_sub = self.create_subscription(Empty, 'WSKR/stop', self._on_stop, 1)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.status_pub = self.create_publisher(String, 'arduino/status', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.cmd_timer = self.create_timer(1.0 / max(self.cmd_rate_hz, 1.0), self._send_cmd_tick)

        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader_thread.start()

        self.get_logger().info('mecanum_serial_bridge ready.')

    def destroy_node(self) -> bool:
        """On shutdown, stop the reader thread and tell the Arduino to halt."""
        self._reader_stop.set()
        try:
            self._send_line('X')
        except Exception:  # noqa: BLE001
            pass
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()

    @staticmethod
    def _clamp_multiplier(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _on_set_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name == 'drive_speed_multiplier':
                try:
                    v = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason='drive_speed_multiplier must be numeric',
                    )
                if not 0.0 <= v <= 1.0:
                    return SetParametersResult(
                        successful=False,
                        reason='drive_speed_multiplier must be in [0.0, 1.0]',
                    )
                self.drive_speed_multiplier = v
                self.get_logger().info(f'drive_speed_multiplier = {v:.3f}')
        return SetParametersResult(successful=True)

    def _on_twist(self, msg: Twist) -> None:
        """Cache the newest cmd_vel and its arrival time."""
        with self._cmd_lock:
            self._latest_twist = msg
            self._latest_twist_stamp = self.get_clock().now()

    def _on_stop(self, _msg: Empty) -> None:
        """Hard-stop request: clear the cached twist and send ``X`` to the base."""
        # Clear the latest twist so the next cmd tick does not restart motion.
        with self._cmd_lock:
            self._latest_twist = None
        self._send_line('X')
        self.get_logger().info('WSKR/stop received — sent X to Arduino.')

    def _send_cmd_tick(self) -> None:
        """Fixed-rate tick: send the latest Twist as a V command (or zeros if stale)."""
        with self._cmd_lock:
            twist = self._latest_twist
            stamp = self._latest_twist_stamp
        now = self.get_clock().now()
        age_s = (now - stamp).nanoseconds * 1e-9
        stale = twist is None or age_s > self.cmd_timeout_s
        if stale:
            if self.stop_on_timeout:
                self._send_line('V,0,0,0')
            return
        k = self.drive_speed_multiplier
        vx_cm = twist.linear.x * 100.0 * k
        vy_cm = twist.linear.y * 100.0 * k
        omega_deg = twist.angular.z * 180.0 / math.pi * k
        self._send_line(f'V,{vx_cm:.4f},{vy_cm:.4f},{omega_deg:.4f}')

    def _send_line(self, line: str) -> None:
        """Write one newline-terminated ASCII line to the serial port."""
        payload = (line + '\n').encode('ascii', errors='ignore')
        with self._serial_lock:
            self.ser.write(payload)

    def _serial_reader(self) -> None:
        """Background thread: read the serial port, split on newlines, dispatch lines."""
        buf = b''
        while not self._reader_stop.is_set():
            try:
                chunk = self.ser.read(128)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'serial read error: {exc}')
                continue
            if not chunk:
                continue
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                self._handle_line(line.decode('ascii', errors='ignore').strip())

    def _handle_line(self, line: str) -> None:
        """Route one decoded line: ``O,...`` is odometry, everything else is status."""
        if not line:
            return
        if line.startswith('O,'):
            self._handle_odom_line(line)
            return
        msg = String()
        msg.data = line
        self.status_pub.publish(msg)
        if line in ('Ready', 'STOP', 'Done') or line.startswith('ERROR'):
            self.get_logger().info(f'arduino: {line}')

    def _handle_odom_line(self, line: str) -> None:
        """Parse ``O,dyaw,dx,dy,dt``, accumulate global pose, publish ``/odom`` (+ TF)."""
        parts = line.split(',')
        if len(parts) != 5:
            return
        try:
            dyaw_deg = float(parts[1])
            dx_cm = float(parts[2])
            dy_cm = float(parts[3])
            dt_ms = float(parts[4])
        except ValueError:
            return

        dyaw = math.radians(dyaw_deg)
        dx = dx_cm * 0.01  # cm -> m
        dy = dy_cm * 0.01
        dt = max(dt_ms * 1e-3, 1e-6)

        yaw_mid = self.yaw + 0.5 * dyaw
        c = math.cos(yaw_mid)
        s = math.sin(yaw_mid)
        self.x += dx * c - dy * s
        self.y += dx * s + dy * c
        self.yaw = (self.yaw + dyaw + math.pi) % (2.0 * math.pi) - math.pi

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = yaw_to_quat(self.yaw)
        odom.twist.twist.linear.x = dx / dt
        odom.twist.twist.linear.y = dy / dt
        odom.twist.twist.angular.z = dyaw / dt
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.rotation = yaw_to_quat(self.yaw)
            self.tf_broadcaster.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MecanumSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
