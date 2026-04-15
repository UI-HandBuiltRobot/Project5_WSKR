"""WSKR Dead-Reckoning Fuser — "keep a heading estimate alive, even if we can't see the tag."

Two sources of truth can tell us where the target is:
    (1) Vision — the approach server sees the ArUco tag and publishes a
        fresh ``visual_obs`` heading.
    (2) Dead reckoning — the robot is rotating, and ``/odom`` from the
        Arduino bridge tells us how fast. We integrate that yaw rate to
        update the heading while the tag is off-screen.

This node is the single owner of the *fused* heading and the tracking mode.
It switches between the two sources with hysteresis (asymmetric thresholds
prevent chatter near the FOV edge):

    visual --> dead_reckoning
        when |heading| > dr_handoff_deg (default 80°), OR
        when no fresh visual_obs arrives within visual_obs_freshness_s.
    dead_reckoning --> visual
        when a fresh visual_obs arrives AND its heading is inside
        ±visual_reacquire_deg (default 60°).

Sign convention: a CCW body rotation (+yaw_rate) makes the target appear to
move to the right in the image, i.e. the heading-to-target *decreases*. So
dead-reckoning updates with ``heading -= yaw_rate * dt``.

Topics:
    subscribes  /odom                                — robot yaw rate.
    subscribes  WSKR/heading_to_target/visual_obs    — fresh bbox heading.
    publishes   WSKR/heading_to_target               — fused heading (deg).
    publishes   WSKR/tracking_mode                   — ``visual`` or ``dead_reckoning``.
"""
from __future__ import annotations

import math
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String


VISUAL = 'visual'
DEAD_RECKONING = 'dead_reckoning'


def _wrap180(deg: float) -> float:
    """Wrap an angle into the half-open interval [-180, +180)."""
    return (deg + 180.0) % 360.0 - 180.0


class DeadReckoningNode(Node):
    def __init__(self) -> None:
        super().__init__('WSKR_dead_reckoning')

        self.declare_parameter('dr_handoff_deg', 80.0)
        self.declare_parameter('visual_reacquire_deg', 60.0)
        self.declare_parameter('visual_obs_freshness_s', 0.5)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.dr_handoff_deg = float(self.get_parameter('dr_handoff_deg').value)
        self.visual_reacquire_deg = float(self.get_parameter('visual_reacquire_deg').value)
        self.visual_obs_freshness_s = float(self.get_parameter('visual_obs_freshness_s').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        if self.visual_reacquire_deg >= self.dr_handoff_deg:
            self.get_logger().warn(
                'visual_reacquire_deg (%.1f) should be < dr_handoff_deg (%.1f) for hysteresis.'
                % (self.visual_reacquire_deg, self.dr_handoff_deg)
            )

        self.heading_deg = 0.0
        self.mode = VISUAL
        self._latest_visual_obs: Optional[float] = None
        self._latest_visual_obs_t: Optional[float] = None
        self._latest_yaw_rate_rad_s = 0.0
        self._last_tick_t: Optional[float] = None

        self.visual_obs_sub = self.create_subscription(
            Float32, 'WSKR/heading_to_target/visual_obs', self._on_visual_obs, 10
        )
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, 20)

        self.heading_pub = self.create_publisher(Float32, 'WSKR/heading_to_target', 10)
        self.mode_pub = self.create_publisher(String, 'WSKR/tracking_mode', 10)

        self.tick_timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._tick)
        self.get_logger().info(
            'Dead-reckoning fuser ready (handoff=%.1f°, reacquire=%.1f°, freshness=%.2fs).'
            % (self.dr_handoff_deg, self.visual_reacquire_deg, self.visual_obs_freshness_s)
        )

    def _now_s(self) -> float:
        """Current ROS time as a plain float in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_visual_obs(self, msg: Float32) -> None:
        """Cache the newest bbox heading and when it arrived."""
        self._latest_visual_obs = float(msg.data)
        self._latest_visual_obs_t = self._now_s()

    def _on_odom(self, msg: Odometry) -> None:
        """Cache the newest body-frame yaw rate from the Arduino odometry."""
        self._latest_yaw_rate_rad_s = float(msg.twist.twist.angular.z)

    def _visual_obs_is_fresh(self, now: float) -> bool:
        """True if the last visual_obs is recent enough to trust."""
        if self._latest_visual_obs is None or self._latest_visual_obs_t is None:
            return False
        return (now - self._latest_visual_obs_t) <= self.visual_obs_freshness_s

    def _tick(self) -> None:
        """One fusion step: update heading, swap modes if thresholds crossed, publish."""
        now = self._now_s()
        dt = 0.0 if self._last_tick_t is None else max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        fresh = self._visual_obs_is_fresh(now)

        if self.mode == VISUAL:
            if fresh:
                # Snap to the fresh observation. Wrap defensively: the bbox
                # heading should already be bounded by the lens FOV, but we
                # never want a stale value to leak past ±180°.
                self.heading_deg = _wrap180(float(self._latest_visual_obs))
            # Exit to DR on loss of fresh obs OR on heading leaving the visual band.
            if (not fresh) or abs(self.heading_deg) > self.dr_handoff_deg:
                self.mode = DEAD_RECKONING
                self.get_logger().info(
                    'visual → dead_reckoning (heading=%.1f°, fresh=%s)'
                    % (self.heading_deg, fresh)
                )
        else:  # DEAD_RECKONING
            yaw_rate_deg = math.degrees(self._latest_yaw_rate_rad_s)
            # Integrate yaw rate and wrap into [-180, +180). Without the wrap
            # the heading accumulates indefinitely as the robot spins, which
            # eventually produces garbage like +1042° and breaks every
            # downstream consumer that assumes a bounded angle.
            self.heading_deg = _wrap180(self.heading_deg - yaw_rate_deg * dt)
            # Reacquisition is gated by the FRESH visual observation, not by
            # the dead-reckoned heading. DR can drift out of the cone while
            # the actual tag is well inside it — checking the obs directly
            # lets us snap back the moment the tag reappears.
            if (
                fresh
                and self._latest_visual_obs is not None
                and abs(self._latest_visual_obs) <= self.visual_reacquire_deg
            ):
                self.mode = VISUAL
                self.heading_deg = _wrap180(float(self._latest_visual_obs))
                self.get_logger().info(
                    'dead_reckoning → visual (snapped to %.1f°)' % self.heading_deg
                )

        heading_msg = Float32()
        heading_msg.data = float(self.heading_deg)
        self.heading_pub.publish(heading_msg)

        mode_msg = String()
        mode_msg.data = self.mode
        self.mode_pub.publish(mode_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeadReckoningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
