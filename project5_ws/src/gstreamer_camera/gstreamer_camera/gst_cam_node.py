#!/usr/bin/env python3
"""GStreamer Camera Node — "where all the camera frames come from."

Pulls MJPEG frames from a ``v4l2`` camera (e.g. a USB webcam on
``/dev/video0``) using a GStreamer pipeline and republishes them as
``sensor_msgs/CompressedImage`` on ``camera1/image_raw/compressed``.

We publish COMPRESSED (JPEG) instead of raw BGR on purpose: a 1080p raw
frame is ~6 MB and DDS has to serialize one copy per subscriber, which was
the single largest cost on the Jetson. Each subscriber decodes the ~500 KB
JPEG locally with OpenCV (libjpeg-turbo) — much cheaper.

Parameters:
    pipeline         — GStreamer pipeline string. Must end with ``appsink name=sink``.
    frame_id         — frame_id written into the header.
    topic            — topic name (default ``camera1/image_raw/compressed``).
    publish_rate_hz  — throttle; 0 means "every frame the pipeline produces".

Override ``pipeline`` to swap cameras or source resolutions. The default
assumes a single 1080p @ 30 fps MJPEG USB camera on ``/dev/video0``.
"""
from __future__ import annotations

import sys
import threading
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst  # noqa: E402  (needs gi.require_version first)

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Minimal pipeline — just grab MJPEG frames and pass them through.
# max-buffers=1 drop=true on the appsink means the worker always pulls
# the freshest frame; older samples are silently dropped if the worker
# is behind schedule.
DEFAULT_PIPELINE = (
    'v4l2src device=/dev/video0 ! '
    'image/jpeg, width=1920, height=1080, framerate=30/1 ! '
    'appsink name=sink max-buffers=1 drop=true sync=false'
)


class GstCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('gstreamer_camera')

        self.declare_parameter('pipeline', DEFAULT_PIPELINE)
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('topic', 'camera1/image_raw/compressed')
        # Throttle published rate; 0 = publish every frame pulled.
        self.declare_parameter('publish_rate_hz', 10.0)

        pipeline_str = str(self.get_parameter('pipeline').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        topic = str(self.get_parameter('topic').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self._min_interval_ns = (
            int(1e9 / publish_rate_hz) if publish_rate_hz > 0.0 else 0
        )
        # Schedule-based rate limiter: advance the target time by one
        # interval per publish so jitter averages out.
        self._next_publish_ns: Optional[int] = None

        self.pub = self.create_publisher(CompressedImage, topic, IMAGE_QOS)

        Gst.init(None)

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as exc:
            raise RuntimeError(f'Failed to parse GStreamer pipeline: {exc}') from exc

        self.appsink = self.pipeline.get_by_name('sink')
        if self.appsink is None:
            raise RuntimeError(
                'GStreamer pipeline is missing an element named "sink". '
                'Pipeline must end with `appsink name=sink`.'
            )

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_bus_message)

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('GStreamer pipeline failed to enter PLAYING state')

        state_ret, state, _pending = self.pipeline.get_state(5 * Gst.SECOND)
        if state_ret != Gst.StateChangeReturn.SUCCESS or state != Gst.State.PLAYING:
            raise RuntimeError(
                f'GStreamer pipeline did not reach PLAYING within 5s '
                f'(state={state.value_nick}, ret={state_ret.value_nick}).'
            )

        self._stop = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            f'gstreamer_camera ready; publishing CompressedImage on {topic!r} '
            f'(target={publish_rate_hz:.1f} Hz, frame_id={self.frame_id!r}).'
        )

    # ---- worker thread --------------------------------------------------

    def _worker(self) -> None:
        """Pull-based frame loop (decoupled from GStreamer streaming thread)."""
        pull_timeout_ns = int(0.5 * Gst.SECOND)
        while not self._stop.is_set():
            sample = self.appsink.emit('try-pull-sample', pull_timeout_ns)
            if sample is None:
                continue
            self._handle_sample(sample)

    def _handle_sample(self, sample) -> None:
        """Copy one JPEG buffer out of GStreamer and publish it (with rate limit)."""
        if self._min_interval_ns > 0:
            now_ns = self.get_clock().now().nanoseconds
            if self._next_publish_ns is None:
                self._next_publish_ns = now_ns
            if now_ns < self._next_publish_ns:
                return
            self._next_publish_ns += self._min_interval_ns
            if self._next_publish_ns < now_ns:
                self._next_publish_ns = now_ns + self._min_interval_ns

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return
        try:
            # Copy JPEG bytes out of the GStreamer buffer. Typically ~400-800 KB
            # for a 1080p webcam — 10x smaller than raw BGR.
            jpeg_bytes = bytes(map_info.data)
        finally:
            buf.unmap(map_info)

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = jpeg_bytes
        try:
            self.pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'publish failed: {exc}')

    # ---- GStreamer bus --------------------------------------------------

    def _on_bus_message(self, _bus, message) -> bool:
        """Forward GStreamer bus errors/warnings to the ROS logger; exit on fatal."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f'GStreamer ERROR: {err.message} | {debug}')
            self.pipeline.set_state(Gst.State.NULL)
            self._loop.quit()
            rclpy.try_shutdown()
            sys.exit(1)
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self.get_logger().warn(f'GStreamer WARNING: {err.message} | {debug}')
        elif t == Gst.MessageType.EOS:
            self.get_logger().info('GStreamer pipeline reached EOS.')
        return True

    # ---- lifecycle ------------------------------------------------------

    def destroy_node(self) -> bool:
        """Stop the worker thread and tear down the GStreamer pipeline cleanly."""
        self._stop.set()
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._loop.quit()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = GstCameraNode()
    except Exception as exc:  # noqa: BLE001
        print(f'[gstreamer_camera] startup failed: {exc}', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
