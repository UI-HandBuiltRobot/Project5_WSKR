"""WSKR Floor Finder Node — "where can the robot safely drive?"

Takes the tilted-down camera feed, figures out which pixels look like floor
(based on color similarity to a sample patch near the bottom of the image
plus a gradient/edge test), and publishes a black-and-white mask where white
= floor and black = obstacle/not-floor.

Topics:
    subscribes  camera1/image_raw/compressed  — main robot camera (JPEG).
    publishes   WSKR/floor_mask               — mono8 binary floor mask.

Parameters (all live-tunable via ``ros2 param set`` or the Floor_Tuner GUI):
    resize_width/height, blur_kernel_size, bottom_sample_fraction,
    bottom_sample_height_fraction, morph_kernel_size, val_range,
    color_distance_threshold, gradient_threshold, highlight_thresh.

This is one of two "senses" the robot uses to navigate: the floor mask feeds
the whisker node (wskr_range_node) which ray-casts from the robot's feet
outward to measure how far it can drive in each direction.
"""
import threading

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

from .find_floor import Floor


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# Parameter names wired to Floor instance setters. Changing any of these via
# SetParameters live-reconfigures the running floor detector (wskr_floor_tuner
# relies on this to make its "Apply to /WSKR node" button actually work).
_FLOOR_PARAM_NAMES = (
    'resize_width',
    'resize_height',
    'blur_kernel_size',
    'bottom_sample_fraction',
    'bottom_sample_height_fraction',
    'morph_kernel_size',
    'val_range',
    'color_distance_threshold',
    'gradient_threshold',
    'highlight_thresh',
)


class WSKRFloorNode(Node):
    def __init__(self) -> None:
        super().__init__('wskr_floor')

        self.declare_parameter('resize_width', 640)
        self.declare_parameter('resize_height', 360)
        self.declare_parameter('blur_kernel_size', 9)
        self.declare_parameter('bottom_sample_fraction', 0.5)
        self.declare_parameter('bottom_sample_height_fraction', 0.25)
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('val_range', 40)
        self.declare_parameter('color_distance_threshold', 20)
        self.declare_parameter('gradient_threshold', 14)
        self.declare_parameter('highlight_thresh', 230)

        self.bridge = CvBridge()
        self.floor = Floor()
        self._floor_lock = threading.Lock()

        self._apply_all_params_to_floor()
        self.floor.enable_floor_mask(True)
        self.floor.enable_color_dist(True)

        # Live-reconfigure: re-push params into the Floor instance whenever
        # SetParameters is called (by wskr_floor_tuner's Apply button, etc.).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.image_sub = self.create_subscription(
            CompressedImage,
            'camera1/image_raw/compressed',
            self.image_callback,
            IMAGE_QOS,
        )
        self.mask_pub = self.create_publisher(Image, 'WSKR/floor_mask', IMAGE_QOS)

        self.get_logger().info(
            'WSKR floor node ready. Subscribed to camera1/image_raw/compressed; '
            'publishing WSKR/floor_mask.'
        )

    # ------------------------------------------------------------------ params

    def _apply_all_params_to_floor(self) -> None:
        """Read every floor param from the ROS param store and push to Floor."""
        gp = self.get_parameter
        with self._floor_lock:
            self.floor.set_resize_dimensions(
                gp('resize_width').value,
                gp('resize_height').value,
            )
            self.floor.set_blur_kernel_size(gp('blur_kernel_size').value)
            self.floor.set_bottom_sample_size(
                gp('bottom_sample_fraction').value,
                gp('bottom_sample_height_fraction').value,
            )
            self.floor.set_morph_kernel_size(gp('morph_kernel_size').value)
            self.floor.set_val_range(gp('val_range').value)
            self.floor.set_color_distance_threshold(gp('color_distance_threshold').value)
            self.floor.set_gradient_threshold(gp('gradient_threshold').value)
            self.floor.set_highlight_thresh(gp('highlight_thresh').value)

    def _on_set_parameters(self, params) -> SetParametersResult:
        """rclpy calls this BEFORE the new values land in the param store, so
        we compute the merged view (proposed overlaid on current) and push it
        into Floor. Returning success=True lets rclpy commit the change."""
        relevant = [p for p in params if p.name in _FLOOR_PARAM_NAMES]
        if not relevant:
            return SetParametersResult(successful=True)

        gp = self.get_parameter
        merged = {name: gp(name).value for name in _FLOOR_PARAM_NAMES}
        for p in relevant:
            merged[p.name] = p.value

        try:
            with self._floor_lock:
                self.floor.set_resize_dimensions(
                    merged['resize_width'], merged['resize_height'],
                )
                self.floor.set_blur_kernel_size(merged['blur_kernel_size'])
                self.floor.set_bottom_sample_size(
                    merged['bottom_sample_fraction'],
                    merged['bottom_sample_height_fraction'],
                )
                self.floor.set_morph_kernel_size(merged['morph_kernel_size'])
                self.floor.set_val_range(merged['val_range'])
                self.floor.set_color_distance_threshold(merged['color_distance_threshold'])
                self.floor.set_gradient_threshold(merged['gradient_threshold'])
                self.floor.set_highlight_thresh(merged['highlight_thresh'])
        except Exception as exc:  # noqa: BLE001
            return SetParametersResult(successful=False, reason=str(exc))

        return SetParametersResult(successful=True)

    def image_callback(self, msg: CompressedImage) -> None:
        # Decode at half resolution (libjpeg-turbo fast path). The Floor
        # detector internally resizes to 640x360 anyway, so feeding it
        # 960x540 instead of 1920x1080 just makes its resize cheaper with
        # no loss of mask quality.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            self.get_logger().error('Failed to decode compressed image.')
            return

        # Serialize against param updates so a tuner push doesn't change
        # Floor's internal state mid-processing.
        with self._floor_lock:
            self.floor.find_floor(frame)
            floor_mask = self.floor.get_floor_mask()
        if floor_mask is None:
            self.get_logger().warn('Floor detector returned no mask for current frame.')
            return

        # Publish the mask at its native processing resolution (resize_*).
        # Downstream nodes work in normalized coords, so they're resolution-agnostic.
        _, floor_mask = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)
        floor_mask = floor_mask.astype(np.uint8)

        out_msg = self.bridge.cv2_to_imgmsg(floor_mask, encoding='mono8')
        out_msg.header = msg.header
        self.mask_pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WSKRFloorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
