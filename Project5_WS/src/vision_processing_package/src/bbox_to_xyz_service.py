#!/usr/bin/env python3
"""Bbox-to-XYZ Service (1D / power-law version) — "how far away is that thing?"

Given a bounding box on the main camera, estimates the forward distance
(mm) to the object using a calibrated power-law of the bbox's bottom-edge
y-position in the frame. Assumes the object sits on the floor and the
camera is fixed — i.e. "things further away appear higher in the image".

Companion to ``bbox_to_xyz_service_2D.py`` which does a full 2D homography
mapping; this 1D version is the simpler fallback.

Services:
    serves  bbox_to_xyz_service  — BboxToXYZ.srv → returns (x_mm, y_mm, z_mm).
"""

import rclpy
from rclpy.node import Node

from robot_interfaces.srv import BboxToXYZ


class BboxToXYZServiceNode(Node):
    def __init__(self):
        """Register the `bbox_to_xyz_service` endpoint and initialize node logging."""
        super().__init__('bbox_to_xyz_node')

        self.create_service(BboxToXYZ, 'bbox_to_xyz_service', self._handle_request)
        self.get_logger().info('BboxToXYZ Service ready.')

    def _handle_request(self, request, response):
        """Convert bbox vertical position into forward distance using the calibrated power-law mapping."""
        if request.image_height > 0:
            y_norm = float(request.bbox_y) / float(request.image_height)
            h_norm = float(request.bbox_height) / float(request.image_height)
        else:
            # If image dimensions are not provided, treat bbox values as already normalized.
            y_norm = float(request.bbox_y)
            h_norm = float(request.bbox_height)

        # Use the middle of the bottom edge: bbox center-y plus half bbox height.
        bottom_mid_y_norm = y_norm + (h_norm / 2.0)

        if bottom_mid_y_norm <= 0.0:
            self.get_logger().error(
                f'Invalid bottom_mid_y_norm={bottom_mid_y_norm:.6f}; cannot apply power-law calibration.'
            )
            response.success = False
            response.x_mm = 0.0
            response.y_mm = 0.0
            response.z_mm = 0.0
            return response

        # Power-law calibration: x_mm = a * (y_norm)^b.
        x_mm = 119.25232 * (bottom_mid_y_norm ** -1.31798)

        self.get_logger().info(
            f'BboxToXYZ: bottom_mid_y_norm={bottom_mid_y_norm:.4f} -> x_mm={x_mm:.1f} (y=0, z=0)'
        )

        response.success = True
        response.x_mm = float(x_mm)
        response.y_mm = 0.0
        response.z_mm = 0.0

        return response


def main(args=None):
    """Run the BboxToXYZ node until interrupted, then shut ROS down cleanly."""
    rclpy.init(args=args)
    node = BboxToXYZServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()