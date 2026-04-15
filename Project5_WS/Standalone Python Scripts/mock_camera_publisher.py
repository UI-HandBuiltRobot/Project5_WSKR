from pathlib import Path
from typing import Optional

import argparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def pick_image_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title='Select an image file',
        filetypes=[
            ('JPEG files', '*.jpg *.jpeg'),
            ('All image files', '*.png *.bmp *.tif *.tiff'),
            ('All files', '*.*'),
        ],
    )
    root.destroy()

    if not selected:
        return None
    return selected


class MockCameraPublisher(Node):
    def __init__(self, image_path: str, topic: str, publish_hz: float):
        super().__init__('mock_camera_publisher')

        resolved_path = str(Path(image_path).expanduser().resolve())
        frame = cv2.imread(resolved_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f'Failed to load image: {resolved_path}')

        self.frame = frame
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)

        period_s = 1.0 / max(0.1, publish_hz)
        self.timer = self.create_timer(period_s, self.publish_frame)

        self.get_logger().info(
            f'Publishing {resolved_path} to {topic} at {1.0 / period_s:.1f} Hz'
        )

    def publish_frame(self):
        msg = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Publish a single image repeatedly to a ROS2 image topic.'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='',
        help='Path to image file. If omitted, a file picker is opened.',
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='camera1/img_raw',
        help='ROS2 topic to publish image messages to.',
    )
    parser.add_argument(
        '--hz',
        type=float,
        default=5.0,
        help='Publish frequency in Hz.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image.strip()
    if not image_path:
        picked = pick_image_file()
        if not picked:
            print('No image selected. Provide --image or choose a file in the picker dialog.')
            return
        image_path = picked

    rclpy.init()
    try:
        node = MockCameraPublisher(image_path=image_path, topic=args.topic, publish_hz=args.hz)
    except Exception as exc:
        print(f'Error: {exc}')
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
