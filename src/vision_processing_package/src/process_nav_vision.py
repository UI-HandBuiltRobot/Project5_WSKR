"""Nav Vision — passthrough image republisher (placeholder for future nav processing).

Right now this node simply subscribes to ``camera1/image_raw`` and republishes
each frame on ``nav_image_topic``. It exists as a hook for navigation-specific
image processing (filtering, warping, overlays) to be added later.

Topics:
    subscribes  camera1/image_raw    — raw camera feed.
    publishes   nav_image_topic      — pass-through for now.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class NavVision(Node):

    # node definition and parameters
    def __init__(self):
        super().__init__('nav_vision_node')
        self.publisher_ = self.create_publisher(Image, 'nav_image_topic', 10)
        self.subscription = self.create_subscription(
            Image,
            'camera1/image_raw', # camera1 to follow usb_cam topic, might be parameter
            self.read_image,
            10)
        self.subscription  # prevent unused variable warning. stores it as a variable to avoid losing the callback
    #    self.timer = self.create_timer(0.1, self.timer_callback)  # publish every 0.1 seconds. should likely make this a parameter in launch file
        self.bridge = CvBridge()
        
    def publish_image(self, cv_image):
        """Wrap an OpenCV BGR image as a ROS Image and publish it."""
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.publisher_.publish(ros_image)

    def read_image(self, msg):
        """Convert an incoming ROS Image into OpenCV and republish it."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.publish_image(cv_image)


def main(args=None):
    rclpy.init(args=args)
    node = NavVision()
    node.get_logger().info('NavVision Node ready.')

    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()

