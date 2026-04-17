"""Object Selection Service — "out of all the detections, pick one to chase."

When the state manager is in SELECT, it hands this node a full list of
detected objects and asks for a single target. The current policy is
"pick the closest one" — the detection with the smallest ``distance``
field is returned.

Services:
    serves  select_object_service  — SelectObject.srv.
"""
import rclpy
from rclpy.node import Node
from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.srv import SelectObject

class ObjectSelection(Node):
    def __init__(self):
        super().__init__('object_selection_node')

        # Create service to select object from detections sent by state manager
        self.srv = self.create_service(SelectObject, 'select_object_service', self.detect_object_callback)

        self.get_logger().info("ObjectSelection service ready.")

    def detect_object_callback(self, request, response):
        """Select the closest detection (minimum distance) from the input list."""
        #Select one object from the detected objects sent in the request.
        #Picks the object with the minimum distance.
        
        self.get_logger().info('Received request to select object')

        # request.detected_objs is the ImgDetectionData sent from state manager
        detected_objects = request.detected_objs
        id = request.id
        self.get_logger().info(f"Request id: {id}")

        # Pick object with minimum distance
        self.get_logger().info("checking min distance")

        min_index = detected_objects.distance.index(min(detected_objects.distance))
        self.get_logger().info("min index: {}".format(min_index))
        for i in range(len(detected_objects.distance)):
            self.get_logger().info(f"Object distances: {detected_objects.distance[i]}")


        selected_object = ImgDetectionData()
        selected_object.image_width = detected_objects.image_width
        selected_object.image_height = detected_objects.image_height
        selected_object.inference_time = detected_objects.inference_time
        selected_object.x = [detected_objects.x[min_index]]
        selected_object.y = [detected_objects.y[min_index]]
        selected_object.width = [detected_objects.width[min_index]]
        selected_object.height = [detected_objects.height[min_index]]
        selected_object.class_name = [detected_objects.class_name[min_index]]
        selected_object.confidence = [detected_objects.confidence[min_index]]
        selected_object.aspect_ratio = [detected_objects.aspect_ratio[min_index]]
        selected_object.detection_ids = [detected_objects.detection_ids[min_index]]
        selected_object.distance = [detected_objects.distance[min_index]]

        response.success = True
        response.selected_obj = selected_object

        self.get_logger().info(f"Selected object at distance: {selected_object.distance[0]}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSelection()
    node.get_logger().info('Object Selection Node ready.')

    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
