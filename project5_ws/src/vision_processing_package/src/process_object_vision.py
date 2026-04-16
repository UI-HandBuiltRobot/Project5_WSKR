#!/usr/bin/env python3
"""Process Object Vision — "call an external model to detect objects in the current frame."

Holds the latest camera frame in memory. When another node calls the
``detect_objects_service_v2`` service, this node encodes the frame as base64
and POSTs it to Roboflow's hosted inference API. The JSON response is
converted into our ``ImgDetectionData`` message (one bbox per detection).

A second service (``get_obj_properties_service``) augments individual
detections with class-specific geometry by calling the appropriate
``BboxToXYZ`` service.

This is the "toy" branch of the pipeline. ArUco-only workflows (e.g.
``Start_Aruco_Approach``) skip this node entirely.

Topics / services:
    subscribes  camera1/image_raw              — raw camera frame cache.
    serves      detect_objects_service_v2      — "what do you see right now?"
    serves      get_obj_properties_service     — add XYZ / size to a detection.
"""

import base64
import math
import time

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.srv import BboxToXYZ, DetectObjectsV2, GetObjProperties


class VisionInferenceService(Node):
    CAMERA_TOPIC = 'camera1/image_raw'
    DETECT_SERVICE_NAME = 'detect_objects_service_v2'
    OBJ_PROPERTIES_SERVICE_NAME = 'get_obj_properties_service'
    SIGNED_AR_ROTATION_DEGREES = '15'
    ROBOFLOW_API_KEY = 'LOmSMYkTrpOfXucsFHfT'
    ROBOFLOW_API_URL = 'https://detect.roboflow.com'
    ROBOFLOW_MODEL_ID = 'qlearning_block_ar'
    ROBOFLOW_MODEL_VERSION = '3'

    def __init__(self):
        """Create camera subscription plus inference/property services."""
        super().__init__('vision_inference_service')

        self.bridge = CvBridge()
        self.latest_frame = None
        cb_group = ReentrantCallbackGroup()

        self.create_subscription(Image, self.CAMERA_TOPIC, self.image_callback, 10)
        self.create_service(
            DetectObjectsV2,
            self.DETECT_SERVICE_NAME,
            self.handle_detect_objects,
            callback_group=cb_group,
        )
        self.create_service(
            GetObjProperties,
            self.OBJ_PROPERTIES_SERVICE_NAME,
            self.handle_get_obj_properties,
            callback_group=cb_group,
        )
        self.detect_client = self.create_client(
            DetectObjectsV2,
            self.DETECT_SERVICE_NAME,
            callback_group=cb_group,
        )
        self.bbox_xyz_client = self.create_client(
            BboxToXYZ,
            'bbox_to_xyz_service',
            callback_group=cb_group,
        )

        self.get_logger().info('VisionInferenceService initialized.')

    def image_callback(self, msg: Image):
        """Cache latest camera frame for service handlers."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f'Image decode failed: {exc}')

    def handle_detect_objects(self, request, response):
        """Run optional-rotation inference and return `ImgDetectionData`."""
        if self.latest_frame is None:
            response.success = False
            return response

        try:
            rotation_text = (request.rotation_degrees or '').strip()
            frame = self.apply_optional_rotation(self.latest_frame, rotation_text)
            response.detections = self.run_roboflow_inference(frame)
            response.success = True
        except Exception as exc:
            self.get_logger().error(f'DetectObjectsV2 failed: {exc}')
            response.success = False

        return response

    def handle_get_obj_properties(self, request, response):
        """Compute signed AR and XYZ by chaining detect and bbox services."""

        try:
            base_resp = self.call_detect_objects_service(request.id, '')
            rot_text = self.SIGNED_AR_ROTATION_DEGREES
            rotated_resp = self.call_detect_objects_service(request.id, rot_text)

            base_det = self.extract_detection_by_index(base_resp.detections, request.id)
            rotated_det = self.match_rotated_detection(
                rotated_resp.detections,
                base_det,
                rot_text,
                base_resp.detections.image_width,
                base_resp.detections.image_height,
            )

            raw_ar = self._aspect_ratio(base_det)
            rotated_ar = self._aspect_ratio(rotated_det)
            signed_ar = abs(raw_ar) if (rotated_ar - raw_ar) >= 0.0 else -abs(raw_ar)
            xyz_resp = self.call_bbox_to_xyz_service(
                base_det['x'],
                base_det['y'],
                base_det['width'],
                base_det['height'],
                base_resp.detections.image_width,
                base_resp.detections.image_height,
            )

            response.success = True
            response.signed_aspect_ratio = float(signed_ar)
            response.class_name = str(base_det['class_name'])
            response.x = float(base_det['x'])
            response.y = float(base_det['y'])
            response.width = float(base_det['width'])
            response.height = float(base_det['height'])
            response.x_mm = float(xyz_resp.x_mm)
            response.y_mm = float(xyz_resp.y_mm)
            response.z_mm = float(xyz_resp.z_mm)
        except Exception as exc:
            self.get_logger().error(f'GetObjProperties failed: {exc}')
            response.success = False
            response.signed_aspect_ratio = 0.0
            response.class_name = ''
            response.x = 0.0
            response.y = 0.0
            response.width = 0.0
            response.height = 0.0
            response.x_mm = 0.0
            response.y_mm = 0.0
            response.z_mm = 0.0

        return response

    def call_detect_objects_service(self, request_id, rotation_degrees_text):
        """Call DetectObjectsV2 and return successful result."""
        if not self.detect_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('DetectObjectsV2 service unavailable.')

        req = DetectObjectsV2.Request()
        req.id = int(request_id)
        req.rotation_degrees = str(rotation_degrees_text)

        future = self.detect_client.call_async(req)
        start_time = time.perf_counter()
        while not future.done():
            if (time.perf_counter() - start_time) > 5.0:
                raise RuntimeError('DetectObjectsV2 timed out.')
            time.sleep(0.01)

        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('DetectObjectsV2 call failed.')

        return result

    def call_bbox_to_xyz_service(
        self,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        image_width,
        image_height,
    ):
        """Call BboxToXYZ and return successful result."""
        if not self.bbox_xyz_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('BboxToXYZ service unavailable.')

        req = BboxToXYZ.Request()
        req.bbox_x = float(bbox_x)
        req.bbox_y = float(bbox_y)
        req.bbox_width = float(bbox_width)
        req.bbox_height = float(bbox_height)
        req.image_width = int(image_width)
        req.image_height = int(image_height)

        future = self.bbox_xyz_client.call_async(req)
        start_time = time.perf_counter()
        while not future.done():
            if (time.perf_counter() - start_time) > 5.0:
                raise RuntimeError('BboxToXYZ timed out.')
            time.sleep(0.01)

        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('BboxToXYZ call failed.')

        return result

    def extract_detection_by_index(self, detections_msg, detection_index):
        """Return one detection dict after validating index against aligned arrays."""
        count = min(
            len(detections_msg.x),
            len(detections_msg.y),
            len(detections_msg.width),
            len(detections_msg.height),
            len(detections_msg.class_name),
            len(detections_msg.confidence),
        )
        if count == 0:
            raise RuntimeError('No detections returned.')
        if detection_index < 0 or detection_index >= count:
            raise RuntimeError(f'Detection index {detection_index} out of range for {count}.')

        idx = int(detection_index)
        return {
            'index': idx,
            'class_name': detections_msg.class_name[idx],
            'x': float(detections_msg.x[idx]),
            'y': float(detections_msg.y[idx]),
            'width': float(detections_msg.width[idx]),
            'height': float(detections_msg.height[idx]),
        }

    def match_rotated_detection(
        self,
        detections_msg,
        base_detection,
        rotation_degrees_text,
        image_width,
        image_height,
    ):
        """Match rotated detections to the base detection by class and nearest predicted rotated center."""
        count = min(
            len(detections_msg.x),
            len(detections_msg.y),
            len(detections_msg.width),
            len(detections_msg.height),
            len(detections_msg.class_name),
            len(detections_msg.confidence),
        )
        if count == 0:
            raise RuntimeError('No detections were returned for rotated inference.')

        try:
            rotation_degrees = float(rotation_degrees_text)
        except ValueError as exc:
            raise RuntimeError(
                f'Invalid signed_ar_rotation_degrees value: {rotation_degrees_text}'
            ) from exc

        center_x = float(image_width) / 2.0
        center_y = float(image_height) / 2.0
        theta = math.radians(rotation_degrees)

        shifted_x = float(base_detection['x']) - center_x
        shifted_y = float(base_detection['y']) - center_y
        predicted_x = (shifted_x * math.cos(theta)) - (shifted_y * math.sin(theta)) + center_x
        predicted_y = (shifted_x * math.sin(theta)) + (shifted_y * math.cos(theta)) + center_y

        all_candidates = [
            {
                'index': int(i),
                'class_name': detections_msg.class_name[i],
                'x': float(detections_msg.x[i]),
                'y': float(detections_msg.y[i]),
                'width': float(detections_msg.width[i]),
                'height': float(detections_msg.height[i]),
            }
            for i in range(count)
        ]
        class_candidates = [
            detection
            for detection in all_candidates
            if detection['class_name'] == base_detection['class_name']
        ]
        candidates = class_candidates or all_candidates

        return min(
            candidates,
            key=lambda detection: math.hypot(
                detection['x'] - predicted_x,
                detection['y'] - predicted_y,
            ),
        )

    def apply_optional_rotation(self, image_bgr, rotation_text):
        """Return image unchanged for empty rotation; otherwise return affine-rotated image."""
        if rotation_text == '':
            return image_bgr

        rotation_degrees = float(rotation_text)
        rotation_center = (image_bgr.shape[1] / 2.0, image_bgr.shape[0] / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_degrees, 1.0)
        return cv2.warpAffine(
            image_bgr,
            rotation_matrix,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )

    def run_roboflow_inference(self, image_bgr):
        """Encode frame, call Roboflow, and map predictions into `ImgDetectionData`."""

        success, encoded = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise RuntimeError('Failed to encode image for Roboflow inference.')

        image_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        endpoint = (
            f'{self.ROBOFLOW_API_URL.rstrip("/")}/'
            f'{self.ROBOFLOW_MODEL_ID}/{self.ROBOFLOW_MODEL_VERSION}'
        )

        start_time = time.perf_counter()
        resp = requests.post(
            endpoint,
            params={'api_key': self.ROBOFLOW_API_KEY},
            data=image_b64,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10.0,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        resp.raise_for_status()
        data = resp.json()

        predictions = data.get('predictions', [])
        inference_time_ms = float(data.get('time', elapsed_ms / 1000.0)) * 1000.0

        msg = ImgDetectionData()
        msg.image_width = int(image_bgr.shape[1])
        msg.image_height = int(image_bgr.shape[0])
        msg.inference_time = inference_time_ms

        for pred in predictions:
            cx = float(pred.get('x', 0.0))
            cy = float(pred.get('y', 0.0))
            w = float(pred.get('width', 0.0))
            h = float(pred.get('height', 0.0))

            msg.detection_ids.append(str(pred.get('detection_id', '')))
            msg.x.append(cx)
            msg.y.append(cy)
            msg.width.append(w)
            msg.height.append(h)
            msg.distance.append(0.0)
            msg.class_name.append(str(pred.get('class', 'unknown')))
            msg.confidence.append(float(pred.get('confidence', 0.0)))
            msg.aspect_ratio.append((w / h) if h > 0.0 else 0.0)

        return msg

    def _aspect_ratio(self, detection):
        """Compute width/height for detection dict."""
        height = float(detection['height'])
        if height <= 0.0:
            raise RuntimeError('Detected bbox height must be positive.')
        return float(detection['width']) / height


def main(args=None):
    """Start the multi-threaded vision node executor and keep services alive until shutdown."""
    rclpy.init(args=args)
    node = VisionInferenceService()
    executor = MultiThreadedExecutor(num_threads=2)
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