#!/home/ros_setup/ros2_ws/venv/bin/python
"""State Manager — the top-level "what should the robot do right now?" controller.

This is a finite state machine. At any moment the robot is in exactly one
state, and each state kicks off one action or service call. The callback
that completes that work decides which state to enter next.

States (see :class:`RobotState`):
    IDLE          — do nothing.
    SEARCH        — call the object-detection service; → SELECT on success.
    SELECT        — call the object-selection service; → APPROACH_OBJ.
    APPROACH_OBJ  — call the approach service; → GRASP when close enough.
    GRASP         — run the XArm grasp action; → FIND_BOX on success.
    FIND_BOX      — (stub) → APPROACH_BOX.
    APPROACH_BOX  — (stub) → DROP.
    DROP          — (stub) → IDLE.
    WANDER        — placeholder for free-roaming behavior.
    STOPPED/ERROR — halt / error reporting.

This node is intended for the full toy-pick-and-drop workflow. For
ArUco-only testing you probably want to skip it and launch the WSKR stack
+ the ``Start_Aruco_Approach`` GUI directly.
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from enum import Enum
import threading
from std_msgs.msg import String
from robot_interfaces.srv import DetectObjects  # type: ignore
from robot_interfaces.srv import SelectObject  # type: ignore
from robot_interfaces.srv import ApproachObject as ApproachObjectSrv  # type: ignore
from robot_interfaces.action import XArm  # type: ignore


class RobotState(Enum):
    IDLE = 0
    SEARCH = 1
    SELECT = 2
    APPROACH_OBJ = 3
    GRASP = 4
    FIND_BOX = 5
    APPROACH_BOX = 6
    DROP = 7
    STOPPED = 8
    ERROR = 9
    WANDER = 10


class StateManager:
    """Thread-safe holder for the current state plus a couple of scratch fields."""

    def __init__(self):
        self._state = RobotState.SEARCH  # start in SEARCH for immediate testing
        self._lock = threading.Lock()
        self.detected_objects = None
        self.selected_object = None

    def get_state(self):
        """Return the current state (thread-safe)."""
        with self._lock:
            return self._state

    def set_state(self, new_state):
        """Replace the current state (thread-safe)."""
        with self._lock:
            self._state = new_state


class StateManagerNode(Node):

    def __init__(self, state_manager):
        super().__init__('state_manager_node')

        # Store reference to state manager
        self.state_manager = state_manager
        self.state_timer = None

        # Service client for object detection
        self.detection_client = self.create_client(DetectObjects, 'detect_objects_service')
        while not self.detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for object detection service...")
        self.get_logger().info("ObjectDetection service client ready.")

        # Service client for object selection
        self.selection_client = self.create_client(SelectObject, 'select_object_service')
        while not self.selection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for object selection service...")
        self.get_logger().info("ObjectSelection service client ready.")

        # Service client for object approach
        self.approach_client = self.create_client(ApproachObjectSrv, 'approach_object_service')
        while not self.approach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for object approach service...")
        self.get_logger().info("ObjectApproach service client ready.")

        # Action client for object grasp
        self.grasp_action_client = ActionClient(self, XArm, 'xarm_grasp_action')
        while not self.grasp_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for xarm grasp action server...")
        self.get_logger().info("XArm grasp action client ready.")

        self.request_id = 0

        # Command subscription
        #self.create_subscription(String, 'command', self.command_callback, 10)

        # Start the initial state
        self.start_state(self.state_manager.get_state())

    # ---------------- COMMANDS ----------------
    def command_callback(self, msg: String):
        """Force the state machine into a named state via a ``command`` topic message."""
        cmd = msg.data.lower()
        if self.state_manager.get_state() == RobotState.STOPPED:
            self.get_logger().info("Robot is STOPPED. Ignoring command.")
            return

        command_map = {
            "idle": RobotState.IDLE,
            "search": RobotState.SEARCH,
            "select": RobotState.SELECT,
            "approach_obj": RobotState.APPROACH_OBJ,
            "grasp": RobotState.GRASP,
            "find_box": RobotState.FIND_BOX,
            "approach_box": RobotState.APPROACH_BOX,
            "drop": RobotState.DROP,
            "wander": RobotState.WANDER
        }

        if cmd in command_map:
            self.start_state(command_map[cmd])
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    # ---------------- FSM ----------------
    def cancel_state_timer(self):
        """Stop the one-shot timer that drives the current state's work."""
        if self.state_timer is not None:
            self.state_timer.cancel()
            self.state_timer = None

    def start_state(self, state: RobotState):
        """Enter a new state: update the flag and schedule its work."""
        self.cancel_state_timer()
        self.state_manager.set_state(state)
        self.get_logger().info(f"STATE moved to {state.name}")

        if state == RobotState.IDLE:
            return

        elif state == RobotState.SEARCH:
            self.get_logger().info("Starting SEARCH state...")
            self.state_timer = self.create_timer(1.0, self.search)

        elif state == RobotState.SELECT:
            self.get_logger().info("Entering SELECT state...")
            self.state_timer = self.create_timer(1.0, self.select_obj)

        elif state == RobotState.APPROACH_OBJ:
            self.get_logger().info("Entering APPROACH_OBJ state...")
            self.state_timer = self.create_timer(1.5, self.approach_obj)

        elif state == RobotState.GRASP:
            self.get_logger().info("Entering GRASP state...")
            self.state_timer = self.create_timer(1.0, self.grasp)

        elif state == RobotState.FIND_BOX:
            self.get_logger().info("Entering FIND_BOX state...")
            self.state_timer = self.create_timer(1.0, self.find_box)

        elif state == RobotState.APPROACH_BOX:
            self.get_logger().info("Entering APPROACH_BOX state...")
            self.state_timer = self.create_timer(1.5, self.approach_box)

        elif state == RobotState.DROP:
            self.get_logger().info("Entering DROP state...")
            self.state_timer = self.create_timer(1.0, self.drop)

        elif state == RobotState.WANDER:
            self.get_logger().info("Entering WANDER state...")
            self.state_timer = self.create_timer(2.0, self.wander)

    # ---------------- STATE METHODS ----------------
    def search(self):
        """SEARCH: ask the vision service what's in view."""
        self.cancel_state_timer()
        self.get_logger().info("SEARCH: requesting object detection...")

        req = DetectObjects.Request()
        req.id = 1  ## NEED TO UPDATE THIS. MAYBE WITH TIMESTAMP OR JUST INCREMENTAL COUNTING

        future = self.detection_client.call_async(req)
        future.add_done_callback(self.detection_done)

    def detection_done(self, future):
        """Detection finished: advance to SELECT if anything was found, else retry."""
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Detection service failed: {e}")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)
            return

        if response.success and len(response.detections.x) > 0:
            self.get_logger().info(f"Detected objects: {len(response.detections.x)}")
            self.state_manager.detected_objects = response.detections
            self.state_manager.set_state(RobotState.SELECT)
            self.start_state(RobotState.SELECT)
        else:
            self.get_logger().warn("No objects detected, switching to WANDER")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)

    def select_obj(self):
        """SELECT: ask the selection service which detected object to go after."""
        self.cancel_state_timer()
        self.get_logger().info("Entering SELECT state: requesting object selection.")

        req = SelectObject.Request()
        req.id = 1
        req.detected_objs = self.state_manager.detected_objects

        future = self.selection_client.call_async(req)
        future.add_done_callback(self.selection_done)

    def selection_done(self, future):
        """Selection finished: advance to APPROACH_OBJ if something was chosen."""
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Selection service failed: {e}")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)
            return

        if response.success:
            self.get_logger().info(f"Selected object.")
            self.state_manager.selected_object = response.selected_obj
            self.state_manager.set_state(RobotState.APPROACH_OBJ)
            self.start_state(RobotState.APPROACH_OBJ)
        else:
            self.get_logger().warn("No objects selected, retrying SEARCH.")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)

    def approach_obj(self):
        """APPROACH_OBJ: call the approach service and let WSKR drive toward it."""
        self.cancel_state_timer()
        self.get_logger().info("Entering APPROACH_OBJ state: requesting object approach.")

        req = ApproachObjectSrv.Request()
        req.id = 1
        req.selected_obj = self.state_manager.selected_object

        future = self.approach_client.call_async(req)
        future.add_done_callback(self.approach_done)

    def approach_done(self, future):
        """Approach finished: advance to GRASP if close enough, else restart."""
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Approach service failed: {e}")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)
            return
        
        # did arduino fail to move?
        if response.movement_success:
            self.get_logger().info("Robot moved toward target.")
        else:
            self.get_logger().warn("Robot failed to move/Arduino error. Retrying Stopping")
            # need to add a wait or stop state here or a way to reset arduino

        # close enough to grasp
        if response.proximity_success:
            self.get_logger().info(f"Object within grasping distance.")
            self.state_manager.set_state(RobotState.GRASP)
            self.start_state(RobotState.GRASP)
        else:
            self.get_logger().warn("Not within range. Retrying SEARCH.")
            self.state_manager.set_state(RobotState.SEARCH)
            self.start_state(RobotState.SEARCH)


    def grasp(self):
        """GRASP: fire off the XArm grasp action and wait for feedback."""
        self.cancel_state_timer()
        self.get_logger().info("Entering GRASP state: grasping object.")

        goal_msg = XArm.Goal()
        self.request_id += 1
        goal_msg.id = self.request_id
        goal_msg.selected_obj = self.state_manager.selected_object

        send_goal_future = self.grasp_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.grasp_feedback_callback,
        )
        send_goal_future.add_done_callback(self.grasp_goal_response_done)

    def grasp_feedback_callback(self, feedback_msg):
        """Log per-stage progress while the grasp action is running."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"GRASP feedback: stage={feedback.current_stage}, progress={feedback.progress:.2f}, success={feedback.success}"
        )

    def grasp_goal_response_done(self, future):
        """Grasp action accepted/rejected: wait for the result if accepted."""
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Failed to send GRASP action goal: {e}")
            self.state_manager.set_state(RobotState.GRASP)
            self.start_state(RobotState.GRASP)
            return

        if not goal_handle.accepted:
            self.get_logger().warn("GRASP action goal rejected. Retrying GRASP.")
            self.state_manager.set_state(RobotState.GRASP)
            self.start_state(RobotState.GRASP)
            return

        self.get_logger().info("GRASP action goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_done)

    def grasp_result_done(self, future):
        """Grasp result: advance to FIND_BOX on success, retry GRASP otherwise."""
        try:
            action_response = future.result()
            result = action_response.result
        except Exception as e:
            self.get_logger().error(f"Grasp action failed: {e}")
            self.state_manager.set_state(RobotState.GRASP)
            self.start_state(RobotState.GRASP)
            return

        if result.current_number == 1:
            self.get_logger().info("Object grasped.")
            self.state_manager.set_state(RobotState.FIND_BOX)
            self.start_state(RobotState.FIND_BOX)
        else:
            self.get_logger().warn("No object in gripper. Trying to grasp again.")
            self.state_manager.set_state(RobotState.GRASP)
            self.start_state(RobotState.GRASP)

    def find_box(self):
        """FIND_BOX: stub — immediately advance to APPROACH_BOX."""
        self.cancel_state_timer()
        self.start_state(RobotState.APPROACH_BOX)

    def approach_box(self):
        """APPROACH_BOX: stub — immediately advance to DROP."""
        self.cancel_state_timer()
        self.start_state(RobotState.DROP)

    def drop(self):
        """DROP: stub — return to IDLE."""
        self.cancel_state_timer()
        self.start_state(RobotState.IDLE)

    def wander(self):
        """WANDER: placeholder for free-roam behavior."""
        self.cancel_state_timer()
        self.start_state(RobotState.WANDER)


# ---------------- MAIN ----------------
def main(args=None):
    rclpy.init(args=args)

    state_manager = StateManager()
    node = StateManagerNode(state_manager)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
