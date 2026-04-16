cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

#find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImgDetectionData.msg"
  "srv/SelectObject.srv"
  "srv/DetectObjects.srv"
  "srv/DetectObjectsV2.srv"
  "srv/BboxToXYZ.srv"
  "srv/GetObjProperties.srv"
  "srv/GraspObject.srv"
  "srv/ApproachObject.srv"
  "srv/ArduinoController.srv"
  "srv/GA.srv"
  "action/XArm.action"
  "action/ApproachObject.action"
  DEPENDENCIES std_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

## install the python scripts for ROS2 to find
#install(
#  PROGRAMS
#    src/manager_to_vision.py
#  DESTINATION lib/${PROJECT_NAME}   
#)

# WILL NEED LATER TO LAUNCH ALL OF THE INTERFACE NODES
# Install launch files if you have them
#install(
#  DIRECTORY launch
#  DESTINATION share/${PROJECT_NAME}/launch
#)

ament_package()
