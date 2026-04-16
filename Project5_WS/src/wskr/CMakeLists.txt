cmake_minimum_required(VERSION 3.8)
project(wskr)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(robot_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ApproachObject.action"
  DEPENDENCIES robot_interfaces
)

ament_python_install_package(wskr)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  wskr/models/
  DESTINATION share/${PROJECT_NAME}/models
  FILES_MATCHING PATTERN "*.json"
)

install(PROGRAMS
  scripts/wskr_floor.py
  scripts/wskr_range.py
  scripts/wskr_approach_action.py
  scripts/wskr_dead_reckoning.py
  scripts/wskr_autopilot.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
