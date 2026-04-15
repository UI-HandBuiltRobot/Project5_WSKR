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

ament_python_install_package(WSKR)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  WSKR/model_Model_R1_Hardware_heading.json
  WSKR/params_Model_R1_Hardware_heading.json
  DESTINATION share/${PROJECT_NAME}/WSKR
)

install(PROGRAMS
  scripts/WSKR_floor.py
  scripts/WSKR_range.py
  scripts/WSKR_approach_action.py
  scripts/WSKR_dead_reckoning.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
