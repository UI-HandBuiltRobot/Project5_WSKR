cmake_minimum_required(VERSION 3.8)
project(vision_processing_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install the Python executables
install(PROGRAMS
  src/process_object_vision.py
  src/object_selection.py
  src/bbox_to_xyz_service.py
  src/bbox_to_xyz_service_2D.py
  src/process_nav_vision.py
  src/cv2_orientation_testing.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/ # The directory containing your launch files within your package
  DESTINATION share/${PROJECT_NAME} # The destination in the install space
)

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

ament_package()