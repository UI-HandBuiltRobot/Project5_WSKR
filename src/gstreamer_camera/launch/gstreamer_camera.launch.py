"""Standalone launch for gstreamer_camera_node.

Publishes sensor_msgs/CompressedImage (JPEG) on camera1/image_raw/compressed
by default. The node-level pipeline is a minimal v4l2src -> appsink that
hands through raw MJPEG bytes; no GPU decode required.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pipeline_arg = DeclareLaunchArgument(
        'pipeline', default_value='',
        description=(
            'GStreamer pipeline ending in `appsink name=sink`. '
            'If blank, the node-level default (Jetson L4T HW pipeline) is used.'
        ),
    )
    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value='camera_frame',
        description='ROS TF frame_id stamped on each image message.',
    )
    topic_arg = DeclareLaunchArgument(
        'topic', default_value='camera1/image_raw/compressed',
        description='Output CompressedImage topic.',
    )

    # Build the parameters dict, omitting pipeline if the launch arg is empty
    # so the node keeps its compiled default.
    def _make_params(context):
        params = {
            'frame_id': LaunchConfiguration('frame_id').perform(context),
            'topic': LaunchConfiguration('topic').perform(context),
        }
        pipe = LaunchConfiguration('pipeline').perform(context)
        if pipe:
            params['pipeline'] = pipe
        return [params]

    from launch.actions import OpaqueFunction

    def _spawn(context):
        return [
            Node(
                package='gstreamer_camera',
                executable='gstreamer_camera_node',
                name='gstreamer_camera',
                output='screen',
                parameters=_make_params(context),
            )
        ]

    return LaunchDescription([
        pipeline_arg,
        frame_id_arg,
        topic_arg,
        OpaqueFunction(function=_spawn),
    ])
