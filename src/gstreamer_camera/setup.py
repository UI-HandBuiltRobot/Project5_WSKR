from setuptools import setup

package_name = 'gstreamer_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/gstreamer_camera.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Project5 User',
    maintainer_email='user@example.com',
    description='Low-latency GStreamer-backed camera source for the Jetson Orin.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gstreamer_camera_node = gstreamer_camera.gst_cam_node:main',
        ],
    },
)
