from setuptools import find_packages, setup

package_name = 'arduino'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/arduino.launch.py']),
        ('share/' + package_name + '/config', ['config/serial.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Project5 User',
    maintainer_email='user@example.com',
    description='Serial bridge to the mecanum base Arduino: Twist in, Odometry out.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mecanum_serial_bridge = arduino.mecanum_serial_bridge:main',
        ],
    },
)
