from setuptools import setup

package_name = 'utilities'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Consolidated tuning, diagnostic, and teleop GUIs for WSKR.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Floor_Tuner=utilities.Floor_Tuner:main',
            'Heading_Tuner=utilities.Heading_Tuner:main',
            'Mecanum_Teleop=utilities.Mecanum_Teleop:main',
            'WSKR_Dashboard=utilities.WSKR_Dashboard:main',
            'Start_Aruco_Approach=utilities.Start_Aruco_Approach:main',
            'select_object_and_start_navigating_spoof=utilities.select_object_and_start_navigating_spoof:main',
            'select_object_and_start_navigating_live=utilities.select_object_and_start_navigating_live:main',
        ],
    },
)
