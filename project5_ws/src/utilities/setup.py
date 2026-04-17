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
    description='Consolidated tuning, diagnostic, and teleop GUIs for wskr.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'floor_tuner=utilities.floor_tuner:main',
            'heading_tuner=utilities.heading_tuner:main',
            'mecanum_teleop=utilities.mecanum_teleop:main',
            'wskr_dashboard=utilities.wskr_dashboard:main',
            'select_object_and_start_navigating_spoof=utilities.select_object_and_start_navigating_spoof:main',
            'select_object_and_start_navigating_live=utilities.select_object_and_start_navigating_live:main',
        ],
    },
)
