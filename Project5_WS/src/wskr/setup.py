from setuptools import find_packages, setup

package_name = 'wskr'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/Whisker_Calibration.json']),
        ('share/' + package_name + '/launch', ['launch/wskr.launch.py']),
        ('share/' + package_name + '/models', [
            'wskr/models/model_heading_002.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Project5 User',
    maintainer_email='user@example.com',
    description='Floor masking and whisker range estimation nodes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wskr_floor = wskr.wskr_floor_node:main',
            'wskr_range = wskr.wskr_range_node:main',
            'wskr_approach_action = wskr.approach_action_server:main',
            'wskr_dead_reckoning = wskr.dead_reckoning_node:main',
            'wskr_autopilot = wskr.wskr_autopilot:main',
        ],
    },
)
