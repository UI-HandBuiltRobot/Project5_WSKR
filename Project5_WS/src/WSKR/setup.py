from setuptools import find_packages, setup

package_name = 'WSKR'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/FirstCal.json']),
        ('share/' + package_name + '/launch', ['launch/wskr.launch.py']),
        ('share/' + package_name + '/WSKR', [
            'WSKR/model_Model_R1_Hardware_heading.json',
            'WSKR/params_Model_R1_Hardware_heading.json',
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
            'WSKR = WSKR.wskr_floor_node:main',
            'WSKR_range = WSKR.wskr_range_node:main',
            'WSKR_approach_action = WSKR.approach_action_server:main',
            'WSKR_dead_reckoning = WSKR.dead_reckoning_node:main',
        ],
    },
)
