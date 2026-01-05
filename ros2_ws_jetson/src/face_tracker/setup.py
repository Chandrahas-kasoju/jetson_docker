import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'face_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chandrahas Kasoju',
    maintainer_email='chandrahas.kasoju@uni-luebeck.de',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker = face_tracker.tracker:main',
            'aspect_ratio_standalone = face_tracker.aspect_ratio_standalone:main',
        ],
    },
)
