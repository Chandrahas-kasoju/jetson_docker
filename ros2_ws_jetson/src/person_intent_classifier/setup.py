from setuptools import setup

package_name = 'person_intent_classifier'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/intent_classifier_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/intent_classifier.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Classifies person intent (Approach vs Pass) using 2D Bounding Boxes',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'intent_node = person_intent_classifier.intent_node:main',
        ],
    },
)
