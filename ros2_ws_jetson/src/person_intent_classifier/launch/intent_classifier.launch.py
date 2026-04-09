from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = FindPackageShare(package='person_intent_classifier').find('person_intent_classifier')
    
    # Default configuration file path
    default_config_file = os.path.join(pkg_share, 'config', 'intent_classifier_params.yaml')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to the YAML configuration file'
        ),
        
        DeclareLaunchArgument(
            'bbox_topic',
            default_value='/person_bounding_box',
            description='Bounding box input topic'
        ),
        
        DeclareLaunchArgument(
            'radar_topic',
            default_value='/person_detect/filtered_points',
            description='Radar point cloud input topic'
        ),
        
        # Person Intent Classifier Node
        Node(
            package='person_intent_classifier',
            executable='intent_node',
            name='person_intent_classifier',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
            remappings=[
                ('/person_intent_classifier/bbox_topic', LaunchConfiguration('bbox_topic')),
                ('/person_intent_classifier/radar_topic', LaunchConfiguration('radar_topic')),
            ]
        ),
        
        # Face Tracker Aspect Ratio Standalone Node
        Node(
            package='face_tracker',
            executable='aspect_ratio_standalone',
            name='aspect_ratio_standalone',
            output='screen',
        ),
    ])