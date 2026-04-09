# Person Intent Classifier

A ROS2 package for classifying human intent using multi-sensor fusion of camera and radar data. The system analyzes person behavior to determine intent such as wanting to interact, passing by, moving away, or being in close proximity.


## Features

🤖 **Multi-Sensor Fusion**: Combines camera bounding boxes with radar point clouds for robust intent classification

🎯 **Intent Classification**: Detects multiple human behavioral states:
- `CLOSE_PROXIMITY` - Person is very close to the system
- `WANT_TO_INTERACT` - Person is approaching with interaction intent
- `PASSING_BY` - Person is moving past the system
- `MOVING_AWAY` - Person is moving away from the system
- `STATIONARY` - Person is not moving
- `NO_PERSON_DETECTED` - No person detected in the scene

🛡️ **State Stabilization**: Prevents flickering with configurable persistence thresholds

📊 **Data Smoothing**: Reduces noise using moving averages and centroid smoothing

⚙️ **Highly Configurable**: Extensive parameter system for fine-tuning behavior

## System Requirements

- **ROS2**: Humble (tested on ROS2 Humble)
- **Python**: 3.8 or higher
- **Dependencies**:
  - `rclpy` (ROS2 Python client library)
  - `vision_msgs` (ROS2 vision messages)
  - `sensor_msgs` (ROS2 sensor messages)
  - `sensor_msgs_py`(for efficient PointCloud2 processing)
  - `numpy` (numerical computing)
  - `yaml` (configuration parsing)

## Installation

### From Source

1. Clone this repository into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/Manohar174/person_intent_classifier
```

2. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src/person_intent_classifier --ignore-src
sudo apt install ros-humble-sensor-msgs-py
```

3. Build the package:
```bash
colcon build --packages-select person_intent_classifier
source install/setup.bash
```



## Quick Start

### Basic Usage

**Start the Intent Classifier**:
```bash
ros2 launch person_intent_classifier intent_classifier.launch.py
```

or 

```bash
ros2 run person_intent_classifier intent_classifier_node
```

**Monitor Intent Output**:
```bash
ros2 topic echo /person_intent
```

### With Custom Configuration

```bash
ros2 launch person_intent_classifier intent_classifier.launch.py \
  config_file:=/path/to/custom_config.yaml \
  bbox_topic:=/custom/bbox_topic \
  radar_topic:=/custom/radar_topic
```



## API Reference

### ROS Topics

| Type | Topic | Message | Description |
| :--- | :--- | :--- | :--- |
| **Sub** | `/person_bounding_box` | `vision_msgs/BoundingBox2D` | Person detection |
| **Sub** | `/person_detect/filtered_points` | `sensor_msgs/PointCloud2` | Radar detections (x,y,z) |
| **Pub** | `/person_intent` | `std_msgs/String` | Intent classification result |


### Input Topics

#### `/person_bounding_box` (vision_msgs/msg/BoundingBox2D)
Bounding box data from person detection system containing:
- `center.x`, `center.y`: Center coordinates of bounding box
- `size_x`, `size_y`: Width and height of bounding box

#### `/person_detect/filtered_points` (sensor_msgs/msg/PointCloud2)
Filtered point cloud data from radar containing 3D coordinates (x, y, z) for detected person.

### Output Topics

#### `/person_intent` (std_msgs/msg/String)
Published intent classificatio with values:
- `"CLOSE_PROXIMITY"`: Person very close to system. Person size > 70% of image height
- `"WANT_TO_INTERACT"`: Person approaching with interaction intent
- `"PASSING_BY"`: Person moving past the system
- `"MOVING_AWAY"`: Person moving away from system
- `"STATIONARY"`: Person not moving. Low speed movement with distance > 1.0m
- `"NO_PERSON_DETECTED"`: No person detected

### Parameters

All configuration parameters are documented in `config/intent_classifier_params.yaml` with detailed descriptions and recommended ranges.

## Configuration

The system is highly configurable through YAML parameter files. The default configuration is located in `config/intent_classifier_params.yaml`.

### Key Parameters

#### Intent Classification
- `close_threshold_ratio` (0.70): Size ratio threshold for close proximity detection
- `interaction_radius` (0.8): Distance threshold for interaction intent detection
- `radar_timeout` (0.5): Maximum age of radar data to consider valid

#### Camera Analysis
- `approaching_threshold` (0.8): Minimum size increase to detect approaching behavior
- `moving_away_size_threshold` (0.8): Minimum size decrease to detect moving away
- `center_tolerance_ratio` (0.25): Tolerance for center position analysis

#### Radar Processing
- `low_speed_threshold` (0.02): Minimum speed to consider movement
- `moving_away_threshold` (0.1): Threshold for detecting moving away behavior
- `radar_smoothing_window` (5): Number of radar measurements to smooth

#### State Management
- `persistence_threshold` (10): Consecutive detections required for state change
- `smoothing_window` (10): Number of measurements for data smoothing

### Example Custom Configuration

```yaml
/person_intent_classifier:
  ros__parameters:
    # More sensitive to approaching behavior
    approaching_threshold: 0.5
    moving_away_size_threshold: 0.5
    
    # Larger interaction zone
    interaction_radius: 1.2
    
    # Faster state transitions
    persistence_threshold: 5
    
    # Custom topic names
    bbox_topic: '/my_person_detection/bbox'
    radar_topic: '/my_radar/person_points'
```

## Architecture

### Sensor Fusion Strategy

1. **Camera Priority**: Close proximity detection based on person size in image
2. **Radar Trust**: When radar data is fresh, trust radar for motion analysis
3. **Camera Fallback**: When radar data is unavailable, use camera-based motion analysis
4. **State Stabilization**: Apply persistence threshold to prevent state flickering

### Processing Pipeline

```
Camera Bounding Box ─┐
                    ├──→ Fusion Logic → State Stabilizer → Intent Output
Radar Point Cloud ──┘                    ↓
                              /person_intent topic
```




## Troubleshooting

### Common Issues

1. **No Intent Output**
   - Check that input topics are receiving data
   - Verify topic names match configuration
   - Ensure bounding box data is in expected format

2. **Flickering States**
   - Increase `persistence_threshold` parameter
   - Increase `smoothing_window` for more stable data
   - Adjust `radar_timeout` for radar data handling

3. **Inaccurate Classification**
   - Tune `close_threshold_ratio` based on camera setup
   - Adjust `interaction_radius` for your use case
   - Calibrate `approaching_threshold` and `moving_away_size_threshold`

4. **Radar Data Issues**
   - Verify radar point cloud format (x, y, z fields)
   - Check radar calibration and coordinate frame alignment
   - Adjust `radar_smoothing_window` for noisy data

### Debugging

Enable debug logging:
```bash
ros2 launch person_intent_classifier intent_classifier.launch.py \
  params_file:=$(find person_intent_classifier)/config/intent_classifier_params.yaml \
  :=_log_level:=DEBUG
```

Monitor intent changes:
```bash
ros2 topic echo /person_intent --no-arr
```

Check parameter values:
```bash
ros2 param get /person_intent_classifier close_threshold_ratio
```

## Performance Considerations

- **Latency**: Typical processing latency < 50ms
- **CPU Usage**: Optimized for real-time operation on embedded systems
- **Memory**: Low memory footprint with configurable history buffers
- **Frequency**: Configurable update rates up to 20Hz


---

