import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox2D, Point2D
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from sensor_msgs_py import point_cloud2 as pc2
from collections import deque
import numpy as np
import yaml
import os
from pathlib import Path

class StateStabilizer:
    def __init__(self, persistence_thresh=10):
        self.current_state = "UNKNOWN"
        self.pending_state = None
        self.counter = 0
        self.thresh = persistence_thresh

    def update(self, new_state):
        if new_state == self.current_state:
            self.pending_state = None
            self.counter = 0
            return self.current_state

        if new_state == self.pending_state:
            self.counter += 1
        else:
            self.pending_state = new_state
            self.counter = 1

        if self.counter >= self.thresh:
            self.current_state = new_state
            self.counter = 0
            self.pending_state = None

        return self.current_state

class PersonIntentNode(Node):
    def __init__(self):
        super().__init__('person_intent_classifier')

        # Load configuration from YAML file
        self._load_configuration()

        # --- CONFIGURATION ---
        # Declare ROS2 parameters with defaults
        self.declare_parameter('bbox_topic', self.config['bbox_topic'])
        self.declare_parameter('radar_topic', self.config['radar_topic'])

        # Camera Params
        self.declare_parameter('image_width', self.config['image_width'])
        self.declare_parameter('image_height', self.config['image_height'])

        # Intent Classification Parameters
        self.declare_parameter('close_threshold_ratio', self.config['close_threshold_ratio'])
        self.declare_parameter('interaction_radius', self.config['interaction_radius'])
        self.declare_parameter('radar_timeout', self.config['radar_timeout'])

        # State Stabilization Parameters
        self.declare_parameter('persistence_threshold', self.config['persistence_threshold'])

        # Data Smoothing Parameters
        self.declare_parameter('smoothing_window', self.config['smoothing_window'])
        self.declare_parameter('radar_smoothing_window', self.config['radar_smoothing_window'])

        # Radar Processing Parameters
        self.declare_parameter('low_speed_threshold', self.config['low_speed_threshold'])
        self.declare_parameter('moving_away_threshold', self.config['moving_away_threshold'])
        self.declare_parameter('min_time_delta', self.config['min_time_delta'])

        # Camera Motion Analysis Parameters
        self.declare_parameter('approaching_threshold', self.config['approaching_threshold'])
        self.declare_parameter('moving_away_size_threshold', self.config['moving_away_size_threshold'])
        self.declare_parameter('center_tolerance_ratio', self.config['center_tolerance_ratio'])

        # Get parameter values
        self._setup_parameters()

        # --- STATE VARIABLES ---
        # Camera History
        self.history_size_x = deque(maxlen=self.SMOOTHING_WINDOW)  # Added width history
        self.history_size_y = deque(maxlen=self.SMOOTHING_WINDOW)
        self.history_center_x = deque(maxlen=self.SMOOTHING_WINDOW)
        self.prev_cam_avg_w = None  # Added previous width tracker
        self.prev_cam_avg_h = None
        self.prev_cam_avg_cx = None

        # Eye Center Tracking
        self.latest_eye_center = None

        # Radar State
        self.radar_active = False
        self.last_radar_time = 0

        # Radar Smoothing
        self.radar_centroid_history = deque(maxlen=self.radar_smoothing_window)
        self.prev_radar_centroid = None
        self.prev_radar_timestamp = None

        # Fusion Data
        self.latest_intent = "UNKNOWN"
        self.latest_metrics = ""

        # Person detection tracking
        self.last_bbox_time = 0.0
        self.bbox_timeout = 2.0  # Seconds without bbox before showing "No person detected"

        # Subscribers
        self.create_subscription(BoundingBox2D, self.get_parameter('bbox_topic').value, self.camera_callback, 10)
        self.create_subscription(PointCloud2, self.get_parameter('radar_topic').value, self.radar_callback, 10)
        self.create_subscription(Point2D, '/face_tracker/eye_center', self.eye_center_callback, 10)

        # Publisher
        self.intent_publisher = self.create_publisher(String, '/person_intent', 10)

        # Timer for periodic status checks
        self.create_timer(0.2, self.status_check_callback)

        self.get_logger().info("Intent Node Started [FUSION: Camera + Radar]")
        self.get_logger().info(f"Configuration loaded from YAML file")

    def _load_configuration(self):
        """Load configuration from YAML file or use defaults"""
        # Default configuration
        default_config = {
            'bbox_topic': '/person_bounding_box',
            'radar_topic': '/person_detect/filtered_points',
            'image_width': 256,
            'image_height': 192,
            'close_threshold_ratio': 0.80, # NOTE: Consider lowering this since we now use AREA ratio!
            'interaction_radius': 0.8,
            'radar_timeout': 0.5,
            'persistence_threshold': 5,
            'smoothing_window': 5,
            'radar_smoothing_window': 5,
            'low_speed_threshold': 0.02,
            'moving_away_threshold': 0.1,
            'min_time_delta': 0.05,
            'approaching_threshold': 0.8,
            'moving_away_size_threshold': 0.8,
            'center_tolerance_ratio': 0.25,
        }

        try:
            import ament_index_python
            share_dir = ament_index_python.get_package_share_directory('person_intent_classifier')
            yaml_path = Path(share_dir) / 'config' / 'intent_classifier_params.yaml'
        except:
            yaml_path = Path(__file__).parent.parent / 'config' / 'intent_classifier_params.yaml'

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as file:
                    yaml_data = yaml.safe_load(file)
                    if '/person_intent_classifier' in yaml_data and 'ros__parameters' in yaml_data['/person_intent_classifier']:
                        params = yaml_data['/person_intent_classifier']['ros__parameters']
                        self.config = {**default_config, **params}
                        self.get_logger().info(f"Loaded configuration from {yaml_path}")
                    else:
                        self.config = default_config
                        self.get_logger().warn(f"YAML file found but invalid format, using defaults")
            except Exception as e:
                self.config = default_config
                self.get_logger().warn(f"Failed to load YAML file: {e}, using defaults")
        else:
            self.config = default_config
            self.get_logger().info(f"No YAML file found at {yaml_path}, using defaults")

    def _setup_parameters(self):
        """Setup all parameters from ROS2 parameter server"""
        # Camera parameters
        self.IMAGE_WIDTH = float(self.get_parameter('image_width').value)
        self.IMAGE_HEIGHT = float(self.get_parameter('image_height').value)
        self.IMAGE_AREA = self.IMAGE_WIDTH * self.IMAGE_HEIGHT # Total pixels in image
        self.IMAGE_CENTER_X = self.IMAGE_WIDTH / 2.0

        # Classification thresholds
        self.CLOSE_THRESH_RATIO = self.get_parameter('close_threshold_ratio').value
        self.INTERACTION_RADIUS = self.get_parameter('interaction_radius').value
        self.RADAR_TIMEOUT = self.get_parameter('radar_timeout').value

        # Smoothing parameters
        self.SMOOTHING_WINDOW = int(self.get_parameter('smoothing_window').value)
        self.radar_smoothing_window = int(self.get_parameter('radar_smoothing_window').value)

        # Radar processing parameters
        self.LOW_SPEED_THRESHOLD = self.get_parameter('low_speed_threshold').value
        self.MOVING_AWAY_THRESHOLD = self.get_parameter('moving_away_threshold').value
        self.MIN_TIME_DELTA = self.get_parameter('min_time_delta').value

        # Camera motion analysis
        self.APPROACHING_THRESHOLD = self.get_parameter('approaching_threshold').value
        self.MOVING_AWAY_SIZE_THRESHOLD = self.get_parameter('moving_away_size_threshold').value
        self.CENTER_TOLERANCE_RATIO = self.get_parameter('center_tolerance_ratio').value

        # Initialize stabilizer
        self.stabilizer = StateStabilizer(persistence_thresh=int(self.get_parameter('persistence_threshold').value))

    # =========================================
    # EYE CENTER LOGIC
    # =========================================
    def eye_center_callback(self, msg):
        """Stores the latest eye center to be used if BBOX is out of bounds"""
        self.latest_eye_center = msg

    # =========================================
    # RADAR LOGIC
    # =========================================
    def radar_callback(self, msg):
        # 1. Read points
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_list = list(gen)

        if len(points_list) == 0:
            return

        # 2. Get Raw Centroid
        points = np.array([list(p) for p in points_list], dtype=np.float32)
        raw_centroid = np.mean(points, axis=0)

        # 3. SMOOTH THE CENTROID
        self.radar_centroid_history.append(raw_centroid)
        centroid = np.mean(self.radar_centroid_history, axis=0)

        current_time = self.get_clock().now().nanoseconds / 1e9
        self.radar_active = True
        self.last_radar_time = current_time

        # 4. Calculate Velocity
        if self.prev_radar_centroid is None:
            self.prev_radar_centroid = centroid
            self.prev_radar_timestamp = current_time
            return

        dt = current_time - self.prev_radar_timestamp
        if dt < self.MIN_TIME_DELTA:
            return

        velocity = (centroid - self.prev_radar_centroid) / dt
        speed = np.linalg.norm(velocity)

        # 5. PHYSICS LOGIC
        P = centroid
        V = velocity
        v_sq = np.dot(V, V)
        dist = np.linalg.norm(P)

        intent = "UNKNOWN"
        metrics = ""

        moving_away_score = np.dot(P, V)

        if v_sq < self.LOW_SPEED_THRESHOLD:
            if dist < 0.5:
                  intent = "WANT_TO_INTERACT"
            else:
                 intent = "STATIONARY"
            metrics = f"Dist: {dist:.1f}m"
        elif moving_away_score > self.MOVING_AWAY_THRESHOLD:
            intent = "MOVING_AWAY"
            metrics = f"Vel: {speed:.1f}m/s (Away)"
        else:
            t_c = -np.dot(P, V) / v_sq
            P_closest = P + (V * t_c)
            dca = np.linalg.norm(P_closest)

            if dca < self.INTERACTION_RADIUS:
                intent = "CLOSE_PROXIMITY"
                metrics = f"DCA: {dca:.2f}m (Hit)"
            else:
                intent = "PASSING_BY"
                metrics = f"DCA: {dca:.2f}m (Miss)"

        self.latest_intent = intent
        self.latest_metrics = f"[RADAR] {metrics}"

        self.prev_radar_centroid = centroid
        self.prev_radar_timestamp = current_time

    # =========================================
    # CAMERA LOGIC (Decision Master)
    # =========================================
    def camera_callback(self, msg):
        self.last_bbox_time = self.get_clock().now().nanoseconds / 1e9

        now = self.get_clock().now().nanoseconds / 1e9
        is_radar_fresh = (now - self.last_radar_time) < self.RADAR_TIMEOUT

        # Get values from message
        bbox_cx = msg.center.position.x
        bbox_cy = msg.center.position.y
        bbox_w = msg.size_x
        bbox_h = msg.size_y

        # --- OUT OF BOUNDS OVERRIDE LOGIC ---
        if bbox_w > self.IMAGE_WIDTH or bbox_h > self.IMAGE_HEIGHT:
            if self.latest_eye_center is not None:
                bbox_cx = self.latest_eye_center.x
                bbox_cy = self.latest_eye_center.y
            
            bbox_w = min(bbox_w, self.IMAGE_WIDTH)
            bbox_h = min(bbox_h, self.IMAGE_HEIGHT)

        # --- 1. Process Camera Data ---
        self.history_size_x.append(bbox_w)
        self.history_size_y.append(bbox_h)
        self.history_center_x.append(bbox_cx)

        if len(self.history_size_y) < self.SMOOTHING_WINDOW:
            return

        curr_w = np.mean(self.history_size_x)
        curr_h = np.mean(self.history_size_y)
        curr_cx = np.mean(self.history_center_x)

        if self.prev_cam_avg_h is None:
            self.prev_cam_avg_w = curr_w
            self.prev_cam_avg_h = curr_h
            self.prev_cam_avg_cx = curr_cx
            return

        # --- 2. Decide Source ---
        final_intent = "UNKNOWN"
        debug_str = ""

        # Priority: Close Proximity (NOW USING AREA RATIO)
        curr_area = curr_w * curr_h
        area_ratio = curr_area / self.IMAGE_AREA

        if area_ratio > self.CLOSE_THRESH_RATIO:
            final_intent = "WANT_TO_INTERACT"
            debug_str = f"Visual Priority ({area_ratio*100:.0f}% Area)"

        elif is_radar_fresh and self.radar_active:
            # Trust Radar
            final_intent = self.latest_intent
            debug_str = self.latest_metrics

        else:
            # Fallback Camera (still relying on height delta for simple approaching/retreating detection)
            delta_h = curr_h - self.prev_cam_avg_h

            if delta_h > self.APPROACHING_THRESHOLD:
                if abs(curr_cx - self.IMAGE_CENTER_X) < (self.IMAGE_WIDTH * self.CENTER_TOLERANCE_RATIO):
                    final_intent = "CLOSE_PROXIMITY"
                else:
                    final_intent = "PASSING_BY"
            elif delta_h < -self.MOVING_AWAY_SIZE_THRESHOLD:
                final_intent = "MOVING_AWAY"
            else:
                final_intent = "STATIONARY"

            debug_str = "[CAM-ONLY]"

        # --- 3. Stabilize & Output ---
        stable_intent = self.stabilizer.update(final_intent)

        self.prev_cam_avg_w = curr_w
        self.prev_cam_avg_h = curr_h
        self.prev_cam_avg_cx = curr_cx

        # Use ROS logging for better visibility
        self.get_logger().info(f"Intent: {stable_intent} | {debug_str}")
        
        # Publish intent to ROS2 topic
        intent_msg = String()
        intent_msg.data = stable_intent
        self.intent_publisher.publish(intent_msg)

    # =========================================
    # STATUS CHECK (No Person Detection)
    # =========================================
    def status_check_callback(self):
        """Periodic check for person detection status"""
        now = self.get_clock().now().nanoseconds / 1e9

        if (now - self.last_bbox_time) > self.bbox_timeout:
            self.get_logger().info("Intent: NO_PERSON_DETECTED | No bounding box received")
            
            intent_msg = String()
            intent_msg.data = "NO_PERSON_DETECTED"
            self.intent_publisher.publish(intent_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PersonIntentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()