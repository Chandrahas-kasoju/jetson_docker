#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from posture_analysis_msgs.msg import Posture
from vision_msgs.msg import BoundingBox2D, Pose2D, Point2D
import requests
import time






class MediaPipeTrackerNode(Node):
    def __init__(self):
        super().__init__('mediapipe_tracker_node')
        
        # --- Parameters ---
        self.declare_parameter('image_topic', 'hospibot/image_raw')
        self.declare_parameter('pitch_topic', '/servo_command')
        self.declare_parameter('roll_topic', '/stepper_panther')
        self.declare_parameter('dead_zone_percent', 30)
        
        # --- MediaPipe Specific Parameters ---
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('visibility_threshold', 0.5)
        
        # --- PERFORMANCE OPTIMIZATION PARAMETERS FOR RASPBERRY PI ---
        self.declare_parameter('model_complexity', 1)
        self.declare_parameter('enable_visualization', True)

        # --- Publishers and Subscribers ---
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        pitch_topic = self.get_parameter('pitch_topic').get_parameter_value().string_value
        roll_topic = self.get_parameter('roll_topic').get_parameter_value().string_value
        
        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.info_subscription = self.create_subscription(CameraInfo, 'hospibot/camera_info', self.camera_info_callback, 10)
        self.pitch_publisher = self.create_publisher(Int32, pitch_topic, 10)
        self.roll_publisher = self.create_publisher(Int32, roll_topic, 10)
        self.eye_center_pub = self.create_publisher(Point2D, '/face_tracker/eye_center', 10)
        self.publisher_ = self.create_publisher(Posture, 'human_posture', 10)
        self.bb_pub = self.create_publisher(BoundingBox2D, 'person_bounding_box', 10)
        self.image_pub = self.create_publisher(Image, 'bb_box/Image_raw', 10)
        self.image_info_pub = self.create_publisher(CameraInfo, 'bb_box/camera_info', 10)
        
        self.bridge = CvBridge()
        self.center_pose = Pose2D()
        self.bbox_msg = BoundingBox2D()
        
        # --- MediaPipe Model Setup ---
        self.get_logger().info("Loading MediaPipe Pose model...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        min_detection_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        model_complexity = self.get_parameter('model_complexity').get_parameter_value().integer_value
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity)
        
        self.get_logger().info(f'MediaPipe tracker node started. Model Complexity: {model_complexity}, Visualization: {self.enable_visualization}')

        # --- ALARM LOGIC ---
        self.NTFY_TOPIC = "hospibot_lying_down_alarm" 
        self.notification_cooldown = 30.0 
        self.last_notification_time = 0.0

        # --- Multiple Frame Debouncing ---
        self.lying_down_frame_count = 0
        self.LYING_DOWN_FRAME_TRIGGER = 20 # Requires 15 consecutive frames of lying down

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        height, width, _ = cv_image.shape
        
        cv_image.flags.writeable = False
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        cv_image.flags.writeable = True

        pitch_cmd = Int32()
        roll_cmd = Int32()
        
        # Default State: No target detected. Command is 2 (Go to Home).
        pitch_cmd.data = 3
        roll_cmd.data = 3
        
        dead_zone_percent = self.get_parameter('dead_zone_percent').get_parameter_value().integer_value
        dead_zone_x = (width * dead_zone_percent) // 200
        dead_zone_y = (height * dead_zone_percent) // 200
        center_x = width // 2
        center_y = height // 2
        
        # Check if a pose was detected at all
        pose_status, uprightness, spine_vec, hip_angle, side, hip_knee_dist = "Unknown", 0.0, None, 0.0, "", 0.0
        if results.pose_landmarks:
            pose_status, uprightness, spine_vec, hip_angle, side, hip_knee_dist = self.analyze_pose(results.pose_world_landmarks, results.pose_landmarks)
            # Create and publish the ROS message
            posture_msg = Posture()
            posture_msg.posture_class = posture_msg.UNKNOWN # Default
            if pose_status == "Lying Down":
                posture_msg.posture_class = posture_msg.LYING_DOWN
            elif pose_status == "Sitting":
                posture_msg.posture_class = posture_msg.SITTING
            elif pose_status == "Standing":
                posture_msg.posture_class = posture_msg.STANDING
            elif pose_status == "Upright":
                posture_msg.posture_class = posture_msg.UPRIGHT
            
            posture_msg.uprightness_score = int(uprightness * 100)
            self.publisher_.publish(posture_msg)
            
            #send alert if lying down, loggers are not necessary but might be good for testing and debugging..
            if posture_msg.posture_class == Posture.LYING_DOWN:
                self.lying_down_frame_count += 1
                self.get_logger().info(f'Possible Lying Down... Count: {self.lying_down_frame_count}/{self.LYING_DOWN_FRAME_TRIGGER}', throttle_duration_sec=1)
            else:
                if self.lying_down_frame_count > 0:
                    self.get_logger().info('Lying down state was temporary. Resetting counter.')
                self.lying_down_frame_count = 0
            
            if self.lying_down_frame_count >= self.LYING_DOWN_FRAME_TRIGGER:
                self.send_notification()
                self.lying_down_frame_count = 0 # Reset after sending
            

            landmarks = results.pose_landmarks.landmark
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
            
            visibility_threshold = self.get_parameter('visibility_threshold').get_parameter_value().double_value

            # Check if both eyes are clearly visible for tracking
            if left_eye.visibility > visibility_threshold and right_eye.visibility > visibility_threshold:
                # State: Actively tracking. Commands are -1, 0, or 1.
                pitch_cmd.data = 0 # Set to 0 initially for the dead zone case
                roll_cmd.data = 0

                left_eye_x = int(left_eye.x * width)
                left_eye_y = int(left_eye.y * height)
                right_eye_x = int(right_eye.x * width)
                right_eye_y = int(right_eye.y * height)
                
                target_x = (left_eye_x + right_eye_x) // 2
                target_y = (left_eye_y + right_eye_y) // 2

                # Publish Eye Center
                eye_center_msg = Point2D()
                eye_center_msg.x = float(target_x)
                eye_center_msg.y = float(target_y)
                self.eye_center_pub.publish(eye_center_msg)

                #if target_x < center_x - dead_zone_x: roll_cmd.data = 1
                #elif target_x > center_x + dead_zone_x: roll_cmd.data = -1
                
                #if target_y < center_y - dead_zone_y: pitch_cmd.data = 1
                #elif target_y > center_y + dead_zone_y: pitch_cmd.data = -1

                if self.enable_visualization:
                    cv2.circle(cv_image, (target_x, target_y), 7, (255, 0, 0), -1) # Blue for tracking
            else:
                # State: Target present but obscured. Command is 3 (Hold Position).
                pitch_cmd.data = 3
                roll_cmd.data = 3
                self.get_logger().info('Pose detected, but eyes not visible. Publishing hold command (3).', throttle_duration_sec=2)

            if self.enable_visualization:
                self.mp_drawing.draw_landmarks(
                    cv_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.process_frame_and_publish_bbox(landmarks, cv_image, height, width)
                #self.process_frame_and_publish_bbox(landmarks, cv_image, height, width)
                
        
        else:
            # This 'else' corresponds to 'if results.pose_landmarks:'
            # State: No target detected. The command remains the default of 2.
            self.get_logger().info('No target detected. Publishing go to home command (2).', throttle_duration_sec=2)

        # self.pitch_publisher.publish(pitch_cmd)
        # self.roll_publisher.publish(roll_cmd)

        if self.enable_visualization:
            #cv2.rectangle(cv_image, (center_x - dead_zone_x, center_y - dead_zone_y), 
            #              (center_x + dead_zone_x, center_y + dead_zone_y), (0, 0, 255), 2)
            # cv2.imshow("MediaPipe Tracker", cv_image)
            # cv2.waitKey(1)
        
            cv2.putText(cv_image, f'Status: {pose_status}', (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv_image = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            cv_image.header.stamp = self.get_clock().now().to_msg()
            cv_image.header.frame_id = "thermal_camera_link"
            self.image_pub.publish(cv_image)
          
            
            
    def camera_info_callback(self, msg):
        # Directly republish the received CameraInfo message
        self.image_info_pub.publish(msg)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def analyze_pose(self, world_landmarks, screen_landmarks):
        if not world_landmarks or not screen_landmarks:
            return "Unknown", 0.0, None, 0.0, "", 0.0

        wl = world_landmarks.landmark
        sl = screen_landmarks.landmark
        p = self.mp_pose.PoseLandmark

        left_hip_3d = np.array([wl[p.LEFT_HIP].x, wl[p.LEFT_HIP].y, wl[p.LEFT_HIP].z])
        right_hip_3d = np.array([wl[p.RIGHT_HIP].x, wl[p.RIGHT_HIP].y, wl[p.RIGHT_HIP].z])
        left_shoulder_3d = np.array([wl[p.LEFT_SHOULDER].x, wl[p.LEFT_SHOULDER].y, wl[p.LEFT_SHOULDER].z])
        right_shoulder_3d = np.array([wl[p.RIGHT_SHOULDER].x, wl[p.RIGHT_SHOULDER].y, wl[p.RIGHT_SHOULDER].z])

        hip_center = (left_hip_3d + right_hip_3d) / 2
        shoulder_center = (left_shoulder_3d + right_shoulder_3d) / 2
        spine_vector = shoulder_center - hip_center

        norm_spine = np.linalg.norm(spine_vector)
        if norm_spine == 0: return "Unknown", 0.0, None, 0.0, "", 0.0

        spine_vector_normalized = spine_vector / norm_spine
        uprightness_score = abs(spine_vector_normalized[1])

        if uprightness_score < 0.6:
            return "Lying Down", uprightness_score, spine_vector_normalized, 0.0, "", 0.0

        hip_angle, side_used, vertical_hip_knee_dist = 0.0, "", 0.0

        left_side_visibility = sum(sl[lm.value].visibility for lm in [p.LEFT_SHOULDER, p.LEFT_HIP, p.LEFT_KNEE])
        right_side_visibility = sum(sl[lm.value].visibility for lm in [p.RIGHT_SHOULDER, p.RIGHT_HIP, p.RIGHT_KNEE])

        use_left_side = left_side_visibility > right_side_visibility

        side_landmarks_visible = (use_left_side and left_side_visibility > 2.1) or (not use_left_side and right_side_visibility > 2.1)

        if not side_landmarks_visible:
            return "Unknown", uprightness_score, spine_vector_normalized, 0.0, "", 0.0

        if use_left_side:
            shoulder_2d = [sl[p.LEFT_SHOULDER.value].x, sl[p.LEFT_SHOULDER.value].y]
            hip_2d = [sl[p.LEFT_HIP.value].x, sl[p.LEFT_HIP.value].y]
            knee_2d = [sl[p.LEFT_KNEE.value].x, sl[p.LEFT_KNEE.value].y]
            side_used = "Left"
        else:
            shoulder_2d = [sl[p.RIGHT_SHOULDER.value].x, sl[p.RIGHT_SHOULDER.value].y]
            hip_2d = [sl[p.RIGHT_HIP.value].x, sl[p.RIGHT_HIP.value].y]
            knee_2d = [sl[p.RIGHT_KNEE.value].x, sl[p.RIGHT_KNEE.value].y]
            side_used = "Right"

        hip_angle = self.calculate_angle(shoulder_2d, hip_2d, knee_2d)
        vertical_hip_knee_dist = abs(hip_2d[1] - knee_2d[1])

        if hip_angle < 130 or vertical_hip_knee_dist < 0.1:
            return "Sitting", uprightness_score, spine_vector_normalized, hip_angle, side_used, vertical_hip_knee_dist
        elif hip_angle > 150:
            return "Standing", uprightness_score, spine_vector_normalized, hip_angle, side_used, vertical_hip_knee_dist

        return "Upright", uprightness_score, spine_vector_normalized, hip_angle, side_used, vertical_hip_knee_dist
    
    def process_frame_and_publish_bbox(self, landmarks, image, h, w):
        x_min = min([lm.x for lm in landmarks])
        y_min = min([lm.y for lm in landmarks])
        x_max = max([lm.x for lm in landmarks])
        y_max = max([lm.y for lm in landmarks])
        # 2. CREATE AN INSTANCE OF THE Pose2D MESSAGE
        center_point = Point2D()
        center_pose = Pose2D()
        center_point.x = ((x_min + x_max) / 2) * w
        center_point.y = ((y_min + y_max) / 2) * h
        center_pose.position = center_point
        center_pose.theta = 0.0
        # 'theta' is the third field of Pose2D, representing rotation.
        # We can leave it as the default 0.0 for a non-rotated bounding box
        # 3. CREATE THE BoundingBox2D MESSAGE AND ASSIGN THE FIELDS
        bbox_msg = BoundingBox2D()
        bbox_msg.center = center_pose # Assign the Pose2D object
        bbox_msg.size_x = (x_max - x_min) * w
        bbox_msg.size_y = (y_max - y_min) * h
        self.bb_pub.publish(bbox_msg)
        # Optional: Draw the bounding box and landmarks on the image for visualization
        cv2.rectangle(image, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (0, 255, 0), 2)

    def send_notification(self):
        """Sends a push notification via ntfy if cooldown has passed."""
        current_time = time.time()
        if (current_time - self.last_notification_time) > self.notification_cooldown:
            self.get_logger().warn('LYING DOWN DETECTED! Sending ntfy notification...')
            try:
                
                #sound_url = "https://drive.google.com/uc?export=download&id=12zvG0EZ5uab9bDT_Z_B9i83zCWdeAztg"

                requests.post(
                    f"https://ntfy.sh/{self.NTFY_TOPIC}",
                    data="Fall detected! A person is lying down.".encode(encoding='utf-8'),
                    headers={
                        "Title": "TurtleBot Alert: Possible Fall Detected",
                        "Priority": "urgent",
                        "Tags": "warning,rotating_light",
                        #"Attach": sound_url,
                        #"Actions": f"view, Play Alarm Sound, {sound_url}"
                    })
                self.last_notification_time = current_time
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Could not send notification: {e}")
        else:
            self.get_logger().info('Lying down detected, but in notification cooldown period.')
    



def main(args=None):
    rclpy.init(args=args)
    mp_tracker_node = MediaPipeTrackerNode()
    try:
        rclpy.spin(mp_tracker_node)
    except KeyboardInterrupt:
        pass
    finally:
        mp_tracker_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()