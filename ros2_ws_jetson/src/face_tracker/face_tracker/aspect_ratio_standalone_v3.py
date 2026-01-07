#!/usr/bin/env python3

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import os

# --- USE STANDARD MESSAGE ---
from std_msgs.msg import Float32MultiArray

class FaceTrackerNode(Node):
    def __init__(self):
        super().__init__('face_tracker_node')
        self.get_logger().info(f"Starting Face Tracker Node...")
        
        # Constants
        self.IMAGE_TOPIC = "/hospibot/image_raw"
        
        # Define Pose Connections (Standard MediaPipe Pose)
        self.POSE_CONNECTIONS = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
            (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ])

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            self.IMAGE_TOPIC,
            self.image_callback,
            10
        )
        self.bb_pub = self.create_publisher(Float32MultiArray, '/hospibot/pose_bbox', 10)
        
        # MediaPipe Tasks API Setup
        model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        self.get_logger().info(f"Node started. Subscribed to '{self.IMAGE_TOPIC}'")

    def draw_landmarks_custom(self, image, landmarks, connections):
        """Custom landmark drawing function using OpenCV"""
        if not landmarks:
            return

        h, w, _ = image.shape
        
        # Draw connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
                
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            
            cv2.line(image, start_point, end_point, (245, 66, 230), 2) # Magenta

        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (245, 117, 66), -1) # Orange
            cv2.circle(image, (cx, cy), 2, (255, 255, 255), -1) # White center

    def draw_visualizations(
        self, image, landmarks, connections, pose_status, 
        vertical_span, aspect_ratio, spine_angle, side,
        knee_angle, hip_spine_angle,
        thigh_vec_norm, knee_2d,
        hip_length, shoulder_length, torso_ratio  
    ):
        """Draws all landmarks, widgets, and text onto the image"""
        
        # Visual settings
        font_scale = 0.5
        thickness = 1
        y_step = 15
        start_y = 20

        if landmarks:
            self.draw_landmarks_custom(image, landmarks, connections)
        
        # Define colors based on status
        status_color = (0, 255, 0) # Green for Not Lying/No Human
        if pose_status == "Lying Down":
            status_color = (0, 0, 255) # Red for Lying Down

        # --- Draw the status text ---
        cv2.putText(image, f'Status: {pose_status}', (10, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
        
        # --- Draw Metrics (Visible for ALL statuses) ---
        # 1. Angles
        cv2.putText(image, f'Spine Angle: {spine_angle:.1f} deg', (10, start_y + y_step), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 0), thickness)

        cv2.putText(image, f'Knee Angle: {knee_angle:.1f} deg', (10, start_y + y_step*2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 0), thickness)
        
        # 2. Bounding Box
        cv2.putText(image, f'Vert Span: {vertical_span:.3f}', (10, start_y + y_step*3), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 255, 255), thickness)
        
        cv2.putText(image, f'Aspect Ratio: {aspect_ratio:.2f}', (10, start_y + y_step*4), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 255, 255), thickness)

        # 3. Body Proportions (NEW)
        cv2.putText(image, f'S/T Ratio: {torso_ratio:.2f}', (10, start_y + y_step*5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 0, 255), thickness)
        
        # hip spine angle (Ensure this is not None!)
        cv2.putText(image, f'Hip-Spine Ang: {hip_spine_angle:.1f}', (10, start_y + y_step*6), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)

    def get_visible_side(self, sl):
        """Determines which side of the body is more visible"""
        # Map indices manually since PoseLandmark enum might be missing
        LEFT_SHOULDER = 11
        LEFT_HIP = 23
        LEFT_KNEE = 25
        RIGHT_SHOULDER = 12
        RIGHT_HIP = 24
        RIGHT_KNEE = 26

        left_side_visibility = sum(sl[i].visibility for i in [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE])
        right_side_visibility = sum(sl[i].visibility for i in [RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE])
        
        side_used = "Left" if left_side_visibility > right_side_visibility else "Right"
        use_left_side = left_side_visibility > right_side_visibility
        
        return use_left_side, side_used

    def analyze_pose(self, world_landmarks, screen_landmarks):
        """
        Analyzes pose and returns ONLY: "No Human", "Lying Down", or "Not Lying Down".
        Now calculates metrics (Span, Ratio, Spine Angle, Knee Angle, S/T Ratio) for EVERY frame.
        """
        
        # --- Default return values ---
        defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0

        if not world_landmarks or not screen_landmarks: 
            return defaults
            
        wl = world_landmarks
        sl = screen_landmarks
        
        # Indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26

        # --- 1. CALCULATE 2D BODY RATIOS & MIDPOINTS ---
        try:
            ls = sl[LEFT_SHOULDER]
            rs = sl[RIGHT_SHOULDER]
            lh = sl[LEFT_HIP]
            rh = sl[RIGHT_HIP]

            # 1. Shoulder Length (2D)
            shoulder_length = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
            
            # 2. Hip Length (2D)
            hip_length = math.sqrt((lh.x - rh.x)**2 + (lh.y - rh.y)**2)

            # 3. Calculate Midpoints (WE DEFINE THEM HERE TO USE LATER)
            mid_shoulder_x = (ls.x + rs.x) / 2
            mid_shoulder_y = (ls.y + rs.y) / 2
            mid_hip_x = (lh.x + rh.x) / 2
            mid_hip_y = (lh.y + rh.y) / 2
            
            # 4. Torso Height (2D)
            torso_height = math.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
            
            # 5. Ratio
            if torso_height > 0.001:
                torso_ratio = shoulder_length / torso_height
            else:
                torso_ratio = 0.0
            
        except Exception:
            # Fallback if something goes wrong
            shoulder_length, torso_height, torso_ratio, hip_length = 0.0, 0.0, 0.0, 0.0
            mid_shoulder_x, mid_shoulder_y, mid_hip_x, mid_hip_y = 0.0, 0.0, 0.0, 0.0


        # --- 2. PRE-CHECK: ARE KNEES VISIBLE? ---
        l_knee_vis = sl[LEFT_KNEE].visibility
        r_knee_vis = sl[RIGHT_KNEE].visibility
        
        if l_knee_vis < 0.5 and r_knee_vis < 0.5:
            return "No Knee", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0

        # --- 3. CALCULATE 2D METRICS (FOR VISUALIZATION) ---
        visible_landmarks = [lm for lm in sl if lm.visibility > 0.5]
        if not visible_landmarks:
            return defaults

        # A. Bounding Box Metrics
        min_y = min(lm.y for lm in visible_landmarks)
        max_y = max(lm.y for lm in visible_landmarks)
        min_x = min(lm.x for lm in visible_landmarks)
        max_x = max(lm.x for lm in visible_landmarks)
        
        vertical_span = max_y - min_y
        horizontal_span = max_x - min_x
        
        if horizontal_span == 0: horizontal_span = 0.001
        aspect_ratio = vertical_span / horizontal_span 

        # B. Spine Angle Metrics (2D)
        # Ensure variables exist even if try block failed (though unlikely if landmarks exist)
        # We recalculate vec_x/y using the variables defined in Section 1
        
        vec_x = mid_shoulder_x - mid_hip_x
        vec_y = mid_shoulder_y - mid_hip_y 
        
        # Calculate Angle from Vertical (0 = Upright, 90 = Horizontal)
        spine_angle = abs(math.degrees(math.atan2(vec_x, vec_y)))
        if spine_angle > 90:
            spine_angle = 180 - spine_angle

        # C. Knee Angle Metrics (2D)
        use_left_side, side = self.get_visible_side(sl)
        
        if use_left_side:
            hip_lm = sl[LEFT_HIP]
            knee_lm = sl[LEFT_KNEE]
        else:
            hip_lm = sl[RIGHT_HIP]
            knee_lm = sl[RIGHT_KNEE]

        # Vector from Hip to Knee

        thigh_vec_x = knee_lm.x - hip_lm.x
        thigh_vec_y = knee_lm.y - hip_lm.y 
        
        thigh_vec_norm = math.sqrt(thigh_vec_x**2 + thigh_vec_y**2)
        knee_2d = (knee_lm.x, knee_lm.y)

        
        # D. Hip-Spine Angle
        dot_product = (vec_x * thigh_vec_x) + (vec_y * thigh_vec_y)
        mag_spine = math.sqrt(vec_x**2 + vec_y**2)
        mag_hip = math.sqrt(thigh_vec_x**2 + thigh_vec_y**2)
        
        if mag_spine * mag_hip > 0:
            cosine_angle = max(min(dot_product / (mag_spine * mag_hip), 1.0), -1.0)
            hip_spine_angle = math.degrees(math.acos(cosine_angle))
        else:
            hip_spine_angle = 0.0

        # Knee Angle
        knee_angle = abs(math.degrees(math.atan2(thigh_vec_x, thigh_vec_y)))
        if knee_angle > 90:
            knee_angle = 180 - knee_angle

        # --- 4. DETERMINE STATUS ---
        
        ASPECT_RATIO_TALL_THRESHOLD = 2.0  
        ASPECT_RATIO_WIDE_THRESHOLD = 0.6 
        FORESHORTENED_SPAN_THRESHOLD = 0.45
        FORESHORTENED_ASPECT_RATIO_THRESHOLD = 1.2 

        pose_status = "Not Lying Down" 

        if torso_ratio > 0.3:
            if aspect_ratio > ASPECT_RATIO_TALL_THRESHOLD:
                pose_status = "Not Lying Down 1"
            elif aspect_ratio < ASPECT_RATIO_WIDE_THRESHOLD:
                delta_x = abs(vec_x)
                delta_y = abs(vec_y)
                if delta_x > delta_y:
                    pose_status = "Lying Down 2"
                else:
                    pose_status = "Not Lying Down 3"
            elif (vertical_span < FORESHORTENED_SPAN_THRESHOLD and 
                aspect_ratio < FORESHORTENED_ASPECT_RATIO_THRESHOLD) and (thigh_vec_y < thigh_vec_x):
                pose_status = "Lying Down 4"
            else:
                if spine_angle > 60: ## CHECK THIS LATER
                    pose_status = "Lying Down 5"
                else:
                    pose_status = "Not Lying Down 6"


            if pose_status == "Lying Down" and (thigh_vec_y < thigh_vec_x):
                if hip_spine_angle < 160:
                    pose_status = "Not Lying Down 7"
                else:
                    pose_status = "Lying Down 8"

            return (pose_status, vertical_span, aspect_ratio, spine_angle, side,
                    knee_angle, hip_spine_angle,
                    None, None,
                    hip_length, shoulder_length, torso_ratio)

        elif torso_ratio <= 0.3:
            if aspect_ratio > ASPECT_RATIO_TALL_THRESHOLD:
                pose_status = "Not Lying Down 1b"
            elif aspect_ratio < ASPECT_RATIO_WIDE_THRESHOLD:
                delta_x = abs(vec_x)
                delta_y = abs(vec_y)
                if delta_x > delta_y and spine_angle > 45 and knee_angle > 45:
                    pose_status = "Lying Down 2b"
                else:
                    pose_status = "Not Lying Down 3b"
            elif (vertical_span < FORESHORTENED_SPAN_THRESHOLD and 
                aspect_ratio < FORESHORTENED_ASPECT_RATIO_THRESHOLD) and spine_angle > 45 and knee_angle > 45:
                pose_status = "Lying Down 4b"
            else:
                pose_status = "Not Lying Down 5b"
            
            return (pose_status, vertical_span, aspect_ratio, spine_angle, side,
                    knee_angle, hip_spine_angle,
                    None, None,
                    hip_length, shoulder_length, torso_ratio)
        else:
            return "No TORSO Ratio", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, None, 0.0, 0.0, 0.0

    def process_frame_and_publish_bbox(self, landmarks, image, h, w):
        """Calculates bbox from landmarks, publishes ROS msg, draws on image."""
        if not landmarks:
            return

        x_min = min([lm.x for lm in landmarks])
        y_min = min([lm.y for lm in landmarks])
        x_max = max([lm.x for lm in landmarks])
        y_max = max([lm.y for lm in landmarks])
        
        px_min_x = int(x_min * w)
        px_min_y = int(y_min * h)
        px_max_x = int(x_max * w)
        px_max_y = int(y_max * h)

        # --- CREATE STANDARD ROS MESSAGE ---
        msg = Float32MultiArray()
        msg.data = [float(px_min_x), float(px_min_y), float(px_max_x), float(px_max_y)]
        
        self.bb_pub.publish(msg)
        
        # Draw the bounding box
        cv2.rectangle(image, (px_min_x, px_min_y), (px_max_x, px_max_y), (0, 255, 0), 2)

    def image_callback(self, msg):
        """This function is called for every frame from the rosbag."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            try:
                cv_image_mono = self.bridge.imgmsg_to_cv2(msg, "mono8")
                cv_image = cv2.cvtColor(cv_image_mono, cv2.COLOR_GRAY2BGR)
            except Exception as e2:
                self.get_logger().error(f"Failed to convert image (tried bgr8 and mono8): {e2}")
                return

        cv_image.flags.writeable = False
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.detector.detect(mp_image)
        
        cv_image.flags.writeable = True

        # Check if any landmarks were detected
        if detection_result.pose_landmarks:
            # Take the first detected pose
            pose_landmarks = detection_result.pose_landmarks[0]
            pose_world_landmarks = detection_result.pose_world_landmarks[0]
            
            h, w, _ = cv_image.shape
            self.process_frame_and_publish_bbox(pose_landmarks, cv_image, h, w)

            (pose_status, vertical_span, aspect_ratio, spine_angle, side,
             knee_angle, hip_spine_angle,
             thigh_vec_norm, knee_2d,
             hip_length, shoulder_length, torso_ratio) = self.analyze_pose(
                pose_world_landmarks, pose_landmarks
            )
            
            self.draw_visualizations(
                cv_image, 
                pose_landmarks, 
                self.POSE_CONNECTIONS,
                pose_status, 
                vertical_span, 
                aspect_ratio, 
                spine_angle, 
                side,
                knee_angle, hip_spine_angle,
                thigh_vec_norm, knee_2d,
                hip_length, shoulder_length, torso_ratio
            )
        else:
             # Default is "No Human"
            defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, None, 0.0, 0.0, 0.0
            (pose_status, vertical_span, aspect_ratio, spine_angle, side,
             knee_angle, hip_spine_angle,
             thigh_vec_norm, knee_2d,
             hip_length, shoulder_length, torso_ratio) = defaults
             
            self.draw_visualizations(
                cv_image, 
                None, 
                self.POSE_CONNECTIONS,
                pose_status, 
                vertical_span, 
                aspect_ratio, 
                spine_angle, 
                side,
                knee_angle, hip_spine_angle,
                thigh_vec_norm, knee_2d,
                hip_length, shoulder_length, torso_ratio
            )
        
        cv2.imshow('Posture Analyzer (ROS 2)', cv_image)
        
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            self.get_logger().info("Shutdown requested via 'q' or ESC.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()