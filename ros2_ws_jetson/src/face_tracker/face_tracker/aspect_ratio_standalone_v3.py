#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import depthai as dai 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math

# --- USE STANDARD MESSAGE ---
from std_msgs.msg import Float32MultiArray

USE_ROSBAG = True
USE_LAPTOP_CAM = False

ROSBAG_IMAGE_TOPIC = "/hospibot/image_raw" 

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def draw_visualizations(
    image, landmarks, connections, pose_status, 
    vertical_span, aspect_ratio, spine_angle, side,
    knee_angle, hip_spine_angle,
    thigh_vec_norm, knee_2d,
    hip_length, shoulder_length, torso_ratio  
):
    """Draws all landmarks, widgets, and text onto the image"""
    
    # Visual settings based on input source
    if USE_ROSBAG:
        font_scale = 0.5
        thickness = 1
        y_step = 15
        start_y = 20
    else:
        font_scale = 1.0
        thickness = 2
        y_step = 40
        start_y = 40

    landmark_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=thickness, circle_radius=2)
    connection_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=thickness, circle_radius=2)

    if landmarks:
        mp_drawing.draw_landmarks(
            image, landmarks, connections,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec
        )
    
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


def get_visible_side(sl):
    """Determines which side of the body is more visible"""
    p = mp_pose.PoseLandmark
    left_side_visibility = sum(sl[lm.value].visibility for lm in [p.LEFT_SHOULDER, p.LEFT_HIP, p.LEFT_KNEE])
    right_side_visibility = sum(sl[lm.value].visibility for lm in [p.RIGHT_SHOULDER, p.RIGHT_HIP, p.RIGHT_KNEE])
    
    side_used = "Left" if left_side_visibility > right_side_visibility else "Right"
    use_left_side = left_side_visibility > right_side_visibility
    
    return use_left_side, side_used


def analyze_pose(world_landmarks, screen_landmarks):
    """
    Analyzes pose and returns ONLY: "No Human", "Lying Down", or "Not Lying Down".
    Now calculates metrics (Span, Ratio, Spine Angle, Knee Angle, S/T Ratio) for EVERY frame.
    """
    
    # --- Default return values ---
    defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0

    if not world_landmarks or not screen_landmarks: 
        return defaults
        
    wl = world_landmarks.landmark
    sl = screen_landmarks.landmark
    p = mp_pose.PoseLandmark

    # --- 1. CALCULATE 2D BODY RATIOS & MIDPOINTS ---
    try:
        ls = sl[p.LEFT_SHOULDER.value]
        rs = sl[p.RIGHT_SHOULDER.value]
        lh = sl[p.LEFT_HIP.value]
        rh = sl[p.RIGHT_HIP.value]

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
    l_knee_vis = sl[p.LEFT_KNEE.value].visibility
    r_knee_vis = sl[p.RIGHT_KNEE.value].visibility
    
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
    use_left_side, side = get_visible_side(sl)
    
    if use_left_side:
        hip_lm = sl[p.LEFT_HIP.value]
        knee_lm = sl[p.LEFT_KNEE.value]
    else:
        hip_lm = sl[p.RIGHT_HIP.value]
        knee_lm = sl[p.RIGHT_KNEE.value]

    # Vector from Hip to Knee



    thigh_vec_x = knee_lm.x - hip_lm.x
    thigh_vec_y = knee_lm.y - hip_lm.y 

    
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


class RosbagPoseReader(Node):
    def __init__(self, pose_model):
        super().__init__('pose_analyzer_node')
        self.get_logger().info(f"Starting Pose Analyzer Node...")
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            ROSBAG_IMAGE_TOPIC,
            self.image_callback,
            10
        )
        self.bb_pub = self.create_publisher(Float32MultiArray, '/hospibot/pose_bbox', 10)
        
        self.pose = pose_model
        
        self.get_logger().info(f"Node started. Subscribed to '{ROSBAG_IMAGE_TOPIC}'")
        self.get_logger().info("Now, play your rosbag in another terminal!")

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
        results = self.pose.process(image_rgb)
        cv_image.flags.writeable = True

        if results.pose_landmarks:
            h, w, _ = cv_image.shape
            self.process_frame_and_publish_bbox(results.pose_landmarks.landmark, cv_image, h, w)

        # Default is "No Human"
        defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, None, 0.0, 0.0, 0.0
        (pose_status, vertical_span, aspect_ratio, spine_angle, side,
         knee_angle, hip_spine_angle,
         thigh_vec_norm, knee_2d,
         hip_length, shoulder_length, torso_ratio) = defaults
        
        if results.pose_landmarks:
            (pose_status, vertical_span, aspect_ratio, spine_angle, side,
             knee_angle, hip_spine_angle,
             thigh_vec_norm, knee_2d,
             hip_length, shoulder_length, torso_ratio) = analyze_pose(
                results.pose_world_landmarks, results.pose_landmarks
            )

        draw_visualizations(
            cv_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
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


def main():
    # Params
    show_video = True 

    # MediaPipe Pose Model
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    if USE_ROSBAG:
        rclpy.init()
        pose_analyzer_node = RosbagPoseReader(pose_model=pose)
        try:
            rclpy.spin(pose_analyzer_node)
        except KeyboardInterrupt:
            print("Keyboard interrupt, shutting down.")
        finally:
            pose_analyzer_node.destroy_node()
            cv2.destroyAllWindows()
            
    elif USE_LAPTOP_CAM:
        print("Setting up Laptop camera (cv2.VideoCapture)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open laptop camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Laptop camera setup successful.")
        
        main_loop(cap, pose, show_video, is_laptop=True)
        
    else:
        print("Setting up OAK-D Pro camera...")
        oak_device, q_rgb = setup_oakd()
        if oak_device is None:
            return
        print("OAK-D camera setup successful.")
        
        main_loop(q_rgb, pose, show_video, is_laptop=False, oak_device=oak_device)

    print("Script finished.")


def setup_oakd():
    try:
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewSize(1280, 720)
        cam_rgb.setVideoSize(1280, 720)
        cam_rgb.setFps(30)
        cam_rgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        cam_rgb.video.link(xout_rgb.input)
        oak_device = dai.Device(pipeline)
        q_rgb = oak_device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        return oak_device, q_rgb
    except Exception as e:
        print(f"Failed to initialize OAK-D camera: {e}")
        return None, None


def main_loop(camera_source, pose, show_video, is_laptop=True, oak_device=None):
    print('Posture Estimator Standalone has started.')
    print("Press 'q' or ESC to quit.")
    try:
        while True:
            image = None
            if is_laptop:
                success, frame = camera_source.read()
                if not success:
                    print("Ignoring empty laptop camera frame. (Or video ended?)")
                    break 
                image = cv2.flip(frame, 1) 
            else: 
                in_rgb = camera_source.get()
                if in_rgb is None:
                    continue
                image = in_rgb.getCvFrame()

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image.flags.writeable = True

            # --- FIX 5: Default unpacking here ---
            defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, None, None, 0.0, 0.0, 0.0
            (pose_status, vertical_span, aspect_ratio, spine_angle, side,
             knee_angle, hip_spine_angle,
             thigh_vec_norm, knee_2d,
             hip_length, shoulder_length, torso_ratio) = defaults
            
            if results.pose_landmarks:
                (pose_status, vertical_span, aspect_ratio, spine_angle, side,
                 knee_angle, hip_spine_angle,
                 thigh_vec_norm, knee_2d,
                 hip_length, shoulder_length, torso_ratio) = analyze_pose(
                    results.pose_world_landmarks, results.pose_landmarks
                )
            
            if show_video:
                draw_visualizations(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    pose_status, 
                    vertical_span, 
                    aspect_ratio, 
                    spine_angle, 
                    side,
                    knee_angle, hip_spine_angle,
                    thigh_vec_norm, knee_2d,
                    hip_length, shoulder_length, torso_ratio
                )
                
                cv2.imshow('Posture Analyzer (Standalone)', image)
                if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
                    break 
    
    finally:
        print('Shutting down...')
        if is_laptop:
            if camera_source: camera_source.release()
        else:
            if oak_device: oak_device.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()