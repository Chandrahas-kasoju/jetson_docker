import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math
from std_msgs.msg import Float32MultiArray, ColorRGBA
from vision_msgs.msg import Point2D, BoundingBox2D
import time
import requests

class YOLOLandmark:
    def __init__(self, x, y, conf, img_w, img_h):
        # Prevent division by zero if img_w/img_h is 0 somehow
        img_w = max(img_w, 1)
        img_h = max(img_h, 1)
        self.x = x / img_w
        self.y = y / img_h
        self.visibility = conf

class ThermalPoseComplex(Node):
    def __init__(self):
        super().__init__('thermal_pose_complex')
        self.model = YOLO('yolov8n-pose.pt')
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/hospibot/image_raw',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(Image, '/hospibot/image_pose_annotated', 10)
        self.bb_pub = self.create_publisher(BoundingBox2D, '/hospibot/pose_bbox', 10)
        self.eye_center_pub = self.create_publisher(Point2D, '/face_tracker/eye_center', 10)
        self.led_pub = self.create_publisher(ColorRGBA, '/led_color', 10)
        self.get_logger().info('Complex Heuristics Pose Node started.')
        
        self.NTFY_TOPIC = "hospibot_lying_down_alarm"
        self.notification_cooldown = 30.0
        self.last_notification_time = 0.0
        self.lying_down_frame_count = 0
        self.LYING_DOWN_FRAME_TRIGGER = 5 
        self.missed_lying_down_frames = 0
        self.MAX_MISSED_FRAMES = 5
        
        self.last_blink_time = time.time()
        self.led_blink_on = True
        self.current_led_state = 'cyan'
        self.alarm_triggered_time = 0.0
        
        # Publish cyan on startup
        led_msg = ColorRGBA(r=0.0, g=255.0, b=255.0, a=0.5)
        self.led_pub.publish(led_msg)
        
        # --- Thresholds perfectly matching aspect_ratio_standalone_v3.py ---
        self.aspect_ratio_tall_threshold = 2.0
        self.aspect_ratio_wide_threshold = 0.6
        self.foreshortened_span_threshold = 0.45
        self.foreshortened_aspect_ratio_threshold = 1.2
        self.spine_angle_1 = 60.0
        self.spine_angle_2 = 45.0
        self.knee_angle_threshold = 45.0
        self.hip_spine_angle_threshold = 160.0

    def send_notification(self):
        current_time = time.time()
        if (current_time - self.last_notification_time) > self.notification_cooldown:
            self.get_logger().warn('LYING DOWN DETECTED! Sending ntfy notification...')
            try:
                requests.post(
                    f"https://ntfy.sh/{self.NTFY_TOPIC}",
                    data="Fall detected! A person is lying down.".encode(encoding='utf-8'),
                    headers={
                        "Title": "TurtleBot Alert: Possible Fall Detected",
                        "Priority": "urgent",
                        "Tags": "warning,rotating_light",
                    })
                self.last_notification_time = current_time
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Could not send notification: {e}")
        else:
            self.get_logger().info('Not sending notification cause its still in notification cooldown period.')

    def process_frame_and_publish_bbox(self, det, h, w):
        x_min, y_min, x_max, y_max = det['xyxy_orig']
        
        msg = BoundingBox2D()
        msg.center.position.x = float((x_min + x_max) / 2.0)
        msg.center.position.y = float((y_min + y_max) / 2.0)
        msg.center.theta = 0.0
        msg.size_x = float(x_max - x_min)
        msg.size_y = float(y_max - y_min)
        self.bb_pub.publish(msg)

        kpts = det['res_obj'].keypoints.xy[0].cpu().numpy()
        unrot_kpts = self.unrotate_points(kpts, det['rot_code'], w, h)
        
        # YOLO indices: 1 = L-Eye, 2 = R-Eye
        if len(unrot_kpts) > 2:
            left_eye = unrot_kpts[1]
            right_eye = unrot_kpts[2]
            
            if left_eye[0] > 0 and right_eye[0] > 0:
                eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
                eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
                
                eye_msg = Point2D()
                eye_msg.x = float(eye_center_x)
                eye_msg.y = float(eye_center_y)
                self.eye_center_pub.publish(eye_msg)

    def unrotate_points(self, pts, rot_code, orig_w, orig_h):
        if len(pts) == 0: return pts
        unrot_pts = np.zeros_like(pts)
        if rot_code == cv2.ROTATE_90_CLOCKWISE:
            unrot_pts[:, 0] = pts[:, 1]
            unrot_pts[:, 1] = orig_h - pts[:, 0]
        elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            unrot_pts[:, 0] = orig_w - pts[:, 1]
            unrot_pts[:, 1] = pts[:, 0]
        else:
            unrot_pts = pts.copy()
        return unrot_pts

    def get_visible_side(self, sl):
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14

        left_side_visibility = sum(sl[i].visibility for i in [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE])
        right_side_visibility = sum(sl[i].visibility for i in [RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE])
        
        side_used = "Left" if left_side_visibility > right_side_visibility else "Right"
        use_left_side = left_side_visibility > right_side_visibility
        return use_left_side, side_used

    def analyze_pose(self, sl):
        # sl is a list of YOLOLandmark objects
        defaults = "No Human", 0.0, 0.0, 0.0, "", 0.0, 0.0, 0.0, 0.0, 0.0
        if len(sl) < 15:
            return defaults
            
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14

        # 1. RATIOS & MIDPOINTS
        try:
            ls, rs = sl[LEFT_SHOULDER], sl[RIGHT_SHOULDER]
            lh, rh = sl[LEFT_HIP], sl[RIGHT_HIP]

            shoulder_length = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
            hip_length = math.sqrt((lh.x - rh.x)**2 + (lh.y - rh.y)**2)

            mid_shoulder_x = (ls.x + rs.x) / 2
            mid_shoulder_y = (ls.y + rs.y) / 2
            mid_hip_x = (lh.x + rh.x) / 2
            mid_hip_y = (lh.y + rh.y) / 2
            
            torso_height = math.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
            torso_ratio = shoulder_length / torso_height if torso_height > 0.001 else 0.0
            
        except Exception:
            shoulder_length, torso_height, torso_ratio, hip_length = 0.0, 0.0, 0.0, 0.0
            mid_shoulder_x, mid_shoulder_y, mid_hip_x, mid_hip_y = 0.0, 0.0, 0.0, 0.0

        # 2. ARE KNEES VISIBLE?
        l_knee_vis = sl[LEFT_KNEE].visibility
        r_knee_vis = sl[RIGHT_KNEE].visibility
        
        if l_knee_vis < 0.2 and r_knee_vis < 0.2: # YOLO confs are usually slightly different than MP
            return "No Knee", 0.0, 0.0, 0.0, "", 0.0, 0.0, 0.0, 0.0, 0.0

        # 3. 2D METRICS
        visible_landmarks = [lm for lm in sl if lm.visibility > 0.2]
        if not visible_landmarks:
            return defaults

        min_y = min(lm.y for lm in visible_landmarks)
        max_y = max(lm.y for lm in visible_landmarks)
        min_x = min(lm.x for lm in visible_landmarks)
        max_x = max(lm.x for lm in visible_landmarks)
        
        vertical_span = max_y - min_y
        horizontal_span = max_x - min_x
        if horizontal_span == 0: horizontal_span = 0.001
        aspect_ratio = vertical_span / horizontal_span 

        vec_x = mid_shoulder_x - mid_hip_x
        vec_y = mid_shoulder_y - mid_hip_y 
        
        spine_angle = abs(math.degrees(math.atan2(vec_x, vec_y)))
        if spine_angle > 90:
            spine_angle = 180 - spine_angle

        use_left_side, side = self.get_visible_side(sl)
        hip_lm = sl[LEFT_HIP] if use_left_side else sl[RIGHT_HIP]
        knee_lm = sl[LEFT_KNEE] if use_left_side else sl[RIGHT_KNEE]

        thigh_vec_x = knee_lm.x - hip_lm.x
        thigh_vec_y = knee_lm.y - hip_lm.y 

        dot_product = (vec_x * thigh_vec_x) + (vec_y * thigh_vec_y)
        mag_spine = math.sqrt(vec_x**2 + vec_y**2)
        mag_hip = math.sqrt(thigh_vec_x**2 + thigh_vec_y**2)
        
        if mag_spine * mag_hip > 0:
            cosine_angle = max(min(dot_product / (mag_spine * mag_hip), 1.0), -1.0)
            hip_spine_angle = math.degrees(math.acos(cosine_angle))
        else:
            hip_spine_angle = 0.0

        knee_angle = abs(math.degrees(math.atan2(thigh_vec_x, thigh_vec_y)))
        if knee_angle > 90:
            knee_angle = 180 - knee_angle

        # 4. DETERMINE STATUS
        pose_status = "Not Lying Down" 

        if torso_ratio > 0.3:
            if aspect_ratio > self.aspect_ratio_tall_threshold:
                pose_status = "Not Lying Down 1"
            elif aspect_ratio < self.aspect_ratio_wide_threshold:
                delta_x = abs(vec_x)
                delta_y = abs(vec_y)
                if delta_x > delta_y:
                    pose_status = "Lying Down1"
                else:
                    pose_status = "Not Lying Down 3"
            elif (vertical_span < self.foreshortened_span_threshold and 
                aspect_ratio < self.foreshortened_aspect_ratio_threshold) and (thigh_vec_y < thigh_vec_x):
                pose_status = "Lying Down2"
            else:
                pose_status = "Lying Down3"

            if pose_status == "Lying Down1" and (thigh_vec_y < thigh_vec_x):
                if hip_spine_angle < self.hip_spine_angle_threshold:
                    pose_status = "Not Lying Down 7"
                else:
                    pose_status = "Lying Down4"

        elif torso_ratio <= 0.3:
            if aspect_ratio > self.aspect_ratio_tall_threshold:
                pose_status = "Not Lying Down 1b"
            elif aspect_ratio < self.aspect_ratio_wide_threshold:
                delta_x = abs(vec_x)
                delta_y = abs(vec_y)
                if delta_x > delta_y and spine_angle > self.spine_angle_2 and knee_angle > self.knee_angle_threshold:
                    pose_status = "Lying Down"
                else:
                    pose_status = "Not Lying Down 3b"
            elif (vertical_span < self.foreshortened_span_threshold and 
                aspect_ratio < self.foreshortened_aspect_ratio_threshold) and spine_angle > self.spine_angle_2 and knee_angle > self.knee_angle_threshold:
                pose_status = "Lying Down"
            else:
                pose_status = "Not Lying Down 5b"
                
        return (pose_status, vertical_span, aspect_ratio, spine_angle, side,
                knee_angle, hip_spine_angle, hip_length, shoulder_length, torso_ratio)

    def draw_visualizations(self, image, start_y, pose_status, vertical_span, aspect_ratio, spine_angle, side, knee_angle, hip_spine_angle, torso_ratio):
        font_scale = 0.5
        thickness = 1

        if "Not Lying Down" in pose_status:
            status_color = (0, 255, 0) # Green
        elif "Lying Down" in pose_status:
            status_color = (0, 0, 255) # Red
        else:
            status_color = (0, 255, 0) # Default green

        cv2.putText(image, f'Status: {pose_status}', (10, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_image.dtype == np.uint16 or cv_image.dtype == np.int16:
                cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
                base_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                base_img = cv_image
                
            orig_h, orig_w = base_img.shape[:2]
            final_output = base_img.copy()
            
            orientations = [
                (None, None),
                (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE)
            ]
            
            all_detections = []
            
            for rot_code, unrot_code in orientations:
                test_img = cv2.rotate(base_img, rot_code) if rot_code is not None else base_img.copy()
                results = self.model(test_img, conf=0.5, verbose=False)
                
                for i in range(len(results[0].boxes)):
                    box = results[0].boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    corners = np.array([
                        [xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], 
                        [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]
                    ])
                    unrot_corners = self.unrotate_points(corners, rot_code, orig_w, orig_h)
                    xyxy_orig = [np.min(unrot_corners[:,0]), np.min(unrot_corners[:,1]), 
                                 np.max(unrot_corners[:,0]), np.max(unrot_corners[:,1])]
                    
                    all_detections.append({
                        'rot_code': rot_code,
                        'unrot_code': unrot_code,
                        'conf': float(box.conf[0].cpu().numpy()),
                        'xyxy_orig': xyxy_orig,
                        'res_obj': results[0][i],
                        'test_img': test_img
                    })
            
            any_lying_down = False
            
            if len(all_detections) > 0:
                bboxes = []
                scores = []
                for det in all_detections:
                    x1, y1, x2, y2 = det['xyxy_orig']
                    bboxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(det['conf'])
                    
                indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.2, 0.4)
                
                if len(indices) > 0:
                    first_det = all_detections[indices.flatten()[0]]
                    self.process_frame_and_publish_bbox(first_det, orig_h, orig_w)
                    
                    text_y_offset = 20
                    for idx in indices.flatten():
                        det = all_detections[idx]
                        
                        # Draw YOLO skeleton
                        black_canvas = np.zeros_like(det['test_img'])
                        drawn_canvas = det['res_obj'].plot(img=black_canvas)
                        if det['unrot_code'] is not None:
                            drawn_canvas = cv2.rotate(drawn_canvas, det['unrot_code'])
                        mask = np.any(drawn_canvas != [0, 0, 0], axis=-1)
                        final_output[mask] = drawn_canvas[mask]
                        
                        # Apply COMPLEX Logic
                        kpts = det['res_obj'].keypoints.xy[0].cpu().numpy()
                        kpts_conf = det['res_obj'].keypoints.conf[0].cpu().numpy() if det['res_obj'].keypoints.conf is not None else np.ones(17)
                        
                        unrot_kpts = self.unrotate_points(kpts, det['rot_code'], orig_w, orig_h)
                        
                        # Create YOLOLandmark list
                        sl = []
                        for i in range(len(unrot_kpts)):
                            # Set visibility to 0 if the keypoint wasn't detected [0,0]
                            vis = kpts_conf[i] if (unrot_kpts[i][0] > 0 and unrot_kpts[i][1] > 0) else 0.0
                            sl.append(YOLOLandmark(unrot_kpts[i][0], unrot_kpts[i][1], vis, orig_w, orig_h))
                            
                        metrics = self.analyze_pose(sl)
                        (pose_status, vertical_span, aspect_ratio, spine_angle, side,
                         knee_angle, hip_spine_angle, hip_length, shoulder_length, torso_ratio) = metrics
                         
                        if "Lying Down" in pose_status and "Not" not in pose_status:
                            any_lying_down = True
                            
                        self.draw_visualizations(
                            final_output, text_y_offset, pose_status, vertical_span, 
                            aspect_ratio, spine_angle, side, knee_angle, hip_spine_angle, torso_ratio
                        )
                        text_y_offset += 150 # Shift down for next person

            # --- ALARM TRIGGER LOGIC ---
            if any_lying_down:
                self.missed_lying_down_frames = 0
                self.lying_down_frame_count += 1
                self.get_logger().info(f'Possible Lying Down... Count: {self.lying_down_frame_count}/{self.LYING_DOWN_FRAME_TRIGGER}', throttle_duration_sec=1)
                
                if self.lying_down_frame_count >= self.LYING_DOWN_FRAME_TRIGGER:
                    self.send_notification()
                    self.alarm_triggered_time = time.time()
                    self.lying_down_frame_count = 0 
            else:
                if self.lying_down_frame_count > 0:
                    self.missed_lying_down_frames += 1
                    if self.missed_lying_down_frames > self.MAX_MISSED_FRAMES:
                        self.get_logger().info('Lying down state reset.')
                        self.lying_down_frame_count = 0
                        self.missed_lying_down_frames = 0
                
            # --- LED STATE MACHINE ---
            current_time = time.time()
            if current_time - self.alarm_triggered_time < 5.0:  # Blink for 5 seconds after alarm triggers
                if current_time - self.last_blink_time > 0.25:  # Blink every 0.25s
                    self.led_blink_on = not self.led_blink_on
                    self.last_blink_time = current_time
                    
                led_msg = ColorRGBA()
                if self.led_blink_on:
                    led_msg.r = 255.0; led_msg.g = 0.0; led_msg.b = 0.0; led_msg.a = 0.5
                else:
                    led_msg.r = 0.0; led_msg.g = 0.0; led_msg.b = 0.0; led_msg.a = 0.5
                self.led_pub.publish(led_msg)
                self.current_led_state = 'blinking'
            else:
                led_msg = ColorRGBA(r=0.0, g=255.0, b=255.0, a=0.5)
                self.led_pub.publish(led_msg)
                self.current_led_state = 'cyan'

            cv2.imshow("Complex Logic Pose Node", final_output)
            cv2.waitKey(1)
            
            annotated_msg = self.bridge.cv2_to_imgmsg(final_output, encoding="bgr8")
            annotated_msg.header = msg.header
            self.publisher_.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ThermalPoseComplex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
