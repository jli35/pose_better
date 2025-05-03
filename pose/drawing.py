import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from .feature_extraction import extract_joint_angles

def draw_landmarks_on_image(rgb_image, detection_result, draw_angles=False):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        if draw_angles:
            joint_angles = extract_joint_angles(detection_result)

            for joint, angle in joint_angles.items():
                if joint == 'left_elbow':
                    anchor = pose_landmarks[13]  # left elbow
                elif joint == 'right_elbow':
                    anchor = pose_landmarks[14]  # right elbow
                elif joint == 'left_knee':
                    anchor = pose_landmarks[25]  # left knee
                elif joint == 'right_knee':
                    anchor = pose_landmarks[26]  # right knee
                else:
                    continue  # Skip joints not in the list
                
                # Convert normalized coordinates to pixel coordinates
                h, w = rgb_image.shape[:2]
                x = int(anchor.x * w)
                y = int(anchor.y * h)
                
                text = f"{joint}: {np.degrees(angle):.2f}°"
                cv2.putText(annotated_image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image
