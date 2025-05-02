import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarker
from .drawing import draw_landmarks_on_image
from .utils import resize_to_screen
from .feature_extraction import extract_joint_angles

def run_camera(landmarker: PoseLandmarker):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_landmarker_result = landmarker.detect_for_video(mp_frame, frame_count)

        annotated_image = draw_landmarks_on_image(frame_rgb, pose_landmarker_result)

        # display angles
        angles = extract_joint_angles(pose_landmarker_result)

        # Annotate angles on image
        for name, angle in angles.items():
            # Pick a landmark to anchor the text near
            if name == 'left_elbow':
                anchor = pose_landmarker_result.pose_landmarks[0][13]  # left elbow
            elif name == 'right_elbow':
                anchor = pose_landmarker_result.pose_landmarks[0][14]  # right elbow
            elif name == 'left_knee':
                anchor = pose_landmarker_result.pose_landmarks[0][25]  # left knee
            elif name == 'right_knee':
                anchor = pose_landmarker_result.pose_landmarks[0][26]  # right knee
            else:
                continue

            # Convert normalized coordinates to pixel coordinates
            h, w = frame_rgb.shape[:2]
            x = int(anchor.x * w)
            y = int(anchor.y * h)
            text = f"{name}: {np.degrees(angle):.2f} degrees"
            cv2.putText(annotated_image, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        resized_image = resize_to_screen(annotated_image)

        cv2.imshow('Pose Landmarks Live', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        frame_count += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
