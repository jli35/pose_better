import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import cv2

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def resize_to_screen(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scaling_factor = min(max_width / w, max_height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

model_path = os.path.join(MODEL_DIR, 'pose_landmarker_full.task')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)



# begin capture
cap = cv2.VideoCapture(0)  # 0 = default camera
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

with PoseLandmarker.create_from_options(options) as landmarker:
    # img_path = os.path.join(DATA_DIR, 'samples', 'golfStart.jpg')
    # mp_image = mp.Image.create_from_file(img_path)
    # pose_landmarker_result = landmarker.detect(mp_image)
    # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
    # resized_image = resize_to_screen(annotated_image)
    # cv2.imshow('Pose Landmarks', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # live footage
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to Mediapipe Image
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # Detect pose
        pose_landmarker_result = landmarker.detect_for_video(mp_frame, frame_count)
        # Draw landmarks
        annotated_image = draw_landmarks_on_image(frame_rgb, pose_landmarker_result)
        # Resize for display
        resized_image = resize_to_screen(annotated_image)
        # Show it
        cv2.imshow('Pose Landmarks Live', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        frame_count += 1
        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
