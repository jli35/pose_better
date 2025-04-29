import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker
from .drawing import draw_landmarks_on_image
from .utils import resize_to_screen

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
        resized_image = resize_to_screen(annotated_image)

        cv2.imshow('Pose Landmarks Live', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        frame_count += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
