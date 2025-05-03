import os
import cv2
import mediapipe as mp
from .drawing import draw_landmarks_on_image
from .utils import resize_to_screen

IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'golfStart.jpg')

def run_image(landmarker):
    mp_image = mp.Image.create_from_file(IMAGE_PATH)
    result = landmarker.detect(mp_image)
    
    if result.pose_landmarks:
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result, True)
        resized_image = resize_to_screen(annotated_image)
        cv2.imshow("Pose with Angles", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose detected.")
