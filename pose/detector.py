import os
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(MODEL_DIR, 'pose_landmarker_heavy.task')

def create_pose_landmarker():
    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO
    )
    return PoseLandmarker.create_from_options(options)
