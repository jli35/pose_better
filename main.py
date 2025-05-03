from pose.landmarker import create_pose_landmarker
from pose.camera import run_camera
from pose.image import run_image
from pose.video import run_video  # <- import your saved video handler

def main():
    mode = 'VIDEO'  # Options: 'IMAGE', 'VIDEO', 'LIVE_STREAM'

    if mode == 'IMAGE':
        run = run_image
    elif mode == 'VIDEO':
        run = run_video
    elif mode == 'LIVE_STREAM':
        run = run_camera
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    with create_pose_landmarker(mode='VIDEO') as landmarker:
        run(landmarker)

if __name__ == "__main__":
    main()
