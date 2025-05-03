from pose.landmarker import create_pose_landmarker
from pose.camera import run_camera
from pose.image import run_image

def main():
    mode = 'IMAGE'  # 'IMAGE' or 'VIDEO'
    run = run_image if mode == 'IMAGE' else run_camera

    with create_pose_landmarker(mode=mode) as landmarker:
        run(landmarker)

if __name__ == "__main__":
    main()
