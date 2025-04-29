from pose.detector import create_pose_landmarker
from pose.camera import run_camera

def main():
    with create_pose_landmarker() as landmarker:
        run_camera(landmarker)

if __name__ == "__main__":
    main()
