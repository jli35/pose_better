from .utils import calculate_angle, calculate_velocities, normalize_landmarks, to_np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

def extract_joint_angles(landmarks):
    if not landmarks.pose_landmarks:
        return {}
    l = [to_np(lm) for lm in landmarks.pose_landmarks[0]]
    return {
        'left_elbow': calculate_angle(l[11], l[13], l[15]),
        'right_elbow': calculate_angle(l[12], l[14], l[16]),
        'left_knee': calculate_angle(l[23], l[25], l[27]),
        'right_knee': calculate_angle(l[24], l[26], l[28]),
        'left_shoulder': calculate_angle(l[13], l[11], l[23]),
        'right_shoulder': calculate_angle(l[14], l[12], l[24]),
        'hip_rotation': calculate_angle(l[11], l[23], l[24]),
    }

def extract_features(curr_landmarks: NormalizedLandmarkList,
                     prev_landmarks: NormalizedLandmarkList = None,
                     delta_t: float = 1/30):
    curr = curr_landmarks.landmark
    normed = normalize_landmarks(curr)
    angles = extract_joint_angles(curr)

    if prev_landmarks:
        prev = prev_landmarks.landmark
        velocities = calculate_velocities(curr, prev, delta_t)
    else:
        velocities = None

    return {
        'normalized': normed,
        'angles': angles,
        'velocities': velocities
    }
