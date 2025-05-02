import cv2
import numpy as np

def resize_to_screen(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scaling_factor = min(max_width / w, max_height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def to_np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

def calculate_angle(a, b, c):
    ba, bc = np.array(a) - np.array(b), np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0)) 

def radian_to_degree(a):
    return np.degrees(a)

def normalize_landmarks(landmarks):
    left_hip = to_np(landmarks[23])
    right_hip = to_np(landmarks[24])
    origin = (left_hip + right_hip) / 2
    return [to_np(lm) - origin for lm in landmarks]

def calculate_velocities(curr_landmarks, prev_landmarks, delta_t):
    return [(to_np(curr) - to_np(prev)) / delta_t for curr, prev in zip(curr_landmarks, prev_landmarks)]

