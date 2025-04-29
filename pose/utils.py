import cv2

def resize_to_screen(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scaling_factor = min(max_width / w, max_height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
