import cv2
import time
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision import hand_landmarker

print('start')
options = hand_landmarker.HandLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
print('options created')
with hand_landmarker.HandLandmarker.create_from_options(options) as landmarker:
    print('landmarker created')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    mp_input = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_input, timestamp_ms)
    print('detect done', result)
