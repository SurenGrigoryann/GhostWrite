import cv2
import time
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision import hand_landmarker

options = hand_landmarker.HandLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
)

with hand_landmarker.HandLandmarker.create_from_options(options) as landmarker:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    mp_input = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
    res = landmarker.detect_for_video(mp_input, int(time.time() * 1000))
    print(type(res))
    print([a for a in dir(res) if not a.startswith('_')])
    print('handedness', getattr(res, 'handedness', None))
    print('hand_landmarks len', len(getattr(res, 'hand_landmarks', [])))
