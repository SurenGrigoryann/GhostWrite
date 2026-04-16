import cv2
import time
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision import hand_landmarker

MODEL_PATH = "hand_landmarker.task"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Unable to open the camera.")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1280, 720)

options = hand_landmarker.HandLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

def landmark_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def is_finger_visible(landmarks, mcp, pip, dip, tip):
    # Just check that the landmarks exist and aren't collapsed to the same point
    d_tip_mcp = landmark_distance(landmarks[tip], landmarks[mcp])
    return d_tip_mcp > 0.01

def is_finger_extended(landmarks, mcp, pip, dip, tip):
    return landmarks[tip].y < landmarks[dip].y < landmarks[pip].y < landmarks[mcp].y

def is_right_hand(landmarks):
    return landmarks[4].x > landmarks[20].x

def point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def fingers_close(a, b, threshold=0.06):
    """Returns True if two fingertip landmarks are close together."""
    return landmark_distance(a, b) < threshold

BUTTON_MARGIN = 40
BUTTON_WIDTH = 260
BUTTON_HEIGHT = 90
BUTTON_SPACING = 40
BUTTON_TOP = 20
BUTTON_CONFIRM_FRAMES = 6

MARKER_BUTTON = (
    BUTTON_MARGIN,
    BUTTON_TOP,
    BUTTON_MARGIN + BUTTON_WIDTH,
    BUTTON_TOP + BUTTON_HEIGHT,
)
ERASER_BUTTON = (
    BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_SPACING,
    BUTTON_TOP,
    BUTTON_MARGIN + 2 * BUTTON_WIDTH + BUTTON_SPACING,
    BUTTON_TOP + BUTTON_HEIGHT,
)

def draw_button(frame, rect, label, active=False, hover=False):
    x1, y1, x2, y2 = rect
    fill_color = (220, 220, 220) if active else (40, 40, 40)
    if hover and not active:
        fill_color = (80, 80, 80)
    border_color = (255, 180, 0) if active else (140, 140, 140)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 3)
    text_color = (0, 0, 0) if active else (230, 230, 230)
    cv2.putText(
        frame, label, (x1 + 18, y1 + 58),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA,
    )

ink_strokes = []
current_stroke = None
tool_mode = "marker"
marker_hover = 0
eraser_hover = 0

def erase_ink_at(point, strokes, radius=40):
    radius_sq = radius * radius
    new_strokes = []
    for stroke in strokes:
        kept = []
        for pt in stroke:
            dx = pt[0] - point[0]
            dy = pt[1] - point[1]
            if dx * dx + dy * dy > radius_sq:
                kept.append(pt)
        if len(kept) > 1:
            new_strokes.append(kept)
    return new_strokes

with hand_landmarker.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_input = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_input, timestamp_ms)

        right_index = None
        right_index_extended = False
        right_paused = False
        left_fingertips = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                hand_is_right = is_right_hand(hand_landmarks)

                if not hand_is_right:
                    # ── Left hand: drawing ───────────────────────────────
                    index_vis  = is_finger_visible(hand_landmarks, 5, 6, 7, 8)
                    middle_vis = is_finger_visible(hand_landmarks, 9, 10, 11, 12)

                    if index_vis:
                        right_index = hand_landmarks[8]
                        right_index_extended = True  # always draw when index is visible, pause gesture handles stopping

                    # Pause when index + middle tips are close together
                    if index_vis and middle_vis:
                        if fingers_close(hand_landmarks[8], hand_landmarks[12]):
                            right_paused = True

                else:
                    # ── Right hand: buttons ──────────────────────────────
                    for tip_idx, mcp, pip, dip in [
                        (4,  1,  2,  3),
                        (8,  5,  6,  7),
                        (12, 9,  10, 11),
                        (16, 13, 14, 15),
                        (20, 17, 18, 19),
                    ]:
                        if is_finger_visible(hand_landmarks, mcp, pip, dip, tip_idx):
                            left_fingertips.append((
                                int(hand_landmarks[tip_idx].x * frame.shape[1]),
                                int(hand_landmarks[tip_idx].y * frame.shape[0]),
                            ))

        # ── Button hover (left hand) ─────────────────────────────────────
        marker_hovered = False
        eraser_hovered = False

        if left_fingertips:
            marker_hit = any(point_in_rect(pt, MARKER_BUTTON) for pt in left_fingertips)
            eraser_hit = any(point_in_rect(pt, ERASER_BUTTON) for pt in left_fingertips)

            if marker_hit and not eraser_hit:
                marker_hover += 1
                eraser_hover = 0
                marker_hovered = True
                if marker_hover >= BUTTON_CONFIRM_FRAMES:
                    tool_mode = "marker"
            elif eraser_hit and not marker_hit:
                eraser_hover += 1
                marker_hover = 0
                eraser_hovered = True
                if eraser_hover >= BUTTON_CONFIRM_FRAMES:
                    tool_mode = "eraser"
            else:
                marker_hover = 0
                eraser_hover = 0
        else:
            marker_hover = 0
            eraser_hover = 0

        # ── Drawing (right hand) ─────────────────────────────────────────
        if right_index is not None and right_index_extended and not right_paused:
            ink_x = int(right_index.x * frame.shape[1])
            ink_y = int(right_index.y * frame.shape[0])
            if tool_mode == "eraser":
                ink_strokes = erase_ink_at((ink_x, ink_y), ink_strokes)
                current_stroke = None
            else:
                if current_stroke is None:
                    current_stroke = []
                    ink_strokes.append(current_stroke)
                current_stroke.append((ink_x, ink_y))
        else:
            current_stroke = None

        # ── UI ───────────────────────────────────────────────────────────
        draw_button(frame, MARKER_BUTTON, "MARKER", active=(tool_mode == "marker"), hover=marker_hovered)
        draw_button(frame, ERASER_BUTTON, "ERASER", active=(tool_mode == "eraser"), hover=eraser_hovered)

        if right_paused:
            status_text  = "PAUSED  (separate fingers to resume)"
            status_color = (0, 140, 255)
        else:
            status_text  = f"Mode: {tool_mode.upper()}"
            status_color = (0, 0, 0)

        cv2.putText(frame, status_text,
            (BUTTON_MARGIN, BUTTON_TOP + BUTTON_HEIGHT + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

        cv2.putText(frame, "RIGHT hand = switch tools  |  LEFT hand = draw  |  index+middle together = pause",
            (BUTTON_MARGIN, BUTTON_TOP + BUTTON_HEIGHT + 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)

        # Draw ink strokes
        for stroke in ink_strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i - 1], stroke[i], (0, 0, 0), 8)

        # ── Green circles on all fingertips ──────────────────────────────
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for tip_idx, mcp, pip, dip in [
                    (4,  1,  2,  3),
                    (8,  5,  6,  7),
                    (12, 9,  10, 11),
                    (16, 13, 14, 15),
                    (20, 17, 18, 19),
                ]:
                    if is_finger_visible(hand_landmarks, mcp, pip, dip, tip_idx):
                        cx = int(hand_landmarks[tip_idx].x * frame.shape[1])
                        cy = int(hand_landmarks[tip_idx].y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 14, (0, 255, 80), 2)
                        cv2.circle(frame, (cx, cy), 4,  (0, 255, 80), -1)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()