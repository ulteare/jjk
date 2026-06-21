"""
Step 5: Unlimited Void - polished version

New in this version:
- Aspect-ratio-correct background video (scaled to frame height, center-cropped to width
  -- no more squish/stretch distortion).
- Translucent white finger-tip indicator dots (visual confirmation of hand tracking).
- "INFINITE VOID" label at the bottom of the screen while the effect is active.
- FPS counter + extra performance tuning knobs.

Requires:
    pip install mediapipe opencv-python numpy

First run will auto-download the segmentation model (~16MB) to the script's folder.
Put your video file (infinitevoid.mp4) in the same folder as this script.

Press 'q' to quit.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---- Config ----
VIDEO_PATH = "infinitevoid.mp4"
GESTURE_LABEL = "Infinite Void"
MODEL_PATH = "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

DEBOUNCE_FRAMES = 5
RELEASE_FRAMES = 5
MASK_BLUR_KSIZE = 9
VIDEO_SPEED = 2.5   # playback speed multiplier (achieved via frame skipping)

# --- Performance tuning ---
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
SEGMENTATION_SIZE = 192       # lowered from 256 -> faster inference, slightly softer mask edges
SEGMENT_EVERY_N_FRAMES = 3    # raised from 2 -> fewer segmentation calls per second

# --- Finger indicator config ---
FINGERTIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGERTIP_RADIUS = 12
FINGERTIP_COLOR = (255, 255, 255)    # white (BGR)
FINGERTIP_ALPHA = 0.5                # translucency

# ---- Download segmentation model if missing ----
if not os.path.exists(MODEL_PATH):
    print(f"Downloading segmentation model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the segmentation model automatically ({e}).\n"
            f"Please download it manually from:\n{MODEL_URL}\n"
            f"and place it at: {os.path.abspath(MODEL_PATH)}"
        )

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

WRIST = 0
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


def is_finger_curled(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y > landmarks[pip_idx].y


def is_index_middle_crossed(landmarks):
    index_tip_x = landmarks[INDEX_TIP].x
    middle_tip_x = landmarks[MIDDLE_TIP].x
    index_mcp_x = landmarks[INDEX_MCP].x
    middle_mcp_x = landmarks[MIDDLE_MCP].x

    base_order = index_mcp_x - middle_mcp_x
    tip_order = index_tip_x - middle_tip_x
    crossed = (base_order * tip_order) < 0

    tip_distance = ((landmarks[INDEX_TIP].x - landmarks[MIDDLE_TIP].x) ** 2 +
                     (landmarks[INDEX_TIP].y - landmarks[MIDDLE_TIP].y) ** 2) ** 0.5

    return crossed and tip_distance < 0.08


def is_unlimited_void(landmarks):
    ring_curled = is_finger_curled(landmarks, RING_TIP, RING_PIP)
    pinky_curled = is_finger_curled(landmarks, PINKY_TIP, PINKY_PIP)
    fingers_crossed = is_index_middle_crossed(landmarks)
    return ring_curled and pinky_curled and fingers_crossed


# ---- MediaPipe Image Segmenter ----
BaseOptions = mp_python.BaseOptions
ImageSegmenter = mp_vision.ImageSegmenter
ImageSegmenterOptions = mp_vision.ImageSegmenterOptions
VisionRunningMode = mp_vision.RunningMode

segmenter_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=False,
    output_confidence_masks=True,
)
segmenter = ImageSegmenter.create_from_options(segmenter_options)
BACKGROUND_CLASS_INDEX = 0


def get_person_mask_small(frame_bgr, timestamp_ms):
    """
    Runs segmentation on a small (SEGMENTATION_SIZE x SEGMENTATION_SIZE) version
    of the frame for speed. Returns a small float32 mask (0-1); caller upscales it.
    """
    small = cv2.resize(frame_bgr, (SEGMENTATION_SIZE, SEGMENTATION_SIZE))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment_for_video(mp_image, timestamp_ms)

    bg_confidence = result.confidence_masks[BACKGROUND_CLASS_INDEX].numpy_view()
    person_mask = 1.0 - bg_confidence
    return person_mask


def get_palm_center(landmarks, frame_w, frame_h):
    palm_points = [WRIST, INDEX_MCP, MIDDLE_MCP, 13, 17]
    xs = [landmarks[i].x for i in palm_points]
    ys = [landmarks[i].y for i in palm_points]
    cx = int((sum(xs) / len(xs)) * frame_w)
    cy = int((sum(ys) / len(ys)) * frame_h)
    return cx, cy


# ---- Background video controller ----

def resize_cover(frame, target_w, target_h):
    """
    Scale `frame` so its height matches target_h, preserving aspect ratio,
    then center-crop the width to target_w. Equivalent to CSS `background-size: cover`.
    If the scaled width is narrower than target_w (shouldn't normally happen given
    typical video/webcam ratios, but just in case), pads with black instead of cropping.
    """
    src_h, src_w = frame.shape[:2]
    scale = target_h / src_h
    new_w = int(round(src_w * scale))
    new_h = target_h

    resized = cv2.resize(frame, (new_w, new_h))

    if new_w >= target_w:
        # Center-crop the width
        x_start = (new_w - target_w) // 2
        return resized[:, x_start:x_start + target_w]
    else:
        # Pad width with black bars (rare case: very narrow source video)
        pad_total = target_w - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))


class BackgroundVideo:
    def __init__(self, path, speed=1.0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self.state = "off"
        self.last_frame = None
        self.speed = speed
        self._skip_accumulator = 0.0  # tracks fractional frames owed, for non-integer speeds

    def activate(self):
        self.state = "playing"
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.last_frame = None
        self._skip_accumulator = 0.0

    def deactivate(self):
        self.state = "off"
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.last_frame = None
        self._skip_accumulator = 0.0

    def _read_next_frame(self):
        """
        Reads one 'logical' frame at the configured speed by skipping extra
        physical frames as needed. E.g. speed=1.5 reads ~3 physical frames
        for every 2 logical frames (alternating skip of 1 and 2).
        Returns (ret, frame) from the last physical read.
        """
        # Always consume at least 1 frame
        ret, frame = self.cap.read()

        # speed=1.0 -> extra_owed accumulates 0 each time, never skips
        # speed=1.5 -> extra_owed accumulates 0.5 each time; skip an extra frame
        #              whenever it crosses a whole number
        self._skip_accumulator += (self.speed - 1.0)
        while self._skip_accumulator >= 1.0 and ret:
            ret, frame = self.cap.read()
            self._skip_accumulator -= 1.0

        return ret, frame

    def get_frame(self, target_size):
        w, h = target_size
        if self.state == "off":
            return None

        if self.state == "playing":
            ret, frame = self._read_next_frame()
            if not ret:
                self.state = "frozen"
                if self.last_frame is not None:
                    return resize_cover(self.last_frame, w, h)
                return None
            self.last_frame = frame
            return resize_cover(frame, w, h)

        if self.state == "frozen":
            if self.last_frame is not None:
                return resize_cover(self.last_frame, w, h)
            return None

        return None


def composite(webcam_frame, bg_frame, person_mask_small):
    """person_mask_small is at SEGMENTATION_SIZE resolution; upscale to frame size here."""
    mask_resized = cv2.resize(person_mask_small, (webcam_frame.shape[1], webcam_frame.shape[0]))
    mask_resized = cv2.GaussianBlur(mask_resized, (MASK_BLUR_KSIZE, MASK_BLUR_KSIZE), 0)
    mask_3ch = np.stack([mask_resized] * 3, axis=-1)

    composited = (webcam_frame.astype(np.float32) * mask_3ch +
                  bg_frame.astype(np.float32) * (1 - mask_3ch))
    return composited.astype(np.uint8)


def draw_finger_indicators(frame, hand_landmarks_list):
    """Draws translucent white circles on each fingertip for all detected hands."""
    if not hand_landmarks_list:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    for hand_landmarks in hand_landmarks_list:
        for idx in FINGERTIP_IDS:
            lm = hand_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(overlay, (cx, cy), FINGERTIP_RADIUS, FINGERTIP_COLOR, -1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, FINGERTIP_ALPHA, frame, 1 - FINGERTIP_ALPHA, 0)


def draw_gesture_label(frame, text):
    """Draws a small plain yellow label in the bottom-right corner."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 255, 255)  # yellow in BGR

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 15
    x = w - text_w - margin
    y = h - margin

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


# ---- Main loop ----

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

bg_video = BackgroundVideo(VIDEO_PATH, speed=VIDEO_SPEED)

cv2.namedWindow("Unlimited Void", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Unlimited Void", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

active = False
consecutive_detections = 0
consecutive_absences = 0
gesture_armed = True

frame_count = 0
cached_mask = None
start_time = time.time()

# FPS counter state
fps_display = 0.0
fps_frame_count = 0
fps_timer_start = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_this_frame = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_unlimited_void(hand_landmarks.landmark):
                gesture_this_frame = True

    # --- Debounced toggle logic ---
    if gesture_this_frame:
        consecutive_detections += 1
        consecutive_absences = 0
    else:
        consecutive_absences += 1
        consecutive_detections = 0

    if consecutive_absences >= RELEASE_FRAMES:
        gesture_armed = True

    if gesture_armed and consecutive_detections >= DEBOUNCE_FRAMES:
        active = not active
        gesture_armed = False
        if active:
            bg_video.activate()
        else:
            bg_video.deactivate()

    # --- Compositing (with frame-skipped segmentation) ---
    if active:
        bg_frame = bg_video.get_frame((w, h))
        if bg_frame is not None:
            if frame_count % SEGMENT_EVERY_N_FRAMES == 0 or cached_mask is None:
                timestamp_ms = int((time.time() - start_time) * 1000)
                cached_mask = get_person_mask_small(frame, timestamp_ms)
            frame = composite(frame, bg_frame, cached_mask)

    frame_count += 1

    # --- Finger indicator dots ---
    frame = draw_finger_indicators(frame, results.multi_hand_landmarks)

    # --- Gesture name label (only while active) ---
    if active:
        frame = draw_gesture_label(frame, GESTURE_LABEL)

    # --- FPS counter ---
    fps_frame_count += 1
    elapsed = time.time() - fps_timer_start
    if elapsed >= 0.5:  # update twice a second
        fps_display = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_timer_start = time.time()

    # Debug overlay (remove once happy)
    status = f"active={active} state={bg_video.state} hold={consecutive_detections} FPS={fps_display:.1f}"
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Unlimited Void", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
bg_video.cap.release()
cv2.destroyAllWindows()
