"""
Combined Gestures: Unlimited Void + Hollow Purple

Gesture 1 - UNLIMITED VOID (from step5):
Hand action: Index and middle fingers crossed, ring + pinky curled
Effect: Toggle background video with person segmentation

Gesture 2 - HOLLOW PURPLE (from step6d):
Hand action: Two-phase gesture
  Phase 1 (CHARGING): Thumb, index, middle fingertips touching, ring + pinky curled, palm facing up
  Phase 2 (FLICK): Index and middle fingers rapidly snap away from thumb
Effect: Charging orb that grows, then explodes with flash on flick

Requires:
    pip install mediapipe opencv-python numpy

First run will auto-download the segmentation model (~16MB) to the script's folder.
Put your video file (infinitevoid.mp4) in the same folder as this script.

Press 'q' to quit.
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---- Config ----
VIDEO_PATH = "infinitevoid.mp4"
MODEL_PATH = "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

# --- Unlimited Void config ---
UV_DEBOUNCE_FRAMES = 5
UV_RELEASE_FRAMES = 5
MASK_BLUR_KSIZE = 9
VIDEO_SPEED = 3

# --- Hollow Purple config ---
HP_DEBOUNCE_FRAMES = 5
CHARGE_DURATION = 1.8
ORB_MAX_RADIUS = 70
FLICK_VELOCITY_THRESHOLD = 0.035
TOUCH_DISTANCE_THRESHOLD = 0.06
FLICK_DELAY_DURATION = 0.7
FLICK_FLY_DURATION = 0.35
FLASH_HOLD_DURATION = 0.3
FLASH_FADE_DURATION = 0.4
PARTICLE_COUNT = 22
STAR_INNER_RATIO = 0.075
STAR_SCALE = 0.8

# --- Performance ---
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
SEGMENTATION_SIZE = 192
SEGMENT_EVERY_N_FRAMES = 3

# --- Finger indicator config ---
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGERTIP_RADIUS = 12
FINGERTIP_COLOR = (255, 255, 255)
FINGERTIP_ALPHA = 0.5

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
    max_num_hands=2,  # Support both hands (one for each gesture)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


# ---- Gesture detection functions ----

def is_finger_curled(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y > landmarks[pip_idx].y


def fingertip_distance(landmarks, idx_a, idx_b):
    a, b = landmarks[idx_a], landmarks[idx_b]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# --- Unlimited Void detection ---

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


# --- Hollow Purple detection ---

def get_pinch_center(landmarks, frame_w, frame_h):
    xs = [landmarks[i].x for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)]
    ys = [landmarks[i].y for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)]
    cx = int((sum(xs) / 3) * frame_w)
    cy = int((sum(ys) / 3) * frame_h)
    return cx, cy


def is_palm_facing_up(landmarks):
    wrist_z = landmarks[WRIST].z
    middle_mcp_z = landmarks[MIDDLE_MCP].z
    return middle_mcp_z < wrist_z


def is_charging_pose(landmarks):
    ring_curled = is_finger_curled(landmarks, RING_TIP, RING_PIP)
    pinky_curled = is_finger_curled(landmarks, PINKY_TIP, PINKY_PIP)

    thumb_index_close = fingertip_distance(landmarks, THUMB_TIP, INDEX_TIP) < TOUCH_DISTANCE_THRESHOLD
    thumb_middle_close = fingertip_distance(landmarks, THUMB_TIP, MIDDLE_TIP) < TOUCH_DISTANCE_THRESHOLD
    index_middle_close = fingertip_distance(landmarks, INDEX_TIP, MIDDLE_TIP) < TOUCH_DISTANCE_THRESHOLD

    fingers_pinched = thumb_index_close and thumb_middle_close and index_middle_close
    palm_up = is_palm_facing_up(landmarks)

    return ring_curled and pinky_curled and fingers_pinched and palm_up


def pinch_spread(landmarks):
    return (fingertip_distance(landmarks, THUMB_TIP, INDEX_TIP) +
            fingertip_distance(landmarks, THUMB_TIP, MIDDLE_TIP) +
            fingertip_distance(landmarks, INDEX_TIP, MIDDLE_TIP))


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
    small = cv2.resize(frame_bgr, (SEGMENTATION_SIZE, SEGMENTATION_SIZE))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment_for_video(mp_image, timestamp_ms)

    bg_confidence = result.confidence_masks[BACKGROUND_CLASS_INDEX].numpy_view()
    person_mask = 1.0 - bg_confidence
    return person_mask


# ---- Background video controller ----

def resize_cover(frame, target_w, target_h):
    src_h, src_w = frame.shape[:2]
    scale = target_h / src_h
    new_w = int(round(src_w * scale))
    new_h = target_h

    resized = cv2.resize(frame, (new_w, new_h))

    if new_w >= target_w:
        x_start = (new_w - target_w) // 2
        return resized[:, x_start:x_start + target_w]
    else:
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
        self._skip_accumulator = 0.0

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
        ret, frame = self.cap.read()
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
    mask_resized = cv2.resize(person_mask_small, (webcam_frame.shape[1], webcam_frame.shape[0]))
    mask_resized = cv2.GaussianBlur(mask_resized, (MASK_BLUR_KSIZE, MASK_BLUR_KSIZE), 0)
    mask_3ch = np.stack([mask_resized] * 3, axis=-1)

    composited = (webcam_frame.astype(np.float32) * mask_3ch +
                  bg_frame.astype(np.float32) * (1 - mask_3ch))
    return composited.astype(np.uint8)


# ---- Hollow Purple orb + flash animation ----

PURPLE_CORE = np.array([80, 0, 60])
PURPLE_GLOW = np.array([110, 10, 80])
WHITE = np.array([255, 255, 255])


def lerp_color(c1, c2, t):
    return tuple(int(c) for c in (c1 * (1 - t) + c2 * t))


def lerp_color_np(c1, c2, t):
    return (c1.astype(np.float32) * (1 - t) + c2.astype(np.float32) * t)


def make_radial_gradient_patch(shape, center, max_radius, color_stops):
    h, w = shape[:2]
    cx, cy = center
    r = int(math.ceil(max_radius))

    x0, x1 = max(cx - r, 0), min(cx + r, w)
    y0, y1 = max(cy - r, 0), min(cy + r, h)

    if x1 <= x0 or y1 <= y0:
        return None, None, (0, 0, 0, 0)

    patch_h, patch_w = y1 - y0, x1 - x0
    local_cx, local_cy = cx - x0, cy - y0

    yy, xx = np.mgrid[0:patch_h, 0:patch_w]
    dist = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    norm_dist = np.clip(dist / max_radius, 0, 1)

    gradient = np.zeros((patch_h, patch_w, 3), dtype=np.float32)
    for i in range(len(color_stops) - 1):
        pos0, color0 = color_stops[i]
        pos1, color1 = color_stops[i + 1]
        seg_mask = (norm_dist >= pos0) & (norm_dist <= pos1)
        if pos1 > pos0:
            t = (norm_dist - pos0) / (pos1 - pos0)
        else:
            t = np.zeros_like(norm_dist)
        t = np.clip(t, 0, 1)[..., None]
        seg_color = color0 * (1 - t) + color1 * t
        gradient[seg_mask] = seg_color[seg_mask]

    alpha = (np.clip(1.0 - norm_dist, 0, 1) ** 0.8)[..., None]
    return gradient, alpha, (x0, y0, x1, y1)


_STAR_LENGTH_MULTIPLIERS = [1.8, 2.3, 1.2, 2.5, 1.5, 2.0, 1.1, 1.9]

def draw_star(frame, center, outer_radius, inner_ratio=0.4, points=8, color=(255, 255, 255), rotation=0.0):
    cx, cy = center
    angle_step = math.pi / points
    vertices = []

    for i in range(points * 2):
        if i % 2 == 0:
            point_index = i // 2
            r = outer_radius * _STAR_LENGTH_MULTIPLIERS[point_index]
        else:
            r = outer_radius * inner_ratio
        angle = rotation + i * angle_step
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        vertices.append((int(x), int(y)))
    pts = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(frame, pts, color, lineType=cv2.LINE_AA)
    return frame


class Particle:
    __slots__ = ("x", "y", "speed", "size", "life", "max_life")

    def __init__(self, center, screen_w, screen_h):
        cx, cy = center
        max_dist = math.hypot(screen_w, screen_h) / 2
        dist = max_dist * (0.3 + 0.7 * random.random())
        angle = random.uniform(0, 2 * math.pi)
        self.x = cx + dist * math.cos(angle)
        self.y = cy + dist * math.sin(angle)
        self.speed = random.uniform(120, 260)
        self.size = random.uniform(1.5, 3.5)
        self.max_life = random.uniform(1.2, 2.2)
        self.life = self.max_life

    def update(self, dt, center, inward):
        cx, cy = center
        dx, dy = cx - self.x, cy - self.y
        dist = math.hypot(dx, dy)
        if dist > 1:
            dirx, diry = dx / dist, dy / dist
            if not inward:
                dirx, diry = -dirx, -diry
            self.x += dirx * self.speed * dt
            self.y += diry * self.speed * dt
        self.life -= dt

    def is_dead(self, center, screen_w, screen_h):
        cx, cy = center
        dist = math.hypot(self.x - cx, self.y - cy)
        max_dist = math.hypot(screen_w, screen_h) / 2
        return self.life <= 0 or dist < 6 or dist > max_dist * 1.3

    def position(self):
        return int(self.x), int(self.y)

    def alpha(self):
        return max(self.life / self.max_life, 0.0)


class HollowPurple:
    def __init__(self):
        self.state = "idle"
        self.charge_start_time = 0.0
        self.flick_start_time = 0.0
        self.orb_center = (0, 0)
        self.current_radius = 0.0
        self.prev_spread = None
        self.particles = []
        self.screen_w = WEBCAM_WIDTH
        self.screen_h = WEBCAM_HEIGHT
        self._last_update_time = time.time()

    def update_charging(self, landmarks, frame_w, frame_h):
        now = time.time()
        self.screen_w, self.screen_h = frame_w, frame_h
        if self.state == "idle":
            self.state = "charging"
            self.charge_start_time = now
            self.particles = [Particle(get_pinch_center(landmarks, frame_w, frame_h), frame_w, frame_h)
                               for _ in range(PARTICLE_COUNT)]

        self.orb_center = get_pinch_center(landmarks, frame_w, frame_h)

        if self.state == "charging":
            elapsed = now - self.charge_start_time
            t = min(elapsed / CHARGE_DURATION, 1.0)
            self.current_radius = ORB_MAX_RADIUS * (1 - (1 - t) ** 3)
            if t >= 1.0:
                self.state = "ready"
                self.current_radius = ORB_MAX_RADIUS

        elif self.state == "ready":
            self.current_radius = ORB_MAX_RADIUS

    def check_flick(self, landmarks):
        spread = pinch_spread(landmarks)
        is_flick = False
        if self.prev_spread is not None:
            velocity = spread - self.prev_spread
            if velocity > FLICK_VELOCITY_THRESHOLD:
                is_flick = True
        self.prev_spread = spread

        if is_flick and self.state in ("charging", "ready"):
            self.state = "flick_delay"
            self.flick_start_time = time.time()

    def abandon_charge(self):
        if self.state in ("charging", "ready"):
            self.state = "idle"
            self.current_radius = 0.0
        self.prev_spread = None
        self.particles = []

    def reset_flick_tracking(self):
        """Reset velocity tracking when pose is re-established"""
        self.prev_spread = None

    def update_and_draw(self, frame):
        now = time.time()
        dt = max(now - self._last_update_time, 0.0)
        dt = min(dt, 0.1)
        self._last_update_time = now

        if self.state == "idle":
            return frame

        if self.state in ("charging", "ready"):
            self._update_particles(dt, inward=True)
            anim_t = now - self.charge_start_time
            return self._draw_orb(frame, self.orb_center, self.current_radius,
                                   purple_amount=1.0, anim_t=anim_t)

        if self.state == "flick_delay":
            elapsed = now - self.flick_start_time
            if elapsed >= FLICK_DELAY_DURATION:
                self.state = "flicking"
                self.flick_start_time = now
            else:
                self._update_particles(dt, inward=True)
                anim_t = now - self.charge_start_time
                return self._draw_orb(frame, self.orb_center, ORB_MAX_RADIUS,
                                       purple_amount=1.0, anim_t=anim_t)

        if self.state == "flicking":
            elapsed = now - self.flick_start_time
            t = min(elapsed / FLICK_FLY_DURATION, 1.0)
            eased_t = t ** 2

            max_dim = max(frame.shape[0], frame.shape[1])
            radius = ORB_MAX_RADIUS + eased_t * (max_dim * 1.5 - ORB_MAX_RADIUS)
            purple_amount = max(1.0 - eased_t, 0.0)

            self._update_particles(dt, inward=False)
            anim_t = now - self.charge_start_time
            frame = self._draw_orb(frame, self.orb_center, radius, purple_amount, anim_t)

            if t >= 1.0:
                self.state = "flash"
                self.flick_start_time = now
                self.particles = []
            return frame

        if self.state == "flash":
            elapsed = now - self.flick_start_time
            if elapsed < FLASH_HOLD_DURATION:
                alpha = 1.0
            elif elapsed < FLASH_HOLD_DURATION + FLASH_FADE_DURATION:
                fade_t = (elapsed - FLASH_HOLD_DURATION) / FLASH_FADE_DURATION
                alpha = 1.0 - fade_t
            else:
                self.state = "idle"
                self.current_radius = 0.0
                self.prev_spread = None
                return frame

            white = np.full_like(frame, 255)
            frame = cv2.addWeighted(white, alpha, frame, 1 - alpha, 0)
            return frame

        return frame

    def _update_particles(self, dt, inward):
        for i, p in enumerate(self.particles):
            p.update(dt, self.orb_center, inward)
            if p.is_dead(self.orb_center, self.screen_w, self.screen_h):
                self.particles[i] = Particle(self.orb_center, self.screen_w, self.screen_h)

    def _draw_orb(self, frame, center, radius, purple_amount, anim_t):
        radius = max(int(radius), 1)
        cx, cy = center

        dark_color = lerp_color_np(PURPLE_CORE, WHITE, 1 - purple_amount)
        bright_color = lerp_color_np(PURPLE_GLOW, WHITE, 1 - purple_amount)
        glow_color = tuple(int(c) for c in bright_color)

        pulse = 1.0 + 0.06 * math.sin(anim_t * 6.0)
        glow_radius = radius * 2.2 * pulse
        very_dark_purple = np.array([50, 0, 40])
        outer_dark = lerp_color_np(very_dark_purple, WHITE, 1 - purple_amount)
        color_stops = [
            (0.0, dark_color),
            (0.35, bright_color),
            (0.75, dark_color),
            (1.0, outer_dark),
        ]
        gradient, alpha, (x0, y0, x1, y1) = make_radial_gradient_patch(
            frame.shape, center, glow_radius, color_stops)
        if gradient is not None:
            region = frame[y0:y1, x0:x1].astype(np.float32)
            blended = region * (1 - alpha) + gradient * alpha
            frame[y0:y1, x0:x1] = blended.astype(np.uint8)

        swirl_layers = [
            (radius * 1.05, 1.0, 90, max(int(radius * 0.035), 1)),
            (radius * 0.85, -1.4, 70, max(int(radius * 0.025), 1)),
            (radius * 0.65, 1.8, 50, max(int(radius * 0.018), 1)),
        ]
        overlay = frame.copy()
        for layer_radius, rot_speed, arc_span, thickness in swirl_layers:
            base_angle = math.degrees(anim_t * rot_speed)
            for offset in (0, 180):
                start_angle = base_angle + offset
                end_angle = start_angle + arc_span
                cv2.ellipse(overlay, (cx, cy), (int(layer_radius), int(layer_radius)),
                            0, start_angle, end_angle, glow_color, thickness,
                            lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        star_outer = radius * STAR_SCALE
        circle_color = lerp_color_np(bright_color, WHITE, 0.3)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), int(star_outer * 1.3),
                   tuple(int(c) for c in circle_color), -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        frame = draw_star(frame, (cx, cy), star_outer, inner_ratio=STAR_INNER_RATIO, points=8,
                           color=tuple(int(c) for c in WHITE), rotation=anim_t * 0.6)

        overlay = frame.copy()
        for p in self.particles:
            a = p.alpha()
            if a <= 0:
                continue
            px, py = p.position()
            size = max(int(p.size), 1)
            color = lerp_color(PURPLE_GLOW, WHITE, 1 - purple_amount)
            cv2.circle(overlay, (px, py), size, color, -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        return frame


# ---- UI drawing functions ----

def draw_finger_indicators(frame, hand_landmarks_list):
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
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 255, 255)

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

cv2.namedWindow("JJK Gestures", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("JJK Gestures", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Unlimited Void state
uv_active = False
uv_consecutive_detections = 0
uv_consecutive_absences = 0
uv_gesture_armed = True

# Hollow Purple state
hollow_purple = HollowPurple()
hp_pose_consecutive_frames = 0

# Segmentation state
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

    uv_gesture_this_frame = False
    hp_pose_this_frame = False
    hp_landmarks_this_frame = None
    any_hand_landmarks = None

    # Detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Store the first hand's landmarks for flick detection
            if any_hand_landmarks is None:
                any_hand_landmarks = landmarks

            # Check for Unlimited Void gesture
            if is_unlimited_void(landmarks):
                uv_gesture_this_frame = True

            # Check for Hollow Purple charging pose
            if is_charging_pose(landmarks):
                hp_pose_this_frame = True
                hp_landmarks_this_frame = landmarks

    # --- Unlimited Void debounced toggle logic ---
    if uv_gesture_this_frame:
        uv_consecutive_detections += 1
        uv_consecutive_absences = 0
    else:
        uv_consecutive_absences += 1
        uv_consecutive_detections = 0

    if uv_consecutive_absences >= UV_RELEASE_FRAMES:
        uv_gesture_armed = True

    if uv_gesture_armed and uv_consecutive_detections >= UV_DEBOUNCE_FRAMES:
        uv_active = not uv_active
        uv_gesture_armed = False
        if uv_active:
            bg_video.activate()
        else:
            bg_video.deactivate()

    # --- Unlimited Void compositing ---
    if uv_active:
        bg_frame = bg_video.get_frame((w, h))
        if bg_frame is not None:
            if frame_count % SEGMENT_EVERY_N_FRAMES == 0 or cached_mask is None:
                timestamp_ms = int((time.time() - start_time) * 1000)
                cached_mask = get_person_mask_small(frame, timestamp_ms)
            frame = composite(frame, bg_frame, cached_mask)

    frame_count += 1

    # --- Hollow Purple state machine ---
    if hollow_purple.state == "idle":
        if hp_pose_this_frame:
            hp_pose_consecutive_frames += 1
        else:
            hp_pose_consecutive_frames = 0

        if hp_pose_consecutive_frames >= HP_DEBOUNCE_FRAMES:
            hollow_purple.update_charging(hp_landmarks_this_frame, w, h)

    elif hollow_purple.state in ("charging", "ready"):
        hp_pose_consecutive_frames = 0
        if any_hand_landmarks is not None:
            # Always check for flick if we have hand landmarks
            hollow_purple.check_flick(any_hand_landmarks)
            if hollow_purple.state in ("charging", "ready"):
                # Only update position if still in charging pose
                if hp_pose_this_frame and hp_landmarks_this_frame is not None:
                    hollow_purple.update_charging(hp_landmarks_this_frame, w, h)
        else:
            # Hand left the frame entirely -> abandon
            hollow_purple.abandon_charge()
    else:
        hp_pose_consecutive_frames = 0

    # Hollow Purple animation (works over any background)
    frame = hollow_purple.update_and_draw(frame)

    # --- Finger indicator dots ---
    frame = draw_finger_indicators(frame, results.multi_hand_landmarks)

    # --- Gesture labels ---
    if uv_active:
        frame = draw_gesture_label(frame, "INFINITE VOID")
    if hollow_purple.state != "idle":
        # Draw HP label slightly higher to avoid overlap
        h_temp, w_temp = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (0, 255, 255)
        text = "HOLLOW PURPLE"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 15
        x = w_temp - text_w - margin
        y = h_temp - margin - 25  # Offset from bottom
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # --- FPS counter ---
    fps_frame_count += 1
    elapsed = time.time() - fps_timer_start
    if elapsed >= 0.5:
        fps_display = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_timer_start = time.time()

    # Debug overlay
    status = f"UV={uv_active} HP={hollow_purple.state} FPS={fps_display:.1f}"
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("JJK Gestures", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
bg_video.cap.release()
cv2.destroyAllWindows()
