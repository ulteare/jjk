
"""
Step 6 (v2): Hollow Purple - two-phase gesture with orb charge + flick-fire animation

Visual style:
- Deep/dark purple energy orb with a TRUE smooth radial gradient glow (dark -> bright
  -> dark, per-pixel interpolated, not stepped circles) plus rotating swirl arcs
- Sharp, long 8-pointed star core (glowing light) with a small center, slowly rotating
- Particles gather from across the ENTIRE screen, converging into the orb while
  charging, and scatter back outward during the flick

Phase 1 (CHARGING): thumb, index, middle fingertips touching, ring + pinky curled,
palm facing up. The orb appears at the touch point and grows over time, capping
at a max size (READY state) if you hold the pose without flicking.

Phase 2 (FLICK): index and middle fingers rapidly snap away from the thumb. This is
detected as a velocity spike (fast separation), not just "fingers no longer touching"
-- a slow relaxed-hand release should NOT trigger it.

On flick: the orb scales up rapidly and shifts purple -> white, ending in a full-screen
white flash that holds briefly then fades, auto-resetting back to idle (ready to charge
again immediately).

Requires:
    pip install mediapipe opencv-python numpy

Press 'q' to quit.
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

# ---- Config ----
GESTURE_LABEL = "HOLLOW PURPLE"

DEBOUNCE_FRAMES = 5          # frames the charging pose must hold before we start charging

# --- Orb charge timing ---
CHARGE_DURATION = 1.8        # seconds to go from 0 -> max radius while holding the pose
ORB_MAX_RADIUS = 70          # pixels, final size while in READY state

# --- Flick detection ---
FLICK_VELOCITY_THRESHOLD = 0.035   # normalized-coords/frame; fingertip separation speed to count as a flick
TOUCH_DISTANCE_THRESHOLD = 0.06    # normalized distance; how close index/middle/thumb must be to count as "touching"

# --- Flick/flash animation timing ---
FLICK_DELAY_DURATION = 1.0   # seconds to hold at max size after flick before flying
FLICK_FLY_DURATION = 0.35    # seconds for the orb to rocket toward the viewer
FLASH_HOLD_DURATION = 0.3    # seconds the full white screen holds
FLASH_FADE_DURATION = 0.4    # seconds for the flash to fade back to normal

# --- Particle system ---
PARTICLE_COUNT = 22

# --- Star core shape ---
STAR_INNER_RATIO = 0.075   # lower = sharper/thinner points
STAR_SCALE = 0.8        # relative to orb radius; higher = longer points, larger overall star

# --- Performance ---
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# --- Finger indicator config ---
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGERTIP_RADIUS = 12
FINGERTIP_COLOR = (255, 255, 255)
FINGERTIP_ALPHA = 0.5

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,   # Hollow Purple is single-handed; locking to 1 reduces ambiguity + cost
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


def is_finger_curled(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y > landmarks[pip_idx].y


def fingertip_distance(landmarks, idx_a, idx_b):
    a, b = landmarks[idx_a], landmarks[idx_b]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def get_pinch_center(landmarks, frame_w, frame_h):
    """Average position of thumb/index/middle tips, in pixel coords -- this is
    where the orb will be drawn."""
    xs = [landmarks[i].x for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)]
    ys = [landmarks[i].y for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)]
    cx = int((sum(xs) / 3) * frame_w)
    cy = int((sum(ys) / 3) * frame_h)
    return cx, cy


def is_palm_facing_up(landmarks):
    """
    Approximate orientation check using MediaPipe's relative z-depth.
    z is negative-ish toward the camera (smaller z = closer to camera) in MediaPipe's
    convention, relative to the wrist. When the palm faces the camera/up, the middle
    knuckle (MIDDLE_MCP) tends to sit closer to the camera than the wrist.
    NOTE: this is the least reliable check here -- MediaPipe Hands doesn't give true
    3D orientation. If this misfires on your setup, this is the first thing to retune
    or loosen/remove.
    """
    wrist_z = landmarks[WRIST].z
    middle_mcp_z = landmarks[MIDDLE_MCP].z
    return middle_mcp_z < wrist_z  # middle knuckle closer to camera than wrist


def is_charging_pose(landmarks):
    """Phase 1: thumb+index+middle touching, ring+pinky curled, palm up."""
    ring_curled = is_finger_curled(landmarks, RING_TIP, RING_PIP)
    pinky_curled = is_finger_curled(landmarks, PINKY_TIP, PINKY_PIP)

    thumb_index_close = fingertip_distance(landmarks, THUMB_TIP, INDEX_TIP) < TOUCH_DISTANCE_THRESHOLD
    thumb_middle_close = fingertip_distance(landmarks, THUMB_TIP, MIDDLE_TIP) < TOUCH_DISTANCE_THRESHOLD
    index_middle_close = fingertip_distance(landmarks, INDEX_TIP, MIDDLE_TIP) < TOUCH_DISTANCE_THRESHOLD

    fingers_pinched = thumb_index_close and thumb_middle_close and index_middle_close
    palm_up = is_palm_facing_up(landmarks)

    return ring_curled and pinky_curled and fingers_pinched and palm_up


def pinch_spread(landmarks):
    """Sum of pairwise distances between thumb/index/middle tips -- a single number
    that grows as the pinch opens up. Used to detect flick velocity."""
    return (fingertip_distance(landmarks, THUMB_TIP, INDEX_TIP) +
            fingertip_distance(landmarks, THUMB_TIP, MIDDLE_TIP) +
            fingertip_distance(landmarks, INDEX_TIP, MIDDLE_TIP))


# ---- Orb + flash animation ----

PURPLE_CORE = np.array([80, 0, 60])      # BGR darker deep purple
PURPLE_GLOW = np.array([110, 10, 80])    # BGR darker glow purple
WHITE = np.array([255, 255, 255])


def lerp_color(c1, c2, t):
    return tuple(int(c) for c in (c1 * (1 - t) + c2 * t))


def lerp_color_np(c1, c2, t):
    """Same as lerp_color but returns a float32 numpy array (for gradient math),
    not an int tuple (which is what cv2 drawing functions need)."""
    return (c1.astype(np.float32) * (1 - t) + c2.astype(np.float32) * t)


def make_radial_gradient_patch(shape, center, max_radius, color_stops):
    """
    Same idea as a full-frame radial gradient, but only computes the distance field
    and gradient over a square bounding box around `center` (sized to max_radius),
    not the entire frame. This is the expensive part of the orb glow, and the visible
    effect only ever covers a small region of the frame, so cropping the computation
    is a large performance win with no visual difference.

    Returns (patch_gradient, patch_alpha, (x0, y0, x1, y1)) where the bounds are the
    region of the original frame this patch corresponds to (already clamped to the
    frame's edges).
    """
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

    alpha = (np.clip(1.0 - norm_dist, 0, 1) ** 0.8)[..., None]  # Less fade, more visible
    return gradient, alpha, (x0, y0, x1, y1)


def make_radial_gradient(shape, center, max_radius, color_stops):
    """
    Builds an HxWx3 float32 image where each pixel's color is interpolated along
    color_stops based on its normalized distance from center (0=center, 1=max_radius).
    color_stops: list of (position 0-1, BGR np.array float32), sorted by position.
    Returns (gradient_image, normalized_distance_map).
    NOTE: kept for reference/prototyping; the live script uses make_radial_gradient_patch
    instead for performance (full-frame computation is far more expensive than needed).
    """
    h, w = shape[:2]
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm_dist = np.clip(dist / max_radius, 0, 1)

    gradient = np.zeros((h, w, 3), dtype=np.float32)
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

    return gradient, norm_dist


# Pre-generate star tip length variations (fixed pattern, doesn't affect other randomness)
_STAR_LENGTH_MULTIPLIERS = [1.8, 2.3, 1.2, 2.5, 1.5, 2.0, 1.1, 1.9]  # 8 points

def draw_star(frame, center, outer_radius, inner_ratio=0.4, points=8, color=(255, 255, 255), rotation=0.0):
    """Draws a filled N-pointed star (alternating outer/inner radius vertices) --
    used for the sharp glowing core instead of a plain circle. Some tips are randomly longer."""
    cx, cy = center
    angle_step = math.pi / points  # half-step since we alternate outer/inner each vertex
    vertices = []

    for i in range(points * 2):
        if i % 2 == 0:  # outer point
            point_index = i // 2
            r = outer_radius * _STAR_LENGTH_MULTIPLIERS[point_index]
        else:  # inner point
            r = outer_radius * inner_ratio
        angle = rotation + i * angle_step
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        vertices.append((int(x), int(y)))
    pts = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(frame, pts, color, lineType=cv2.LINE_AA)
    return frame


class Particle:
    """A particle that gathers from anywhere on screen, drifting/accelerating toward
    the orb center while charging. During the flick, particles scatter outward instead."""
    __slots__ = ("x", "y", "speed", "size", "life", "max_life")

    def __init__(self, center, screen_w, screen_h):
        cx, cy = center
        max_dist = math.hypot(screen_w, screen_h) / 2
        dist = max_dist * (0.3 + 0.7 * random.random())  # spread across the whole screen
        angle = random.uniform(0, 2 * math.pi)
        self.x = cx + dist * math.cos(angle)
        self.y = cy + dist * math.sin(angle)
        self.speed = random.uniform(120, 260)  # pixels/sec
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
                dirx, diry = -dirx, -diry  # flick: scatter away from center instead
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
    """
    States:
      idle        - nothing happening, waiting for charging pose
      charging    - orb growing at the pinch point, tracks hand position live
      ready       - orb at max size, holding, still tracks hand position, waiting for flick
      flick_delay - 1 second hold after flick detected, orb stays at max size
      flicking    - orb rockets toward viewer (scale + color animate), no longer tracks hand
      flash       - full white screen hold + fade, then auto-resets to idle
    """

    def __init__(self):
        self.state = "idle"
        self.charge_start_time = 0.0
        self.flick_start_time = 0.0
        self.orb_center = (0, 0)        # last known pinch point, frozen once flicking starts
        self.current_radius = 0.0
        self.prev_spread = None         # previous frame's pinch_spread(), for velocity calc
        self.particles = []
        self.screen_w = WEBCAM_WIDTH
        self.screen_h = WEBCAM_HEIGHT
        self._last_update_time = time.time()

    def update_charging(self, landmarks, frame_w, frame_h):
        """Call every frame while the charging pose is held."""
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
            # Ease-out growth, same shaping as the Unlimited Void portal felt good
            self.current_radius = ORB_MAX_RADIUS * (1 - (1 - t) ** 3)
            if t >= 1.0:
                self.state = "ready"
                self.current_radius = ORB_MAX_RADIUS

        elif self.state == "ready":
            self.current_radius = ORB_MAX_RADIUS

    def check_flick(self, landmarks):
        """Call every frame while in charging/ready state to detect the flick motion."""
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
        """Call when the charging pose breaks WITHOUT a flick (treated as abandoned)."""
        if self.state in ("charging", "ready"):
            self.state = "idle"
            self.current_radius = 0.0
        self.prev_spread = None
        self.particles = []

    def update_and_draw(self, frame):
        now = time.time()
        dt = max(now - self._last_update_time, 0.0)
        dt = min(dt, 0.1)  # clamp to avoid a huge jump if a frame stalls
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
                # Delay over, transition to flicking
                self.state = "flicking"
                self.flick_start_time = now  # Reset timer for the fly animation
            else:
                # Hold at max size during delay
                self._update_particles(dt, inward=True)
                anim_t = now - self.charge_start_time
                return self._draw_orb(frame, self.orb_center, ORB_MAX_RADIUS,
                                       purple_amount=1.0, anim_t=anim_t)

        if self.state == "flicking":
            elapsed = now - self.flick_start_time
            t = min(elapsed / FLICK_FLY_DURATION, 1.0)
            # Ease-in: starts slow, accelerates -- sells the "rushing toward you" feel
            eased_t = t ** 2

            # Radius grows from ORB_MAX_RADIUS to something far bigger than the screen
            max_dim = max(frame.shape[0], frame.shape[1])
            radius = ORB_MAX_RADIUS + eased_t * (max_dim * 1.5 - ORB_MAX_RADIUS)
            purple_amount = max(1.0 - eased_t, 0.0)  # shifts purple -> white as it grows

            self._update_particles(dt, inward=False)
            anim_t = now - self.charge_start_time
            frame = self._draw_orb(frame, self.orb_center, radius, purple_amount, anim_t)

            if t >= 1.0:
                self.state = "flash"
                self.flick_start_time = now  # reuse as flash_start_time
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
                # Done -- auto-reset back to idle, ready to charge again
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
        """
        Draws a swirling energy orb: a smooth radial gradient glow (dark -> bright -> dark,
        not stepped), rotating swirl arcs, and a sharp star core, plus orbiting particles.
        purple_amount=1.0 -> fully purple, 0.0 -> fully white (used during the flick).
        anim_t: seconds since this charge/flick began, drives swirl rotation + pulse.
        """
        radius = max(int(radius), 1)
        cx, cy = center

        dark_color = lerp_color_np(PURPLE_CORE, WHITE, 1 - purple_amount)
        bright_color = lerp_color_np(PURPLE_GLOW, WHITE, 1 - purple_amount)
        glow_color = tuple(int(c) for c in bright_color)

        # --- Smooth radial gradient glow: dark -> bright -> dark, true interpolation ---
        pulse = 1.0 + 0.06 * math.sin(anim_t * 6.0)
        glow_radius = radius * 2.2 * pulse
        # Even darker purple for the outer edge
        very_dark_purple = np.array([50, 0, 40])  # BGR very dark purple for outline
        outer_dark = lerp_color_np(very_dark_purple, WHITE, 1 - purple_amount)
        color_stops = [
            (0.0, dark_color),
            (0.35, bright_color),
            (0.75, dark_color),      # transition zone starts
            (1.0, outer_dark),       # smooth transition to very dark purple at edge
        ]
        gradient, alpha, (x0, y0, x1, y1) = make_radial_gradient_patch(
            frame.shape, center, glow_radius, color_stops)
        if gradient is not None:
            region = frame[y0:y1, x0:x1].astype(np.float32)
            blended = region * (1 - alpha) + gradient * alpha
            frame[y0:y1, x0:x1] = blended.astype(np.uint8)

        # --- Swirl arcs: 3 layers rotating at different speeds/directions ---
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

        # --- Sharp, long 8-pointed star core (glowing light), slowly rotating ---
        star_outer = radius * STAR_SCALE
        circle_color = lerp_color_np(bright_color, WHITE, 0.3)  # blend toward purple
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), int(star_outer * 1.3),
                   tuple(int(c) for c in circle_color), -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        frame = draw_star(frame, (cx, cy), star_outer, inner_ratio=STAR_INNER_RATIO, points=8,
                           color=tuple(int(c) for c in WHITE), rotation=anim_t * 0.6)

        # --- Particles ---
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

cv2.namedWindow("Hollow Purple", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hollow Purple", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

hollow_purple = HollowPurple()
pose_consecutive_frames = 0  # debounce counter for idle -> charging transition

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

    pose_this_frame = False
    landmarks_this_frame = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # single-hand gesture
        landmarks_this_frame = hand_landmarks.landmark
        if is_charging_pose(landmarks_this_frame):
            pose_this_frame = True

    # --- Drive the state machine ---
    # idle -> charging requires DEBOUNCE_FRAMES consecutive detections, to avoid
    # a single jittery frame kicking off a charge by accident.
    # Once charging/ready, every frame's pose check feeds the flick-velocity detector,
    # so we keep evaluating flick even on frames where the strict pose check might
    # momentarily blip false due to landmark noise.
    if hollow_purple.state == "idle":
        if pose_this_frame:
            pose_consecutive_frames += 1
        else:
            pose_consecutive_frames = 0

        if pose_consecutive_frames >= DEBOUNCE_FRAMES:
            hollow_purple.update_charging(landmarks_this_frame, w, h)

    elif hollow_purple.state in ("charging", "ready"):
        pose_consecutive_frames = 0  # reset; only relevant for the idle entry point
        if landmarks_this_frame is not None:
            hollow_purple.check_flick(landmarks_this_frame)
            if hollow_purple.state in ("charging", "ready"):  # check_flick may have changed it
                if pose_this_frame:
                    hollow_purple.update_charging(landmarks_this_frame, w, h)
                else:
                    hollow_purple.abandon_charge()
        else:
            # Hand left the frame entirely -> abandon
            hollow_purple.abandon_charge()
    else:
        pose_consecutive_frames = 0

    # flicking/flash states animate on their own via update_and_draw(), no input needed

    frame = hollow_purple.update_and_draw(frame)

    # --- Finger indicator dots ---
    frame = draw_finger_indicators(frame, results.multi_hand_landmarks)

    # --- Gesture name label (while charging/ready/flicking/flash) ---
    if hollow_purple.state != "idle":
        frame = draw_gesture_label(frame, GESTURE_LABEL)

    # --- FPS counter ---
    fps_frame_count += 1
    elapsed = time.time() - fps_timer_start
    if elapsed >= 0.5:  # update twice a second
        fps_display = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_timer_start = time.time()

    # Debug overlay (remove once happy)
    status = f"state={hollow_purple.state} radius={hollow_purple.current_radius:.0f} FPS={fps_display:.1f}"
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hollow Purple", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
