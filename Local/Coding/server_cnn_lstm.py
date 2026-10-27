"""
KDRT CCTV Detection Backend
============================
Flask + SocketIO server integrating Ensemble V5 TwoStageDetector
with the CCTV monitoring frontend.

Endpoints:
  - WebSocket: detection_update, frame, history_update
  - Static files served at /

Usage:
  python server.py
  python server.py --port 5000
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_skripsi_root():
    """Return the parent folder of this script (D:\\Sistem Skripsi)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_local_env():
    """Load key=value pairs from Credential/.env without overriding OS env."""
    env_path = os.path.join(get_skripsi_root(), "Credential", ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as error:
        print(f"[Env] Failed to load .env: {error}")


load_local_env()

import argparse
import csv
import json
import platform
import re
import subprocess
import tempfile
import time
import threading
import collections
import base64
import uuid
import queue
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, send_from_directory, Response, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore, storage, messaging

# ── Configuration ─────────────────────────────────────────────────────────────
STAGE1_THRESHOLD = float(os.environ.get("KDRT_STAGE1_THRESHOLD", "0.10"))
BUFFER_SIZE = 7
MIN_FIGHT_COUNT = 5
COOLDOWN_SECS = 30
MIN_PERSONS = 2           # Require at least 2 people for KDRT detection
MIN_KEYPOINT_CONF = 0.3   # Minimum avg keypoint confidence to trust pose
MIN_INTERACTION_SCORE = 0.35  # Min interaction intensity to run ensemble (dikembalikan agar lebih sensitif)
YOLO_INFER_SIZE = 416     # Resize for YOLO inference (smaller = faster)
MIN_BBOX_AREA_RATIO = 0.005  # Min bbox area as fraction of frame (0.5%) — filters screen-displayed people
MAX_PERSON_SIZE_RATIO = 4.0  # Max bbox area ratio between 2 persons — rejects TV/monitor ghosts

# ── Realtime-only Configuration (env-tunable; video upload path is unaffected) ──
REALTIME_STAGE1_THRESHOLD = float(os.environ.get("KDRT_REALTIME_STAGE1_THRESHOLD", "0.40"))
REALTIME_INFRARED_THRESHOLD = float(os.environ.get("KDRT_REALTIME_INFRARED_THRESHOLD", "0.02"))
INFRARED_MAX_SATURATION = float(os.environ.get("KDRT_INFRARED_MAX_SATURATION", "20.0"))
INFRARED_MAX_CHANNEL_SPREAD = float(os.environ.get("KDRT_INFRARED_MAX_CHANNEL_SPREAD", "15.0"))
YOLO_INFER_SIZE_RT = int(os.environ.get("KDRT_YOLO_INFER_SIZE_RT", str(YOLO_INFER_SIZE)))
TARGET_SAMPLE_RATE = float(os.environ.get("KDRT_TARGET_SAMPLE_RATE", "6.0"))

# Realtime passive-proximity guard. The model still runs for close interactions,
# but an alert vote is vetoed only when reliable pose observations repeatedly
# indicate no directed strike and no sustained physical contact.
REALTIME_INTERACTION_MIN_COUNT = int(os.environ.get("KDRT_REALTIME_INTERACTION_MIN_COUNT", "3"))
REALTIME_PERSON_MIN_COUNT = int(os.environ.get("KDRT_REALTIME_PERSON_MIN_COUNT", "3"))
PASSIVE_PROXIMITY_WINDOW = int(os.environ.get("KDRT_PASSIVE_PROXIMITY_WINDOW", "5"))
PASSIVE_PROXIMITY_MIN_COUNT = int(os.environ.get("KDRT_PASSIVE_PROXIMITY_MIN_COUNT", "3"))
PASSIVE_MIN_POSE_SUPPORT_RATIO = float(
    os.environ.get("KDRT_PASSIVE_MIN_POSE_SUPPORT_RATIO", "0.60")
)
PASSIVE_MAX_KEYPOINT_VIOLENCE = float(
    os.environ.get("KDRT_PASSIVE_MAX_KEYPOINT_VIOLENCE", "0.05")
)
PASSIVE_PROXIMITY_WINDOW = max(1, PASSIVE_PROXIMITY_WINDOW)
PASSIVE_PROXIMITY_MIN_COUNT = max(
    1, min(PASSIVE_PROXIMITY_WINDOW, PASSIVE_PROXIMITY_MIN_COUNT)
)
PASSIVE_MIN_POSE_SUPPORT_RATIO = min(1.0, max(0.0, PASSIVE_MIN_POSE_SUPPORT_RATIO))
PASSIVE_MAX_KEYPOINT_VIOLENCE = max(0.0, PASSIVE_MAX_KEYPOINT_VIOLENCE)

# ── Video Upload Configuration ────────────────────────────────────────────────
DARK_BRIGHTNESS_THRESHOLD = 65.0
DARK_SATURATION_THRESHOLD = 35.0
DARK_STAGE1_THRESHOLD = float(os.environ.get("KDRT_DARK_THRESHOLD", "0.10"))
GENERAL_STAGE1_THRESHOLD = float(os.environ.get("KDRT_GENERAL_THRESHOLD", "0.10"))
LOW_LIGHT_GENERAL_THRESHOLD = float(os.environ.get("KDRT_LOW_LIGHT_THRESHOLD", "0.10"))
MODEL_OVERRIDE_THRESHOLD = float(os.environ.get("KDRT_MODEL_OVERRIDE_THRESHOLD", "0.90"))
SHORT_ACTION_MIN_FIGHT_COUNT = 1
SHORT_ACTION_MAX_IOU = 0.18
SHORT_ACTION_MIN_LEG_VELOCITY = 0.18
GENERAL_MIN_AGGRESSIVE_MOTION = 0.18
GENERAL_MIN_INTERACTION_SCORE = 0.40
WALKING_LIKE_MAX_LEG_VELOCITY = 0.30
MIN_AGGRESSIVE_MOTION_SCORE = 0.035
STATIC_CONTACT_MAX_AVG_MOTION = 0.010
STATIC_CONTACT_MAX_PEAK_MOTION = 0.025
PASSIVE_OVERLAP_IOU = 0.18
STRONG_MODEL_OVERRIDE = 0.92

# ── Motion Guard for ModelOverride (FASE D) ───────────────────────────────────
# Suppresses ModelOverride bypass when motion is effectively zero.
# Catches "close keypoints + physical contact + no movement" false positives
# where the CNN-LSTM model outputs prob >= 0.90 despite no actual violence.
MOTION_GUARD_MAX_LEG_VELOCITY = float(os.environ.get("KDRT_MOTION_GUARD_MAX_LEG_VELOCITY", "0.02"))
MOTION_GUARD_MAX_CLOSING_SPEED = float(os.environ.get("KDRT_MOTION_GUARD_MAX_CLOSING_SPEED", "0.02"))

# ── Sparse Contact Guard for ModelOverride (FASE E) ───────────────────────────
# Suppresses ModelOverride bypass when physical contact is only sparsely
# detected across the 20-frame window (e.g., 1-4 frames of contact out of 20).
# Catches "close keypoints + brief/noise-induced contact + high prob" false
# positives where the model outputs prob >= 0.90 but there is no sustained
# physical interaction. Only active for realtime (detection_loop); upload path
# does not pass contact_density, so the guard is a no-op there.
CONTACT_DENSITY_THRESHOLD = int(os.environ.get("KDRT_CONTACT_DENSITY_THRESHOLD", "5"))

# ── Contact Scale Multiplier (FASE F) ─────────────────────────────────────────
# Controls how generous `check_physical_contact` is when determining if an
# extremity (wrist/ankle) is "touching" another person's core (head/torso/hips).
# Original value 2.5x was too permissive — triggered contact even when people
# were merely standing close (distance ~265px, threshold ~288px on 1080p frame).
# Lowering to 1.5x reduces false contact detections from 55% → 10% of frames,
# which cascades through SparseContactGuard to suppress FP "close = violence".
# Tune via env var if true positives (actual punches) start getting missed.
CONTACT_SCALE_MULTIPLIER = float(os.environ.get("KDRT_CONTACT_SCALE_MULTIPLIER", "1.5"))

# ── Keypoint Violence Gate (FASE G) ───────────────────────────────────────────
# Motion magnitude and directed keypoint movement provide diagnostic support
# for realtime predictions. They do not block the RGB CNN-LSTM when pose
# estimation is weak, missing, or temporally unstable.
VIOLENCE_MOTION_GATE_THRESHOLD = float(os.environ.get("KDRT_VIOLENCE_MOTION_GATE", "0.30"))
KEYPOINT_VIOLENCE_THRESHOLD = float(os.environ.get("KDRT_KEYPOINT_VIOLENCE_THRESHOLD", "0.10"))

# FASE H: Track N out of M keypoint-supported iterations for diagnostics.
# Final realtime alert stability is controlled by TwoStageDetector's buffer.
SUSTAINED_VIOLENCE_COUNT = int(os.environ.get("KDRT_SUSTAINED_VIOLENCE_COUNT", "3"))
MOMENTUM_WINDOW = int(os.environ.get("KDRT_MOMENTUM_WINDOW", "5"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
SKRIPSI_ROOT = get_skripsi_root()
PROJECT_ROOT = SKRIPSI_ROOT
CREDENTIAL_DIR = os.path.join(SKRIPSI_ROOT, "Credential")
MODEL_DIR = os.path.join(SKRIPSI_ROOT, "Model")
STATIC_DIR = os.path.join(SKRIPSI_ROOT, "Website")
SKRIPSI_TMP = os.path.join(tempfile.gettempdir(), 'kdrt_output')
OUTPUT_BASE_FOLDER = os.environ.get(
    "KDRT_OUTPUT_FOLDER",
    os.path.join(SKRIPSI_TMP, "CSV dan Gambar Fix", "4.4 Blom fix"),
)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'kdrt_tmp_uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PERFORMANCE_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, 'performance')
os.makedirs(PERFORMANCE_FOLDER, exist_ok=True)
QOS_FOLDER = os.environ.get(
    "KDRT_QOS_FOLDER",
    os.path.join(SKRIPSI_TMP, "Laporan Final", "QOS"),
)
os.makedirs(QOS_FOLDER, exist_ok=True)
TEST_METADATA_FOLDER = os.environ.get(
    "KDRT_TEST_METADATA_FOLDER",
    os.path.join(SKRIPSI_TMP, "Laporan Final", "Metadata Pengujian"),
)
os.makedirs(TEST_METADATA_FOLDER, exist_ok=True)
HEURISTIC_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, 'heuristic')
os.makedirs(HEURISTIC_FOLDER, exist_ok=True)
IMAGE_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, 'gambar')
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ── CCTV Recording Configuration ──────────────────────────────────────────────
# Continuous CCTV recording is DISABLED (no cctv_recordings/ folder is created).
# Kept here for reference only; per-alert evidence clips (AlertEvidenceRecorder)
# remain stored under EVIDENCE_FOLDER.
RECORDINGS_FOLDER = os.path.join(BASE_DIR, 'cctv_recordings')
RECORDING_FPS = 10
EVIDENCE_FOLDER = os.environ.get(
    "KDRT_EVIDENCE_FOLDER",
    os.path.join(SKRIPSI_TMP, "Laporan Final", "Evidence video"),
)
EVIDENCE_PRE_SECONDS = float(os.environ.get("KDRT_EVIDENCE_PRE_SECONDS", "5"))
EVIDENCE_POST_SECONDS = float(os.environ.get("KDRT_EVIDENCE_POST_SECONDS", "5"))
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

MODEL_FILES = [
    os.environ.get("KDRT_MODEL_PATH", os.path.join(MODEL_DIR, "hybrid_model.h5"))
]
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-pose.pt")

CNN_IMG_SIZE = 224
CNN_FEATURE_SIZE = 1280

NUM_FRAMES = 20
REALTIME_INTERACTION_MIN_COUNT = max(
    1, min(NUM_FRAMES, REALTIME_INTERACTION_MIN_COUNT)
)
REALTIME_PERSON_MIN_COUNT = max(1, min(NUM_FRAMES, REALTIME_PERSON_MIN_COUNT))
NUM_FEATURES = 70
CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6)
]
UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LOWER_BODY = [11, 12, 13, 14, 15, 16]

def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def is_low_light_frame(frame):
    """Detect dark/IR-like frames from brightness or low color saturation."""
    return measure_lighting_frame(frame)["is_low_light"]


def measure_lighting_frame(frame):
    """Return automatic brightness/saturation metrics and low-light flag."""
    if frame is None:
        return {
            "brightness": 0.0,
            "saturation": 0.0,
            "channel_spread": 0.0,
            "is_low_light": False,
            "is_infrared": False,
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean())
    saturation = float(hsv[:, :, 1].mean())
    channel_spread = float(
        (frame.max(axis=2).astype(np.float32) - frame.min(axis=2).astype(np.float32)).mean()
    )
    is_infrared = (
        saturation <= INFRARED_MAX_SATURATION
        and channel_spread <= INFRARED_MAX_CHANNEL_SPREAD
    )
    return {
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "channel_spread": round(channel_spread, 4),
        "is_low_light": brightness < DARK_BRIGHTNESS_THRESHOLD or saturation < DARK_SATURATION_THRESHOLD,
        "is_infrared": bool(is_infrared),
    }


def extract_keypoint_snapshot(result, max_people=4):
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    kps = result.keypoints.data.cpu().numpy()
    if len(kps) > max_people and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = np.argsort(areas)[-max_people:]
        kps = kps[top_indices]
    return kps


def _keypoint_person_center(person_keypoints):
    """Return a confidence-filtered pose center for temporal person matching."""
    if person_keypoints is None or len(person_keypoints) == 0:
        return None
    valid = person_keypoints[:, 2] >= MIN_KEYPOINT_CONF
    if not np.any(valid):
        return None
    return np.mean(person_keypoints[valid, :2], axis=0)


def match_keypoint_people(previous_keypoints, current_keypoints):
    """Match people between adjacent frames by nearest valid pose center."""
    if previous_keypoints is None or current_keypoints is None:
        return []

    previous_centers = [
        _keypoint_person_center(person) for person in previous_keypoints
    ]
    current_centers = [
        _keypoint_person_center(person) for person in current_keypoints
    ]
    available_current = {
        idx for idx, center in enumerate(current_centers) if center is not None
    }
    matched_pairs = []

    for previous_idx, previous_center in enumerate(previous_centers):
        if previous_center is None or not available_current:
            continue
        current_idx = min(
            available_current,
            key=lambda idx: float(
                np.sum((current_centers[idx] - previous_center) ** 2)
            ),
        )
        available_current.remove(current_idx)
        matched_pairs.append(
            (
                previous_idx,
                current_idx,
                previous_keypoints[previous_idx],
                current_keypoints[current_idx],
            )
        )

    return matched_pairs


def compute_pose_support_ratio(keypoint_window):
    """Measure how often at least two people have usable limb and core keypoints."""
    valid_window = [snapshot for snapshot in keypoint_window if snapshot is not None]
    if not valid_window:
        return 0.0

    core_indices = [0, 5, 6, 11, 12]
    extremity_indices = [9, 10, 15, 16]
    supported_frames = 0

    for snapshot in valid_window:
        reliable_people = 0
        for person in snapshot:
            core_conf = person[core_indices, 2]
            extremity_conf = person[extremity_indices, 2]
            if (
                np.count_nonzero(core_conf >= MIN_KEYPOINT_CONF) >= 3
                and np.count_nonzero(extremity_conf >= MIN_KEYPOINT_CONF) >= 2
            ):
                reliable_people += 1
        if reliable_people >= MIN_PERSONS:
            supported_frames += 1

    return float(supported_frames / max(len(keypoint_window), 1))


def is_passive_realtime_interaction(
    pose_support_ratio,
    keypoint_violence_score,
    contact_density,
    motion_context,
    kick_contact,
):
    """Return True only when multiple reliable signals describe a passive interaction."""
    contact_is_passive = (
        contact_density < CONTACT_DENSITY_THRESHOLD
        or motion_context["static_contact"]
        or motion_context["passive_overlap"]
    )
    return bool(
        pose_support_ratio >= PASSIVE_MIN_POSE_SUPPORT_RATIO
        and keypoint_violence_score < PASSIVE_MAX_KEYPOINT_VIOLENCE
        and contact_is_passive
        and not kick_contact
    )


def compute_temporal_motion_context(keypoint_window, frame_diag, interaction_detail, recent_contact):
    """Summarize limb motion so static contact/overlap is not treated as violence."""
    empty = {
        "avg_extremity_velocity": 0.0,
        "max_extremity_velocity": 0.0,
        "max_leg_velocity": 0.0,
        "closing_speed_norm": 0.0,
        "interaction_score": 0.0,
        "overlap_iou": 0.0,
        "aggressive_motion_score": 0.0,
        "motion_passed": False,
        "static_contact": False,
        "passive_overlap": False,
        "effective_threshold": STAGE1_THRESHOLD,
        "low_light": False,
    }
    valid_window = [k for k in keypoint_window if k is not None and len(k) > 0]
    if len(valid_window) < 2 or frame_diag <= 0:
        return empty

    limb_indices = [9, 10, 13, 14, 15, 16]
    leg_indices = [15, 16]
    limb_velocities = []
    leg_velocities = []

    for prev, cur in zip(valid_window[:-1], valid_window[1:]):
        for _, _, previous_person, current_person in match_keypoint_people(prev, cur):
            for kp_idx in limb_indices:
                if kp_idx >= len(previous_person) or kp_idx >= len(current_person):
                    continue
                px, py, pc = previous_person[kp_idx]
                cx, cy, cc = current_person[kp_idx]
                if pc < MIN_KEYPOINT_CONF or cc < MIN_KEYPOINT_CONF:
                    continue
                velocity = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) / frame_diag)
                limb_velocities.append(velocity)
                if kp_idx in leg_indices:
                    leg_velocities.append(velocity)

    avg_velocity = float(np.mean(limb_velocities)) if limb_velocities else 0.0
    max_velocity = float(np.max(limb_velocities)) if limb_velocities else 0.0
    max_leg_velocity = float(np.max(leg_velocities)) if leg_velocities else 0.0

    closing_speed = interaction_detail.get("closing_speed_px", 0.0) if interaction_detail else 0.0
    avg_diag = interaction_detail.get("avg_bbox_diagonal_px", 0.0) if interaction_detail else 0.0
    try:
        closing_norm = max(0.0, float(closing_speed) / max(float(avg_diag), 1.0))
    except (TypeError, ValueError):
        closing_norm = 0.0

    aggressive_score = max(
        (0.70 * max_velocity) + (0.30 * closing_norm),
        0.90 * max_leg_velocity,
    )
    overlap_iou = float(interaction_detail.get("overlap_iou", 0.0) or 0.0) if interaction_detail else 0.0
    interaction_score = float(interaction_detail.get("interaction_score", 0.0) or 0.0) if interaction_detail else 0.0
    static_contact = (
        recent_contact and
        avg_velocity <= STATIC_CONTACT_MAX_AVG_MOTION and
        max_velocity <= STATIC_CONTACT_MAX_PEAK_MOTION
    )
    passive_overlap = (
        overlap_iou >= PASSIVE_OVERLAP_IOU and
        max_velocity <= STATIC_CONTACT_MAX_PEAK_MOTION and
        closing_norm <= 0.02
    )

    return {
        "avg_extremity_velocity": round(avg_velocity, 5),
        "max_extremity_velocity": round(max_velocity, 5),
        "max_leg_velocity": round(max_leg_velocity, 5),
        "closing_speed_norm": round(closing_norm, 5),
        "interaction_score": round(interaction_score, 5),
        "overlap_iou": round(overlap_iou, 5),
        "aggressive_motion_score": round(float(aggressive_score), 5),
        "motion_passed": bool(aggressive_score >= MIN_AGGRESSIVE_MOTION_SCORE),
        "static_contact": bool(static_contact),
        "passive_overlap": bool(passive_overlap),
        "effective_threshold": STAGE1_THRESHOLD,
        "low_light": False,
    }


def compute_keypoint_violence_score(keypoint_window, frame_diag):
    """Check if extremities are moving TOWARD other persons' cores (FASE G/H).

    Unlike compute_temporal_motion_context which only measures velocity magnitude,
    this function also checks DIRECTION — is a wrist/ankle moving toward another
    person's head/torso/hips? This distinguishes:
    - Violence: fast arm motion directed at another person
    - Non-violence: arm motion but not aimed at anyone (waving, gesturing)

    FASE H fix: proximity is used as a BINARY GATE (close enough to hit?),
    not as a multiplier. Previous formula `velocity * alignment * proximity`
    inflated scores when people stood close — ANY arm movement scored high
    because proximity was always ~1.0. New formula: `velocity * alignment`
    only if within striking distance.

    Returns a score 0.0-1.0. Higher = more violence-like directed motion.
    """
    valid_window = [k for k in keypoint_window if k is not None and len(k) > 0]
    if len(valid_window) < 2 or frame_diag <= 0:
        return 0.0

    violence_scores = []
    extremity_indices = [9, 10, 15, 16]  # left/right wrist, left/right ankle
    core_indices = [0, 5, 6, 11, 12]     # nose, left/right shoulder, left/right hip
    # Striking distance: extremity must be within 15% of frame diagonal from target
    # to be considered capable of violence. Beyond this, motion is not a threat.
    STRIKING_DIST_FRAC = 0.15
    MIN_VIOLENT_VELOCITY = 0.12  # ~12% of frame diagonal per frame pair
    MIN_ALIGNMENT = 0.5          # must be moving at >60° toward target

    for prev, cur in zip(valid_window[:-1], valid_window[1:]):
        if min(len(prev), len(cur)) < 2:
            continue

        for _, attacker_idx, previous_attacker, current_attacker in match_keypoint_people(prev, cur):
            for extremity_idx in extremity_indices:
                if (
                    extremity_idx >= len(previous_attacker)
                    or extremity_idx >= len(current_attacker)
                ):
                    continue

                px, py, pc = previous_attacker[extremity_idx]
                cx, cy, cc = current_attacker[extremity_idx]
                if pc < MIN_KEYPOINT_CONF or cc < MIN_KEYPOINT_CONF:
                    continue

                # Velocity vector (normalized by frame diagonal)
                dx = cx - px
                dy = cy - py
                velocity = float(np.sqrt(dx ** 2 + dy ** 2) / frame_diag)
                if velocity < MIN_VIOLENT_VELOCITY:
                    continue  # too slow to be violent (waving/filter noise)

                # Check if moving TOWARD any other person's core
                for victim_idx in range(len(cur)):
                    if victim_idx == attacker_idx:
                        continue

                    for core_idx in core_indices:
                        if core_idx >= len(cur[victim_idx]):
                            continue
                        tx, ty, tc = cur[victim_idx][core_idx]
                        if tc < MIN_KEYPOINT_CONF:
                            continue

                        # Direction from extremity current position to target
                        dir_x = tx - cx
                        dir_y = ty - cy
                        dir_mag = float(np.sqrt(dir_x ** 2 + dir_y ** 2))
                        if dir_mag < 5:
                            continue

                        # Striking distance gate: only count if within reach
                        dist_frac = dir_mag / frame_diag
                        if dist_frac > STRIKING_DIST_FRAC:
                            continue  # too far to be a strike

                        # Alignment: cosine of angle between velocity and direction
                        vel_mag = float(np.sqrt(dx ** 2 + dy ** 2))
                        if vel_mag < 1:
                            continue
                        alignment = (dx * dir_x + dy * dir_y) / (vel_mag * dir_mag)
                        alignment = max(0.0, alignment)

                        if alignment < MIN_ALIGNMENT:
                            continue  # not directly aimed at target

                        # Score = velocity * alignment (NO proximity multiplier)
                        score = velocity * alignment
                        violence_scores.append(score)

    if not violence_scores:
        return 0.0

    return float(np.max(violence_scores))


def apply_decision_filters(probability, base_threshold, recent_contact, motion_context, low_light, kick_contact=False, contact_density=None):
    effective_threshold = LOW_LIGHT_GENERAL_THRESHOLD if low_light else GENERAL_STAGE1_THRESHOLD
    min_fight_count = MIN_FIGHT_COUNT
    decision_mode = "General"

    # ── Motion Guard for ModelOverride (FASE D) ────────────────────────────────
    # Suppress ModelOverride bypass when motion is effectively zero. Catches
    # false positives where keypoints are close + physical contact is detected
    # but there is no actual movement (e.g., two people standing close talking).
    # Conditions: low aggressive motion + zero leg velocity + zero closing speed
    # + no kick contact. Without this guard, model prob >= 0.90 bypasses all
    # motion filters via the ModelOverride short-circuit below.
    if (
        probability >= MODEL_OVERRIDE_THRESHOLD
        and motion_context["aggressive_motion_score"] < GENERAL_MIN_AGGRESSIVE_MOTION
        and motion_context["max_leg_velocity"] < MOTION_GUARD_MAX_LEG_VELOCITY
        and motion_context["closing_speed_norm"] < MOTION_GUARD_MAX_CLOSING_SPEED
        and not kick_contact
    ):
        return 0.0, effective_threshold, "Motion Guard", min_fight_count, "MotionGuardOverride"

    # ── Sparse Contact Guard for ModelOverride (FASE E) ────────────────────────
    # Suppress ModelOverride bypass when physical contact is only sparsely
    # detected in the 20-frame window. Catches false positives where brief or
    # noise-induced contact triggers recent_contact=True, allowing ModelOverride
    # to bypass the "No Contact" filter despite no sustained interaction.
    # Only active when caller passes contact_density (detection_loop does;
    # video_processing_loop does not — keeping upload behavior unchanged).
    if (
        contact_density is not None
        and probability >= MODEL_OVERRIDE_THRESHOLD
        and contact_density < CONTACT_DENSITY_THRESHOLD
        and not kick_contact
    ):
        return 0.0, effective_threshold, "Sparse Contact", min_fight_count, "SparseContactGuard"

    # A strongly confident domain-adapted model prediction is more reliable than
    # frame-local pose motion, which frequently underestimates fast or dark attacks.
    if probability >= MODEL_OVERRIDE_THRESHOLD:
        return probability, MODEL_OVERRIDE_THRESHOLD, "", SHORT_ACTION_MIN_FIGHT_COUNT, "ModelOverride"

    short_action = (
        low_light and
        probability >= DARK_STAGE1_THRESHOLD and
        motion_context["max_leg_velocity"] >= SHORT_ACTION_MIN_LEG_VELOCITY and
        motion_context["interaction_score"] >= MIN_INTERACTION_SCORE and
        motion_context["overlap_iou"] <= SHORT_ACTION_MAX_IOU and
        motion_context["motion_passed"] and
        kick_contact
    )
    if short_action:
        effective_threshold = DARK_STAGE1_THRESHOLD
        min_fight_count = SHORT_ACTION_MIN_FIGHT_COUNT
        decision_mode = "LowLightShortAction"
        return probability, effective_threshold, "", min_fight_count, decision_mode

    if probability > effective_threshold:
        if not recent_contact:
            return 0.0, effective_threshold, "No Contact", min_fight_count, decision_mode
        if motion_context["aggressive_motion_score"] < GENERAL_MIN_AGGRESSIVE_MOTION:
            return 0.0, effective_threshold, "Weak Motion", min_fight_count, decision_mode
        if motion_context["interaction_score"] < GENERAL_MIN_INTERACTION_SCORE:
            return 0.0, effective_threshold, "Weak Interaction", min_fight_count, decision_mode
        if (not low_light) and (not kick_contact) and motion_context["max_leg_velocity"] >= WALKING_LIKE_MAX_LEG_VELOCITY:
            return 0.0, effective_threshold, "Walking-like Motion", min_fight_count, decision_mode
        if motion_context["static_contact"]:
            return 0.0, effective_threshold, "Static Contact", min_fight_count, decision_mode
        if motion_context["passive_overlap"]:
            return 0.0, effective_threshold, "Passive Overlap", min_fight_count, decision_mode
        min_fight_count = SHORT_ACTION_MIN_FIGHT_COUNT
        if low_light:
            decision_mode = "LowLightGeneral"
        else:
            decision_mode = "ValidatedGeneral"

    return probability, effective_threshold, "", min_fight_count, decision_mode


def apply_runtime_action_policy(
    probability,
    effective_threshold,
    min_fight_count,
    decision_mode,
    filter_reason,
    recent_contact,
    contact_density,
    motion_context,
    keypoint_violence_score,
    latest_keypoint_violence_score,
    kick_contact,
    passive_sustained=False,
):
    """Extension point for model-specific runtime decision policies.

    The default server keeps the existing decision unchanged. A dedicated
    launcher may replace this function without duplicating the CCTV, upload,
    Firebase, evidence, and QoS pipelines.
    """
    return (
        probability,
        effective_threshold,
        min_fight_count,
        decision_mode,
        filter_reason,
    )


def build_video_display_probability(raw_probability, n_persons, interaction,
                                    has_contact, sequence_fill_ratio,
                                    threshold=STAGE1_THRESHOLD):
    """
    UI-only score for uploaded videos so percentage can move smoothly
    from 0-100 without changing the true alert decision logic.
    """
    person_score = clamp01(n_persons / float(MIN_PERSONS))
    interaction_score = clamp01(interaction / max(MIN_INTERACTION_SCORE * 1.5, 1e-6))
    contact_score = 1.0 if has_contact else 0.0
    readiness_score = clamp01(sequence_fill_ratio)

    stage1_score = (
        (0.18 * person_score) +
        (0.32 * interaction_score) +
        (0.20 * contact_score) +
        (0.30 * readiness_score)
    )

    margin = raw_probability - threshold
    model_score = 1.0 / (1.0 + np.exp(-8.0 * margin))

    if readiness_score < 1.0:
        return clamp01(stage1_score * 0.75)

    combined_score = max(stage1_score * 0.55, (0.35 * stage1_score) + (0.65 * model_score))
    if not has_contact:
        combined_score = min(combined_score, 0.72)

    return clamp01(combined_score)

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features_from_keypoints(kp_seq):
    """kp_seq: (20, 17, 3) -> Returns: (20, 70) feature array"""
    T, N, _ = kp_seq.shape
    features = []
    for t in range(T):
        kp = kp_seq[t]
        x, y, c = kp[:, 0], kp[:, 1], kp[:, 2]
        f = []
        f.extend(x.tolist())
        f.extend(y.tolist())
        f.append(c[UPPER_BODY].mean())
        f.append(c[LOWER_BODY].mean())
        f.append(c.mean())
        f.append(c.max())
        for (a, b) in CONNECTIONS:
            f.append(float(np.sqrt((x[a]-x[b])**2 + (y[a]-y[b])**2)))
        valid = c > 0.1
        if valid.sum() > 0:
            f.extend([x[valid].max()-x[valid].min(), y[valid].max()-y[valid].min(),
                      x[valid].mean(), y[valid].mean()])
        else:
            f.extend([0.0, 0.0, 0.0, 0.0])
        dx=x[6]-x[5]; dy=y[6]-y[5]; f.append(float(np.arctan2(dy,dx)))
        dx=x[12]-x[11]; dy=y[12]-y[11]; f.append(float(np.arctan2(dy,dx)))
        sx=(x[5]+x[6])/2-(x[11]+x[12])/2; sy=(y[5]+y[6])/2-(y[11]+y[12])/2
        f.append(float(np.arctan2(sy,sx)))
        dx=(x[5]+x[6])/2-x[0]; dy=(y[5]+y[6])/2-y[0]
        f.append(float(np.arctan2(dy,dx)))
        f.append(float(valid.sum()))
        if t > 0:
            prev = kp_seq[t-1]
            vel = np.sqrt((x-prev[:,0])**2 + (y-prev[:,1])**2)
            f.extend([vel.mean(), vel.max(), vel[UPPER_BODY].mean(),
                      vel[LOWER_BODY].mean(), vel[[9,10,15,16]].mean()])
        else:
            f.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        features.append(f[:70])
    return np.array(features, dtype=np.float32)


def extract_keypoints_from_yolo_result(result, w, h):
    """Extract normalized keypoints (17, 3) from YOLO pose result."""
    kp = np.zeros((17, 3), dtype=np.float32)
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        idx = 0
        if result.boxes is not None and len(result.boxes) > 0:
            idx = result.boxes.conf.argmax().item()
        arr = result.keypoints.data[idx].cpu().numpy()
        for j in range(min(17, len(arr))):
            kp[j, 0] = arr[j, 0] / w
            kp[j, 1] = arr[j, 1] / h
            kp[j, 2] = arr[j, 2]

        # Quality check: if avg confidence is too low, return zeros
        avg_conf = kp[:, 2].mean()
        if avg_conf < MIN_KEYPOINT_CONF:
            return np.zeros((17, 3), dtype=np.float32)

    return kp


def count_persons(result, frame_area=None):
    """Count number of real persons detected by YOLO.
    Filters out tiny bounding boxes and objects with low keypoint confidence."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0
        
    has_kps = result.keypoints is not None and result.keypoints.conf is not None
    boxes = result.boxes.xyxy.cpu().numpy()
    count = 0
    
    for i, b in enumerate(boxes):
        # 1. Area filter (ignore tiny people from TV screens)
        if frame_area and frame_area > 0:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < frame_area * MIN_BBOX_AREA_RATIO:
                continue
                
        # 2. Keypoint confidence filter (ignore curtains/ghost objects)
        if has_kps:
            kp_conf = result.keypoints.conf[i].cpu().numpy()
            if np.mean(kp_conf) < MIN_KEYPOINT_CONF:
                continue
                
        count += 1
        
    return count

def check_physical_contact(result, img_w, img_h, return_details=False):
    """
    Check if any person's extremities (wrists/ankles) are very close to another person's body.
    Returns True if physical contact is detected. This is a strict semantic check.
    Uses scale-invariant distance (shoulder width or bbox size) to avoid perspective overlap errors.
    """
    no_contact_detail = {
        "contact_person": "",
        "core_person": "",
        "extremity_keypoint": "",
        "core_keypoint": "",
        "contact_distance_px": "",
        "contact_threshold_px": "",
    }
    if result.keypoints is None or len(result.keypoints.data) < 2:
        return (False, no_contact_detail) if return_details else False
        
    kps = result.keypoints.data.cpu().numpy() # (N, 17, 3)
    
    # OPTIMIZATION: If > 4 people, only take the 4 largest (closest to the camera)
    # Avoids an O(N^2) loop bottleneck that chokes FPS during crowded scenes
    if len(kps) > 4 and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = np.argsort(areas)[-4:]
        kps = kps[top_indices]
    
    best_detail = no_contact_detail.copy()
    best_distance = float("inf")

    for i in range(len(kps)):
        for j in range(len(kps)):
            if i == j: continue
            
            # Estimate Person j's scale (Shoulder width)
            ls_x, ls_y, ls_c = kps[j, 5]
            rs_x, rs_y, rs_c = kps[j, 6]
            
            scale = img_w * 0.05 # default 5%
            if ls_c > MIN_KEYPOINT_CONF and rs_c > MIN_KEYPOINT_CONF:
                shoulder_width = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
                if shoulder_width > 10: 
                    # PHASE F: contact occurs when an extremity (wrist/ankle)
                    # falls within CONTACT_SCALE_MULTIPLIER x the victim's core shoulder width.
                    # Default 1.5x (previously 2.5x - too wide, triggered false contact
                    # when people were just standing close without actual contact).
                    scale = shoulder_width * CONTACT_SCALE_MULTIPLIER
            
            # Clip scale so it is neither too small (min 5% of screen) nor too large (max 15% of screen)
            # 15% of 640px = 96 pixels (the reach range for a strike landing in the 2D frame)
            scale = np.clip(scale, img_w * 0.05, img_w * 0.15)
            
            # Person i's extremities
            for limb_idx in [9, 10, 15, 16]: 
                if limb_idx >= len(kps[i]): continue
                lx, ly, lc = kps[i, limb_idx]
                if lc < MIN_KEYPOINT_CONF: continue
                
                # Person j's core (head to hips)
                for core_idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]:
                    if core_idx >= len(kps[j]): continue
                    cx, cy, cc = kps[j, core_idx]
                    if cc < MIN_KEYPOINT_CONF: continue
                    
                    dist = np.sqrt((lx - cx)**2 + (ly - cy)**2)
                    if dist < best_distance:
                        best_distance = dist
                        best_detail = {
                            "contact_person": int(i),
                            "core_person": int(j),
                            "extremity_keypoint": int(limb_idx),
                            "core_keypoint": int(core_idx),
                            "contact_distance_px": round(float(dist), 4),
                            "contact_threshold_px": round(float(scale), 4),
                        }
                    if dist < scale:
                        return (True, best_detail) if return_details else True
                        
    return (False, best_detail) if return_details else False


def compute_interaction_score(result, prev_boxes=None, frame_area=None, return_details=False):
    """Compute interaction intensity between closest pair of detected persons.
    Returns (score 0.0-1.0, current_boxes).
    Score components: proximity (0-0.5), velocity (0-0.25), overlap (0-0.25).
    Filters out screen-displayed people by bbox size checks & keypoint confidence.
    """
    empty_detail = {
        "proximity_score": 0.0,
        "velocity_score": 0.0,
        "overlap_score": 0.0,
        "interaction_score": 0.0,
        "center_distance_px": "",
        "avg_bbox_diagonal_px": "",
        "previous_distance_px": "",
        "closing_speed_px": "",
        "overlap_iou": 0.0,
        "interaction_pair": "",
    }
    if result.boxes is None or len(result.boxes) < 2:
        return (0.0, None, empty_detail) if return_details else (0.0, None)

    all_boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) x1,y1,x2,y2
    has_kps = result.keypoints is not None and result.keypoints.conf is not None

    # Filter: remove tiny bounding boxes AND invalid keypoints (ghost objects)
    valid_boxes = []
    for i, b in enumerate(all_boxes):
        # 1. Area filter
        if frame_area and frame_area > 0:
            area = (b[2]-b[0])*(b[3]-b[1])
            if area < frame_area * MIN_BBOX_AREA_RATIO:
                continue
                
        # 2. Keypoint filter
        if has_kps:
            kp_conf = result.keypoints.conf[i].cpu().numpy()
            if np.mean(kp_conf) < MIN_KEYPOINT_CONF:
                continue
                
        valid_boxes.append(b)

    boxes = np.array(valid_boxes)
    n = len(boxes)
    if n < 2:
        result_tuple = (0.0, boxes if len(boxes) > 0 else None)
        return (*result_tuple, empty_detail) if return_details else result_tuple

    # Find closest pair by center distance
    centers = np.array([(b[0]+b[2])/2 for b in boxes]), np.array([(b[1]+b[3])/2 for b in boxes])
    cx, cy = centers

    best_score = 0.0
    best_pair = (0, 1)
    min_dist = float('inf')

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            if dist < min_dist:
                min_dist = dist
                best_pair = (i, j)

    i, j = best_pair

    # Size ratio check: reject if one person is much larger than the other
    area_i = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
    area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
    if area_i > 0 and area_j > 0:
        size_ratio = max(area_i, area_j) / min(area_i, area_j)
        if size_ratio > MAX_PERSON_SIZE_RATIO:
            # One person is much bigger → smaller one is likely from a screen
            return (0.0, boxes, empty_detail) if return_details else (0.0, boxes)

    # 1. Proximity score (0-0.5): closer = higher
    # Normalize by average box diagonal
    diag_i = np.sqrt((boxes[i][2]-boxes[i][0])**2 + (boxes[i][3]-boxes[i][1])**2)
    diag_j = np.sqrt((boxes[j][2]-boxes[j][0])**2 + (boxes[j][3]-boxes[j][1])**2)
    avg_diag = (diag_i + diag_j) / 2
    if avg_diag > 0:
        proximity = max(0, 1.0 - min_dist / (avg_diag * 3))  # Close within 3x body size
    else:
        proximity = 0.0
    proximity_score = proximity * 0.5

    # 2. Velocity score (0-0.25): how fast they moved toward each other
    velocity_score = 0.0
    prev_dist = ""
    closing_speed = ""
    if prev_boxes is not None and len(prev_boxes) >= 2:
        prev_cx = np.array([(b[0]+b[2])/2 for b in prev_boxes])
        prev_cy = np.array([(b[1]+b[3])/2 for b in prev_boxes])
        # Match by closest previous box
        pi = min(range(len(prev_boxes)), key=lambda k: (prev_cx[k]-cx[i])**2 + (prev_cy[k]-cy[i])**2)
        pj = min(range(len(prev_boxes)), key=lambda k: (prev_cx[k]-cx[j])**2 + (prev_cy[k]-cy[j])**2)
        prev_dist = np.sqrt((prev_cx[pi]-prev_cx[pj])**2 + (prev_cy[pi]-prev_cy[pj])**2)
        closing_speed = prev_dist - min_dist  # Positive = getting closer
        if avg_diag > 0:
            velocity_score = max(0, min(0.25, (closing_speed / avg_diag) * 0.5))

    # 3. Overlap score (0-0.25): bounding box IoU
    x1 = max(boxes[i][0], boxes[j][0])
    y1 = max(boxes[i][1], boxes[j][1])
    x2 = min(boxes[i][2], boxes[j][2])
    y2 = min(boxes[i][3], boxes[j][3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_i = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
    area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
    union = area_i + area_j - inter
    iou = inter / union if union > 0 else 0
    overlap_score = min(0.25, iou * 0.5)

    total = proximity_score + velocity_score + overlap_score
    detail = {
        "proximity_score": round(float(proximity_score), 4),
        "velocity_score": round(float(velocity_score), 4),
        "overlap_score": round(float(overlap_score), 4),
        "interaction_score": round(float(total), 4),
        "center_distance_px": round(float(min_dist), 4),
        "avg_bbox_diagonal_px": round(float(avg_diag), 4),
        "previous_distance_px": round(float(prev_dist), 4) if prev_dist != "" else "",
        "closing_speed_px": round(float(closing_speed), 4) if closing_speed != "" else "",
        "overlap_iou": round(float(iou), 4),
        "interaction_pair": f"{i}-{j}",
    }
    return (round(float(total), 3), boxes, detail) if return_details else (round(float(total), 3), boxes)


# ── TwoStageDetector ──────────────────────────────────────────────────────────
class TwoStageDetector:
    def __init__(self, models, threshold=STAGE1_THRESHOLD,
                 buffer_size=BUFFER_SIZE, min_fight=MIN_FIGHT_COUNT,
                 cooldown=COOLDOWN_SECS):
        self.models = models
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.min_fight = min_fight
        self.cooldown = cooldown
        self.prediction_buffer = collections.deque(maxlen=buffer_size)
        self.last_alert_time = 0
        self.total_alerts = 0
        self.current_probability = 0.0
        self.display_probability = 0.0
        self.stage1_result = "Waiting..."
        self.stage2_result = "Waiting..."
        self.is_alert = False

    def predict_ensemble(self, cnn_features):
        X = cnn_features.reshape(1, NUM_FRAMES, CNN_FEATURE_SIZE)
        # Take the first model (we now use a single Keras CNN-LSTM model, not a 5-model ensemble anymore)
        model = self.models[0]
        prob = float(model.predict(X, verbose=0)[0][0])
        return prob

    def update(self, probability, threshold=None, min_fight=None):
        active_threshold = self.threshold if threshold is None else threshold
        required_fight = self.min_fight if min_fight is None else min_fight
        self.current_probability = probability
        is_fight = probability > active_threshold
        self.stage1_result = "FIGHT" if is_fight else "NonFight"
        self.prediction_buffer.append(is_fight)

        if len(self.prediction_buffer) >= required_fight:
            fight_count = sum(self.prediction_buffer)
            self.is_alert = fight_count >= required_fight
            if self.is_alert:
                now = time.time()
                if now - self.last_alert_time > self.cooldown:
                    self.last_alert_time = now
                    self.total_alerts += 1
                    self.stage2_result = "ALERT"
                else:
                    self.stage2_result = "ALERT (cooldown)"
            else:
                self.stage2_result = "Safe"
        else:
            self.is_alert = False
            self.stage2_result = f"Buffering ({len(self.prediction_buffer)}/{required_fight})"

    def update_display_probability(self, value, alpha=0.35):
        value = clamp01(value)
        self.display_probability = ((1.0 - alpha) * self.display_probability) + (alpha * value)


# ══════════════════════════════════════════════════════════════════════════════
# Flask App
# ══════════════════════════════════════════════════════════════════════════════
# STATIC_DIR is defined in the path configuration block above (points to
# SKRIPSI_ROOT/Website where index.html, app.js, style.css and assets live).

app = Flask(__name__, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("KDRT_SECRET_KEY", "kdrt-local-session-key")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Firebase Init ─────────────────────────────────────────────────────────────
FIREBASE_KEY = os.path.join(CREDENTIAL_DIR, "firebase-key.json")
cred = credentials.Certificate(FIREBASE_KEY)
firebase_app = firebase_admin.initialize_app(cred, {
    'storageBucket': None  # Will be set from firebase-key project info
})

# Auto-detect bucket name from project ID
project_id = cred.project_id
bucket_name = f"{project_id}.firebasestorage.app"  # Updated Firebase Storage URL format
firebase_app = None  # Reset

# Re-initialize with bucket
firebase_admin.delete_app(firebase_admin.get_app())
firebase_app = firebase_admin.initialize_app(cred, {
    'storageBucket': bucket_name
})

db = firestore.client()
bucket = storage.bucket()

print(f"[Firebase] Connected to project: {project_id}")
print(f"[Firebase] Storage bucket: {bucket_name}")

FCM_WEB_CONFIG = {
    'apiKey': os.environ.get('KDRT_FIREBASE_API_KEY', ''),
    'authDomain': os.environ.get('KDRT_FIREBASE_AUTH_DOMAIN', f'{project_id}.firebaseapp.com'),
    'projectId': os.environ.get('KDRT_FIREBASE_PROJECT_ID', project_id),
    'storageBucket': os.environ.get('KDRT_FIREBASE_STORAGE_BUCKET', bucket_name),
    'messagingSenderId': os.environ.get('KDRT_FIREBASE_MESSAGING_SENDER_ID', ''),
    'appId': os.environ.get('KDRT_FIREBASE_APP_ID', ''),
}
FCM_VAPID_KEY = os.environ.get('KDRT_FIREBASE_VAPID_KEY', '')


def validate_vapid_public_key(vapid_key):
    result = {
        'present': bool(vapid_key),
        'decoded_bytes': 0,
        'valid': False,
        'error': '',
    }
    if not vapid_key:
        return result
    try:
        padded = vapid_key + ('=' * ((4 - len(vapid_key) % 4) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode('ascii'))
        result['decoded_bytes'] = len(decoded)
        result['valid'] = len(decoded) == 65 and decoded[:1] == b'\x04'
    except Exception as error:
        result['error'] = str(error)
    return result


FCM_VAPID_STATUS = validate_vapid_public_key(FCM_VAPID_KEY)
FCM_ENABLED = all([
    FCM_WEB_CONFIG.get('apiKey'),
    FCM_WEB_CONFIG.get('messagingSenderId'),
    FCM_WEB_CONFIG.get('appId'),
    FCM_VAPID_STATUS['valid'],
])
print(f"[Firebase] Web push enabled: {FCM_ENABLED}")

AUTH_USERS = {
    os.environ.get("KDRT_ADMIN_USERNAME", "admin"): {
        "password": os.environ.get("KDRT_ADMIN_PASSWORD", "admin123"),
        "role": "admin",
    },
    os.environ.get("KDRT_USER_USERNAME", "user"): {
        "password": os.environ.get("KDRT_USER_PASSWORD", "user123"),
        "role": "user",
    },
}


def current_role():
    return session.get("role", "guest")


def is_admin_session():
    return current_role() == "admin"


def public_user_doc(doc_id, data):
    return {
        "id": doc_id,
        "username": data.get("username", ""),
        "role": data.get("role", "user"),
        "active": bool(data.get("active", True)),
        "created_by": data.get("created_by", ""),
    }


def get_firestore_user(username):
    if not username:
        return None, None
    docs = list(
        db.collection("app_users")
        .where("username", "==", username)
        .limit(1)
        .stream()
    )
    if not docs:
        return None, None
    doc = docs[0]
    return doc.id, doc.to_dict() or {}

# ── Global State ──────────────────────────────────────────────────────────────
yolo_model = None
ensemble_models = []
detectors = {}       # {camera_ip: TwoStageDetector}  — per-camera detector
active_streams = {}  # {camera_ip: thread_handle}
stop_flags = {}      # {camera_ip: threading.Event}
latest_frames_raw = {}   # {camera_ip: jpeg_bytes}  — for MJPEG raw stream
latest_frames_anno = {}  # {camera_ip: jpeg_bytes}  — for MJPEG annotated stream

# ── Video Mode State ───────────────────────────────────────────────────────────
video_stop_flags = {}    # {video_id: threading.Event}
video_detectors  = {}    # {video_id: TwoStageDetector}
video_frames_raw = {}    # {video_id: jpeg_bytes}  — raw frame for MJPEG
video_frames_anno = {}   # {video_id: jpeg_bytes}  — annotated frame for MJPEG
video_original_names = {}  # {video_id: original uploaded filename}

cnn_feature_extractor = None

def load_models():
    """Load YOLO, MobileNetV2 Extractor, and CNN-LSTM Model."""
    global yolo_model, ensemble_models, cnn_feature_extractor

    print("\n[Models] Loading YOLOv8 Pose (Gatekeeper)...")
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("[Models] YOLOv8 Pose loaded ✓")

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Load MobileNetV2 Headless
    print("[Models] Loading MobileNetV2 Feature Extractor...")
    from tensorflow.keras.applications import MobileNetV2
    cnn_feature_extractor = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=(CNN_IMG_SIZE, CNN_IMG_SIZE, 3)
    )
    print("[Models] MobileNetV2 loaded ✓")

    print("[Models] Loading CNN-LSTM Model...")
    for mf in MODEL_FILES:
        if os.path.exists(mf):
            try:
                # Load CNN-LSTM (it uses simple BCE, no custom object needed typically)
                m = tf.keras.models.load_model(mf)
                ensemble_models.append(m)
                print(f"  [OK] {os.path.basename(mf)}")
            except Exception as e:
                print(f"  [ERROR] Failed to load model: {e}")
                # Fallback to loading weights if architecture doesn't parse due to batch_shape mismatch
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
                    m = Sequential([
                        Input(shape=(NUM_FRAMES, CNN_FEATURE_SIZE)),
                        Masking(mask_value=0.0),
                        LSTM(128, return_sequences=True), Dropout(0.4),
                        LSTM(64, return_sequences=False), Dropout(0.4),
                        Dense(64, activation='relu'), Dense(1, activation='sigmoid')
                    ])
                    m.load_weights(mf)
                    ensemble_models.append(m)
                    print(f"  [OK] Successfully loaded via load_weights: {os.path.basename(mf)}")
                except Exception as ef:
                     print(f"  [FATAL] load_weights failed: {ef}")
        else:
            print(f"  [!!] Not found: {mf}")

    if not ensemble_models:
        print("[ERROR] No CNN-LSTM model loaded!")
        return False

    print(f"[Models] CNN-LSTM model loaded ✓")
    print(f"[Models] Threshold: {STAGE1_THRESHOLD}, Buffer: {BUFFER_SIZE}")
    print(f"[Models] Detectors created per-camera on demand\n")
    return True


def save_to_firebase(frame, camera_name, camera_ip, probability, source_type='realtime'):
    """Create a Firestore history row; evidence video is attached later."""
    try:
        now = datetime.now()
        safe_name = secure_filename(camera_name.replace(' ', '_')) or 'detection'
        event_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S-%f')}_{safe_name}_alert"

        # Save metadata to Firestore
        doc_data = {
            'event_name': event_name,
            'camera_name': camera_name,
            'camera_ip': camera_ip,
            'probability': round(float(probability), 4),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'date_str': now.strftime('%d - %m - %Y'),
            'time_str': now.strftime('%H:%M'),
            'screenshot_url': '',
            'filename': '',
            'source_type': source_type,
            'evidence_status': 'processing',
        }
        doc_ref = db.collection('kdrt_detections').add(doc_data)

        print(f"[Firebase] History row created: {event_name} (prob: {probability:.3f})")
        send_alert_push_async({
            'id': doc_ref[1].id,
            'title': 'Alert KDRT Terdeteksi',
            'body': f"{camera_name} - probabilitas {float(probability) * 100:.1f}%",
            'camera_name': camera_name,
            'camera_ip': camera_ip,
            'probability': round(float(probability), 4),
            'source_type': source_type,
            'history_id': doc_ref[1].id,
        })
        return {
            'id': doc_ref[1].id,
            'name': event_name,
            'time': now.strftime('%H:%M'),
            'date': now.strftime('%d - %m - %Y'),
            'url': '',
            'evidence_status': 'processing',
            'probability': round(float(probability), 4),
            'camera_name': camera_name,
            'source_type': source_type,
        }
    except Exception as e:
        print(f"[Firebase] Error: {e}")
        return None


def remove_fcm_token(token):
    if not token:
        return
    try:
        db.collection('fcm_tokens').document(token).delete()
        print("[FCM] Removed invalid token")
    except Exception as error:
        print(f"[FCM] Failed to remove token: {error}")


def send_alert_push(payload):
    if not FCM_ENABLED:
        return {'sent': 0, 'skipped': 'FCM config incomplete'}

    try:
        docs = list(db.collection('fcm_tokens').stream())
    except Exception as error:
        print(f"[FCM] Failed to read tokens: {error}")
        return {'sent': 0, 'error': str(error)}

    sent = 0
    for doc in docs:
        data = doc.to_dict() or {}
        token = data.get('token') or doc.id
        if not token:
            continue
        message = messaging.Message(
            token=token,
            notification=messaging.Notification(
                title=payload.get('title', 'Alert KDRT Terdeteksi'),
                body=payload.get('body', 'Sistem mendeteksi indikasi kekerasan.'),
            ),
            webpush=messaging.WebpushConfig(
                fcm_options=messaging.WebpushFCMOptions(link='/'),
                notification=messaging.WebpushNotification(
                    icon='/assets/Frame 9.png',
                    badge='/assets/Frame 9.png',
                    require_interaction=True,
                ),
            ),
            data={
                'type': 'kdrt_alert',
                'history_id': str(payload.get('history_id', '')),
                'camera_name': str(payload.get('camera_name', '')),
                'camera_ip': str(payload.get('camera_ip', '')),
                'probability': str(payload.get('probability', '')),
                'source_type': str(payload.get('source_type', '')),
            },
        )
        try:
            messaging.send(message)
            sent += 1
        except Exception as error:
            err_text = str(error)
            print(f"[FCM] Send failed: {err_text}")
            if 'registration-token-not-registered' in err_text or 'invalid-argument' in err_text:
                remove_fcm_token(token)
    if sent:
        print(f"[FCM] Alert notification sent to {sent} device(s)")
    return {'sent': sent}


def send_alert_push_async(payload):
    thread = threading.Thread(target=send_alert_push, args=(payload,), daemon=True)
    thread.start()


def attach_evidence_video_to_history(firestore_id, video_path, probability,
                                     duration_seconds, pre_seconds, post_seconds):
    """Upload an evidence clip and attach it to an existing Firestore history row."""
    if not video_path or not os.path.exists(video_path):
        return None

    filename = os.path.basename(video_path)
    local_url = f"/evidence_clip/{filename}"
    update_data = {
        'evidence_video_filename': filename,
        'evidence_video_local_path': video_path,
        'evidence_video_local_url': local_url,
        'evidence_video_duration_seconds': round(float(duration_seconds), 4),
        'evidence_pre_seconds': round(float(pre_seconds), 4),
        'evidence_post_seconds': round(float(post_seconds), 4),
        'evidence_probability': round(float(probability), 4),
        'evidence_saved_at': firestore.SERVER_TIMESTAMP,
        'evidence_status': 'ready',
    }

    try:
        blob = bucket.blob(f"evidence_clips/{filename}")
        blob.upload_from_filename(video_path, content_type=get_video_mimetype(filename))
        blob.make_public()
        update_data['evidence_video_url'] = blob.public_url
        print(f"[Firebase] Evidence video uploaded: {filename}")
    except Exception as error:
        print(f"[Firebase] Evidence upload failed, local file kept: {error}")

    if firestore_id:
        try:
            db.collection('kdrt_detections').document(firestore_id).update(update_data)
            print(f"[Firebase] Evidence linked to history: {firestore_id}")
        except Exception as error:
            print(f"[Firebase] Evidence metadata update failed: {error}")
    if firestore_id:
        socketio.emit('history_evidence_update', {
            'id': firestore_id,
            'evidence_video_url': update_data.get('evidence_video_url', local_url),
            'evidence_video_local_url': local_url,
            'evidence_video_download_url': f"/evidence_download/{filename}",
            'evidence_video_filename': filename,
            'evidence_video_duration_seconds': update_data['evidence_video_duration_seconds'],
            'evidence_status': 'ready',
        })
    return update_data


EVIDENCE_TASK_QUEUE = queue.Queue(maxsize=int(os.environ.get("KDRT_EVIDENCE_QUEUE_SIZE", "64")))
EVIDENCE_WORKER_COUNT = max(1, int(os.environ.get("KDRT_EVIDENCE_WORKERS", "1")))
_evidence_workers_started = False
_evidence_workers_lock = threading.Lock()


def mark_evidence_failed(firestore_id, reason):
    if not firestore_id:
        return
    try:
        db.collection('kdrt_detections').document(firestore_id).update({
            'evidence_status': 'failed',
            'evidence_error': str(reason)[:500],
            'evidence_saved_at': firestore.SERVER_TIMESTAMP,
        })
        socketio.emit('history_evidence_update', {
            'id': firestore_id,
            'evidence_status': 'failed',
            'evidence_error': str(reason)[:200],
        })
    except Exception as error:
        print(f"[EvidenceWorker] Failed to mark evidence error: {error}")


def _evidence_worker_loop(worker_index):
    while True:
        task = EVIDENCE_TASK_QUEUE.get()
        try:
            kind = task.get('kind')
            if kind == 'attach':
                attach_evidence_video_to_history(
                    firestore_id=task.get('firestore_id'),
                    video_path=task.get('video_path'),
                    probability=task.get('probability', 0.0),
                    duration_seconds=task.get('duration_seconds', 0.0),
                    pre_seconds=task.get('pre_seconds', EVIDENCE_PRE_SECONDS),
                    post_seconds=task.get('post_seconds', EVIDENCE_POST_SECONDS),
                )
            elif kind == 'upload_clip':
                video_path = task.get('video_path')
                evidence_path, evidence_duration = create_evidence_clip_from_video(
                    video_path=video_path,
                    output_name=task.get('output_name') or 'upload',
                    video_id=task.get('video_id') or 'video',
                    alert_second=task.get('alert_second', 0.0),
                    pre_seconds=task.get('pre_seconds', EVIDENCE_PRE_SECONDS),
                    post_seconds=task.get('post_seconds', EVIDENCE_POST_SECONDS),
                )
                if evidence_path:
                    attach_evidence_video_to_history(
                        firestore_id=task.get('firestore_id'),
                        video_path=evidence_path,
                        probability=task.get('probability', 0.0),
                        duration_seconds=evidence_duration,
                        pre_seconds=task.get('pre_seconds', EVIDENCE_PRE_SECONDS),
                        post_seconds=task.get('post_seconds', EVIDENCE_POST_SECONDS),
                    )
                else:
                    mark_evidence_failed(task.get('firestore_id'), 'Failed to create evidence clip from uploaded video')
            else:
                print(f"[EvidenceWorker] Unknown task kind: {kind}")
        except Exception as error:
            print(f"[EvidenceWorker-{worker_index}] Error: {error}")
            mark_evidence_failed(task.get('firestore_id'), error)
        finally:
            EVIDENCE_TASK_QUEUE.task_done()


def ensure_evidence_workers_started():
    global _evidence_workers_started
    if _evidence_workers_started:
        return
    with _evidence_workers_lock:
        if _evidence_workers_started:
            return
        for i in range(EVIDENCE_WORKER_COUNT):
            thread = threading.Thread(
                target=_evidence_worker_loop,
                args=(i + 1,),
                daemon=True,
                name=f"evidence-worker-{i + 1}",
            )
            thread.start()
        _evidence_workers_started = True
        print(f"[EvidenceWorker] Started {EVIDENCE_WORKER_COUNT} worker(s)")


def enqueue_evidence_task(task):
    ensure_evidence_workers_started()
    try:
        EVIDENCE_TASK_QUEUE.put_nowait(task)
        return True
    except queue.Full:
        print("[EvidenceWorker] Queue full; evidence task skipped")
        mark_evidence_failed(task.get('firestore_id'), 'Antrean evidence penuh')
        return False


def enqueue_evidence_attach(firestore_id, video_path, probability,
                            duration_seconds, pre_seconds, post_seconds):
    if not video_path:
        return False
    return enqueue_evidence_task({
        'kind': 'attach',
        'firestore_id': firestore_id,
        'video_path': video_path,
        'probability': probability,
        'duration_seconds': duration_seconds,
        'pre_seconds': pre_seconds,
        'post_seconds': post_seconds,
    })


def enqueue_upload_evidence_clip(video_path, output_name, video_id, alert_second,
                                 firestore_id, probability,
                                 pre_seconds=EVIDENCE_PRE_SECONDS,
                                 post_seconds=EVIDENCE_POST_SECONDS):
    return enqueue_evidence_task({
        'kind': 'upload_clip',
        'video_path': video_path,
        'output_name': output_name,
        'video_id': video_id,
        'alert_second': alert_second,
        'firestore_id': firestore_id,
        'probability': probability,
        'pre_seconds': pre_seconds,
        'post_seconds': post_seconds,
    })


class AsyncVideoRecorder:
    """Write CCTV frames in the background so live preview stays responsive."""

    def __init__(self, output_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Failed to create video file: {output_path}")

        self.output_path = output_path
        self.queue = queue.Queue(maxsize=max(5, int(fps * 2)))
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.writer.write(frame)
            self.queue.task_done()

    def release(self):
        self.running = False
        self.thread.join(timeout=5)
        self.writer.release()


def get_video_mimetype(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".webm":
        return "video/webm"
    return "video/mp4"


def create_browser_video_writer(output_stem, width, height, fps):
    """Create a browser-playable evidence writer, preferring H264 when available."""
    candidates = [
        (".mp4", "avc1", "MP4/H264-avc1"),
        (".mp4", "H264", "MP4/H264"),
        (".webm", "VP80", "WebM/VP8"),
        (".mp4", "mp4v", "MP4/mp4v"),
    ]
    for ext, fourcc_name, label in candidates:
        output_path = f"{output_stem}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height)))
        if writer.isOpened():
            print(f"[Evidence] Writer codec: {label}")
            return writer, output_path
        writer.release()
    return None, None


class AlertEvidenceRecorder:
    """Create per-alert clips containing frames before, during, and after alert."""

    def __init__(self, camera_name, camera_ip, width, height, fps,
                 pre_seconds=EVIDENCE_PRE_SECONDS, post_seconds=EVIDENCE_POST_SECONDS):
        self.camera_name = camera_name
        self.camera_ip = camera_ip
        self.width = int(width)
        self.height = int(height)
        self.fps = max(float(fps), 1.0)
        self.pre_seconds = max(float(pre_seconds), 0.0)
        self.post_seconds = max(float(post_seconds), 0.0)
        self.pre_buffer = collections.deque(maxlen=max(1, int(round(self.fps * self.pre_seconds))))
        self.active = None

    def add_frame(self, frame, timestamp, alert_active=False):
        if frame is None:
            return
        prepared = self._prepare_frame(frame)
        self.pre_buffer.append((timestamp, prepared.copy()))

        if not self.active:
            return

        if alert_active:
            self.active["end_time"] = max(self.active["end_time"], timestamp + self.post_seconds)

        self.active["writer"].write(prepared)
        self.active["frames_written"] += 1

        if timestamp >= self.active["end_time"]:
            self._finalize_active()

    def start_alert(self, probability, firestore_id=None):
        now = time.time()
        if self.active:
            self.active["end_time"] = max(self.active["end_time"], now + self.post_seconds)
            self.active["max_probability"] = max(self.active["max_probability"], float(probability))
            if firestore_id and not self.active.get("firestore_id"):
                self.active["firestore_id"] = firestore_id
            return self.active["path"]

        safe_camera = secure_filename(self.camera_name.replace(' ', '_')) or 'camera'
        safe_ip = secure_filename(str(self.camera_ip).replace(':', '_').replace('/', '_')) or 'source'
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        output_stem = os.path.join(EVIDENCE_FOLDER, f"{dt}_{safe_camera}_{safe_ip}_evidence")
        writer, output_path = create_browser_video_writer(output_stem, self.width, self.height, self.fps)
        if writer is None:
            print(f"[Evidence] Failed to create clip: {output_stem}")
            return None
        filename = os.path.basename(output_path)

        frames_written = 0
        for _, buffered_frame in list(self.pre_buffer):
            writer.write(buffered_frame)
            frames_written += 1

        self.active = {
            "path": output_path,
            "filename": filename,
            "writer": writer,
            "started_at": now,
            "end_time": now + self.post_seconds,
            "frames_written": frames_written,
            "max_probability": float(probability),
            "firestore_id": firestore_id,
        }
        print(f"[Evidence] Started: {output_path} (pre={self.pre_seconds:.1f}s, post={self.post_seconds:.1f}s)")
        return output_path

    def close(self):
        if self.active:
            self._finalize_active()

    def _prepare_frame(self, frame):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def _finalize_active(self):
        evidence = self.active
        self.active = None
        evidence["writer"].release()
        duration = evidence["frames_written"] / self.fps if self.fps > 0 else 0.0
        print(f"[Evidence] Saved: {evidence['path']} ({duration:.2f}s)")
        enqueue_evidence_attach(
            firestore_id=evidence.get("firestore_id"),
            video_path=evidence["path"],
            probability=evidence["max_probability"],
            duration_seconds=duration,
            pre_seconds=self.pre_seconds,
            post_seconds=self.post_seconds,
        )


class AsyncMJPEGEncoder:
    """Encode MJPEG frames in a background thread so the detection loop stays responsive.

    The main loop calls submit(frame, annotated) with the latest frame; this class
    keeps only the most recent submission and encodes it at the configured interval.
    Stale submissions are silently dropped (latest-wins) which is the correct
    behaviour for live MJPEG streaming.
    """

    def __init__(self, camera_ip, width, height, display_width, quality=65, interval=0.1):
        self.camera_ip = camera_ip
        self.display_width = display_width
        self.quality = quality
        self.interval = interval
        self.w = max(int(width), 1)
        self.h = max(int(height), 1)
        self.target_h = max(1, int(self.h * (display_width / self.w)))

        self._frame = None        # latest raw frame awaiting encode
        self._anno = None         # latest annotated frame awaiting encode
        self._lock = threading.Lock()
        self.running = True
        self.encode_times = []    # populated by encoder thread; merged into perf_stats on release
        self._thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._thread.start()

    def submit(self, frame, annotated=None):
        """Store the latest frame for encoding. Non-blocking; drops stale frames."""
        with self._lock:
            self._frame = frame
            if annotated is not None:
                self._anno = annotated

    def _encode_loop(self):
        last_emit = 0.0
        while self.running:
            now = time.time()
            if now - last_emit < self.interval:
                time.sleep(0.01)
                continue
            with self._lock:
                frame = self._frame
                anno = self._anno
                self._frame = None
                self._anno = None
            if frame is None:
                time.sleep(0.005)
                continue
            last_emit = time.time()
            t0 = time.perf_counter()
            try:
                small_raw = cv2.resize(frame, (self.display_width, self.target_h))
                _, jpg_raw = cv2.imencode('.jpg', small_raw, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                latest_frames_raw[self.camera_ip] = jpg_raw.tobytes()
                if anno is not None:
                    ah, aw = anno.shape[:2]
                    a_scale = self.display_width / aw
                    small_anno = cv2.resize(anno, (self.display_width, max(1, int(ah * a_scale))))
                    _, jpg_anno = cv2.imencode('.jpg', small_anno, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    latest_frames_anno[self.camera_ip] = jpg_anno.tobytes()
            except Exception:
                pass
            self.encode_times.append(time.perf_counter() - t0)

    def release(self):
        self.running = False
        self._thread.join(timeout=2)


def create_recording_writer(camera_name, camera_ip, width, height):
    """Create an async video recorder for the connected CCTV stream."""
    if width <= 0 or height <= 0:
        print(f"[Recording] Invalid frame size for {camera_name}: {width}x{height}")
        return None, None

    now = datetime.now()
    safe_camera = secure_filename(camera_name.replace(' ', '_')) or 'camera'
    safe_ip = secure_filename(str(camera_ip).replace(':', '_').replace('/', '_')) or 'source'
    filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_camera}_{safe_ip}.mp4"
    output_path = os.path.join(RECORDINGS_FOLDER, filename)

    try:
        writer = AsyncVideoRecorder(output_path, width, height, RECORDING_FPS)
    except RuntimeError:
        print(f"[Recording] Failed to create video file: {output_path}")
        return None, None

    print(f"[Recording] Started: {output_path}")
    return writer, output_path


# Force RTSP over TCP (prevents UDP packet loss → h264 errors)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;1000000|fflags;nobuffer"


class RTSPReader:
    """Dedicated thread for reading RTSP frames — prevents buffer overflow."""

    def __init__(self, source):
        self.source = source
        if isinstance(source, int):
            import sys
            if sys.platform == 'win32':
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.frame_id = 0

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        """Continuously read frames — always has the latest frame ready."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_id += 1

    def read(self):
        """Get the latest frame (non-blocking). Returns (ret, frame, frame_id)."""
        with self.lock:
            if self.frame is None:
                return self.ret, None, self.frame_id
            return self.ret, self.frame.copy(), self.frame_id

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()


def detection_loop(camera_ip, camera_name, stop_event, username='', password='', port='554', path='/Streaming/Channels/101'):
    """Background thread: RTSP → YOLO → Features → Ensemble → WebSocket.
    Runs continuously regardless of whether the frontend is monitoring."""
    global detectors

    # Ensure path starts with /
    if path and not path.startswith('/'):
        path = '/' + path

    # Determine video source
    if camera_ip.isdigit():
        source = int(camera_ip)  # Webcam index
    elif username and password:
        source = f"rtsp://{username}:{password}@{camera_ip}:{port}{path}"
    else:
        source = f"rtsp://{camera_ip}:{port}{path}"

    print(f"[Stream] Opening: {source} ({camera_name})")

    reader = RTSPReader(source)

    if not reader.is_opened():
        print(f"[Stream] Failed to open: {source}")
        socketio.emit('connection_error', {
            'ip': camera_ip,
            'error': f'Failed to connect to {source}'
        })
        reader.release()
        return

    w, h = reader.w, reader.h
    print(f"[Stream] Connected: {w}x{h} ({camera_name})")

    # Wait for first frame
    first_frame = None
    for _ in range(30):
        ret, first_frame, _ = reader.read()
        if ret:
            break
        time.sleep(0.1)

    if first_frame is not None:
        frame_h, frame_w = first_frame.shape[:2]
        if w <= 0 or h <= 0:
            w, h = frame_w, frame_h

    # Detect/measure camera FPS for frame-based YOLO throttle.
    # RTSP streams often report FPS=0, so we measure from frame_id rate.
    camera_fps = reader.fps if reader.fps > 1 else 0.0
    _, _, last_read_fid = reader.read()
    if camera_fps <= 1:
        start_fid = last_read_fid
        measure_start = time.time()
        time.sleep(0.5)
        _, _, end_fid = reader.read()
        elapsed = time.time() - measure_start
        measured = float(end_fid - start_fid) / max(elapsed, 0.1)
        camera_fps = measured if measured > 1 else 20.0
    YOLO_EVERY = max(1, int(round(camera_fps / TARGET_SAMPLE_RATE)))
    total_camera_frames = 0
    last_detect_frame = 0
    print(f"[Stream] FPS={camera_fps:.1f}, YOLO_EVERY={YOLO_EVERY} (target {TARGET_SAMPLE_RATE:.1f} samples/s, {camera_name})")

    # Notify frontend: CCTV is truly connected (green dot)
    socketio.emit('cctv_connected', {'ip': camera_ip, 'name': camera_name})

    # Performance settings (defined early so mjpeg_encoder and main loop can use them)
    FRAME_EMIT_INTERVAL = 0.1 # ~10 FPS to browser (reduced to lower GPU contention)
    DETECT_EMIT_INTERVAL = 0.3 # Detection status update rate
    DISPLAY_WIDTH = 640       # Resize width for MJPEG

    # Continuous CCTV recording is disabled: recording_writer stays None so the
    # main loop skips writing full recordings. Per-alert evidence clips still work.
    recording_writer, recording_path = None, None
    evidence_recorder = AlertEvidenceRecorder(camera_name, camera_ip, w, h, RECORDING_FPS)

    # B2: Async MJPEG encoder — moves cv2.resize + imencode off the detection loop.
    mjpeg_encoder = AsyncMJPEGEncoder(
        camera_ip=camera_ip,
        width=w,
        height=h,
        display_width=DISPLAY_WIDTH,
        quality=65,
        interval=FRAME_EMIT_INTERVAL,
    )

    # Create per-camera detector (or reuse existing)
    if camera_ip not in detectors:
        detectors[camera_ip] = TwoStageDetector(
            ensemble_models,
            threshold=REALTIME_STAGE1_THRESHOLD,
        )
    det = detectors[camera_ip]
    det.prediction_buffer.clear()
    det.current_probability = 0.0
    det.display_probability = 0.0
    det.is_alert = False
    det.stage1_result = "Waiting..."
    det.stage2_result = "Waiting..."

    keypoint_buffer = []
    contact_buffer = []
    cnn_buffer = []
    person_buffer = []
    interaction_buffer = []
    kick_contact_buffer = []
    violence_momentum_buffer = []  # FASE H: track sustained violence-like motion
    passive_proximity_buffer = []
    best_interaction_detail = None
    last_frame_emit = 0
    last_detect_emit = 0
    last_detect_time = 0
    last_alert_save = 0
    last_record_time = 0
    prev_boxes = None  
    yolo_process_count = 0
    current_plotted = None  # To hold the latest YOLO annotated frame

    # ── Diagnostics: heuristic CSV + performance metrics (mirrors video_processing_loop) ──
    # Pure logging — no effect on detection logic.
    safe_cam = secure_filename(camera_name.replace(' ', '_')) or 'camera'
    safe_ip = secure_filename(str(camera_ip).replace(':', '_').replace('/', '_')) or 'source'
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"realtime_{safe_cam}_{safe_ip}_{dt_str}"
    stream_started_at = time.perf_counter()
    heuristic_rows = []
    qos_rows = []
    metadata_rows = []
    realtime_csv_writer = None
    previous_qos_pipeline_ms = None
    current_ping_ms = ""
    last_ping_check = 0.0
    perf_stats = {
        "frame_w": w, "frame_h": h, "src_fps": camera_fps,
        "total_frames": 0, "processed_frames": 0, "yolo_every": YOLO_EVERY,
        "wall_time_seconds": 0.0, "wall_fps": 0.0, "active_pipeline_fps": 0.0,
        "yolo_times": [], "heuristic_times": [], "mobilenet_times": [],
        "lstm_times": [], "encode_times": [], "pipeline_times": [],
        "alert_decision_delay_times": [], "alert_save_delay_times": [],
        "firebase_saves": 0,
    }

    try:
        realtime_csv_writer = RealtimeCSVWriter(session_id)
        print(f"[QoS] Real-time CSV started: {realtime_csv_writer.qos_path}")
        print(f"[Metadata] Real-time CSV started: {realtime_csv_writer.metadata_path}")
    except Exception as csv_error:
        print(f"[CSV] Failed to start real-time writer: {csv_error}")

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # B3: Calibrate inference speed to verify YOLO_EVERY is achievable.
        # Runs 3 warm-up iterations (YOLO + MobileNetV2) and measures average
        # pipeline time. If achievable rate < target, prints a warning so the
        # user can adjust KDRT_TARGET_SAMPLE_RATE or KDRT_YOLO_INFER_SIZE_RT.
        calib_frame = None
        for _ in range(10):
            ret_c, calib_frame, _ = reader.read()
            if ret_c and calib_frame is not None:
                break
            time.sleep(0.05)
        if calib_frame is not None:
            calib_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                yolo_model.predict(calib_frame, imgsz=YOLO_INFER_SIZE_RT, verbose=False, conf=0.5)
                calib_rgb = cv2.cvtColor(calib_frame, cv2.COLOR_BGR2RGB)
                calib_resized = cv2.resize(calib_rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype(np.float32)
                calib_pre = preprocess_input(np.expand_dims(calib_resized, axis=0))
                cnn_feature_extractor.predict(calib_pre, verbose=0)
                calib_times.append(time.perf_counter() - t0)
            avg_calib = sum(calib_times) / len(calib_times)
            achievable_rate = 1.0 / avg_calib if avg_calib > 0 else 0.0
            window_sec = NUM_FRAMES / max(achievable_rate, 0.1)
            print(f"[Stream] Calibration: avg pipeline={avg_calib*1000:.0f}ms, "
                  f"achievable={achievable_rate:.1f} samples/s, "
                  f"20-frame window={window_sec:.2f}s (training=3.33s)")
            if achievable_rate < TARGET_SAMPLE_RATE * 0.8:
                print(f"[Stream] WARNING: achievable rate ({achievable_rate:.1f}/s) is below "
                      f"target ({TARGET_SAMPLE_RATE:.1f}/s). Consider lowering "
                      f"KDRT_TARGET_SAMPLE_RATE or KDRT_YOLO_INFER_SIZE_RT.")

        while not stop_event.is_set():
            ret, frame, fid = reader.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            now = time.time()

            # Continuous CCTV recording is disabled (recording_writer is None), so
            # full recordings are not saved. Per-alert evidence buffering still runs.
            if frame.shape[1] != w or frame.shape[0] != h:
                record_frame = cv2.resize(frame, (w, h))
            else:
                record_frame = frame
            evidence_recorder.add_frame(record_frame, now, alert_active=det.is_alert)

            if recording_writer is not None and now - last_record_time >= (1.0 / RECORDING_FPS):
                last_record_time = now
                recording_writer.write(record_frame)

            # ── Frame counting for uniform temporal sampling ──
            # Frame-count-based throttle ensures uniform temporal spacing matching
            # training data (20 frames span ~3.33s at 6 FPS sampling rate).
            frames_passed = fid - last_read_fid
            if frames_passed <= 0:
                time.sleep(0.005)
                continue
            last_read_fid = fid
            total_camera_frames += frames_passed

            # ── YOLO Tracking + Detection (Frame-based ~6 FPS) ──
            if total_camera_frames - last_detect_frame >= YOLO_EVERY:
                last_detect_frame = total_camera_frames
                last_detect_time = now
                yolo_process_count += 1
                pipeline_start = time.perf_counter()

                # B1: pass the original frame to YOLO with imgsz= directly.
                # This avoids the previous "cv2.resize to 416 then YOLO upscales
                # back to 640" double-interpolation that cost ~30% extra YOLO time.
                # Ultralytics' internal letterbox preserves aspect ratio correctly.
                yolo_elapsed = 0.0
                heuristic_elapsed = 0.0
                mobilenet_elapsed = 0.0
                lstm_elapsed = 0.0
                alert_decision_delay = 0.0
                alert_save_delay = 0.0
                yolo_start = time.perf_counter()
                results = yolo_model.predict(frame, imgsz=YOLO_INFER_SIZE_RT, verbose=False, conf=0.5)
                yolo_elapsed = time.perf_counter() - yolo_start
                perf_stats["yolo_times"].append(yolo_elapsed)
                yolo_area = h * w  # frame area for bbox filtering (original resolution)
                heuristic_start = time.perf_counter()
                n_persons = count_persons(results[0], frame_area=yolo_area)

                # Compute interaction score between persons
                interaction, cur_boxes, interaction_detail = compute_interaction_score(
                    results[0],
                    prev_boxes,
                    frame_area=yolo_area,
                    return_details=True
                )
                prev_boxes = cur_boxes
                if (
                    best_interaction_detail is None or
                    interaction_detail["interaction_score"] > best_interaction_detail["interaction_score"]
                ):
                    best_interaction_detail = interaction_detail.copy()
                keypoint_buffer.append(extract_keypoint_snapshot(results[0]))

                # Save the annotated frame from YOLO
                current_plotted = results[0].plot()

                # Diagnostic logging (every 10 seconds = 60 YOLO checks)
                if yolo_process_count % 60 == 0:
                    raw_count = len(results[0].boxes) if results[0].boxes is not None else 0
                    confs = results[0].boxes.conf.cpu().numpy().tolist() if raw_count > 0 else []
                    print(f"[Diag] {camera_name}: raw_boxes={raw_count}, filtered_persons={n_persons}, "
                          f"interaction={interaction:.3f}, confs={[f'{c:.2f}' for c in confs]}, "
                          f"alert={det.is_alert}, prob={det.current_probability:.3f}")

                # Check absolute physical contact (use original frame dimensions)
                contact, contact_detail = check_physical_contact(results[0], w, h, return_details=True)
                kick_contact = contact and str(contact_detail.get("extremity_keypoint", "")) in {"15", "16"}
                contact_buffer.append(contact)
                person_buffer.append(n_persons)
                interaction_buffer.append(interaction)
                kick_contact_buffer.append(kick_contact)
                heuristic_elapsed = time.perf_counter() - heuristic_start
                perf_stats["heuristic_times"].append(heuristic_elapsed)

                lighting_metrics = measure_lighting_frame(frame)
                low_light = lighting_metrics["is_low_light"]
                infrared_mode = lighting_metrics["is_infrared"]

                # Extract CNN features if people are present; otherwise feed a zero mask frame to save computation
                mobilenet_start = time.perf_counter()
                if n_persons > 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype(np.float32)
                    frame_pre = preprocess_input(np.expand_dims(frame_resized, axis=0))
                    cnn_feat = cnn_feature_extractor.predict(frame_pre, verbose=0)[0]
                else:
                    cnn_feat = np.zeros((CNN_FEATURE_SIZE,), dtype=np.float32)
                mobilenet_elapsed = time.perf_counter() - mobilenet_start
                perf_stats["mobilenet_times"].append(mobilenet_elapsed)
                    
                cnn_buffer.append(cnn_feat)
                sequence_fill_ratio = min(1.0, len(cnn_buffer) / float(NUM_FRAMES))
                recent_contact = any(contact_buffer[-NUM_FRAMES:]) if contact_buffer else False
                # PHASE E: contact_density = number of contact frames within the 20-frame window.
                # Used by SparseContactGuard in apply_decision_filters to
                # suppress ModelOverride when contact is only sparse (noise-induced).
                recent_contact_density = sum(contact_buffer[-NUM_FRAMES:]) if contact_buffer else 0
                frame_diag = float(np.sqrt((w ** 2) + (h ** 2)))
                motion_context = compute_temporal_motion_context(
                    keypoint_buffer[-NUM_FRAMES:],
                    frame_diag,
                    interaction_detail,
                    recent_contact,
                )
                # FASE G: Compute keypoint violence score — checks if extremities
                # are moving TOWARD another person's core (direction-aware, not
                # just velocity magnitude). Distinguishes punching from waving.
                keypoint_violence_score = compute_keypoint_violence_score(
                    keypoint_buffer[-NUM_FRAMES:], frame_diag
                )
                latest_keypoint_violence_score = compute_keypoint_violence_score(
                    keypoint_buffer[-2:], frame_diag
                )
                pose_support_ratio = compute_pose_support_ratio(
                    keypoint_buffer[-NUM_FRAMES:]
                )
                interaction_positive_count = sum(
                    score >= MIN_INTERACTION_SCORE
                    for score in interaction_buffer[-NUM_FRAMES:]
                )
                person_positive_count = sum(
                    count >= MIN_PERSONS
                    for count in person_buffer[-NUM_FRAMES:]
                )
                passive_candidate = False
                passive_sustained = False
                effective_threshold = (
                    REALTIME_INFRARED_THRESHOLD if infrared_mode else det.threshold
                )
                min_fight_count = MIN_FIGHT_COUNT
                decision_mode = "General"
                motion_context["low_light"] = bool(low_light)
                motion_context["infrared"] = bool(infrared_mode)
                model_probability = 0.0
                filter_reason = ""
                new_alert_triggered = False

                # Ensemble prediction when buffer is full
                if len(cnn_buffer) >= NUM_FRAMES:
                    recent_persons = max(person_buffer[-NUM_FRAMES:], default=n_persons)
                    recent_interaction = max(interaction_buffer[-NUM_FRAMES:], default=interaction)
                    recent_kick_contact = any(kick_contact_buffer[-NUM_FRAMES:]) if kick_contact_buffer else False
                    persons_sustained = (
                        person_positive_count >= REALTIME_PERSON_MIN_COUNT
                    )
                    interaction_sustained = (
                        interaction_positive_count >= REALTIME_INTERACTION_MIN_COUNT
                    )
                    violence_shortcut = (
                        recent_kick_contact
                        or keypoint_violence_score >= KEYPOINT_VIOLENCE_THRESHOLD
                    )
                    if persons_sustained and (interaction_sustained or violence_shortcut):
                        # Keypoint and motion remain supporting evidence. They
                        # only veto a vote when a reliable passive pattern is sustained.
                        frame_has_violence = (
                            motion_context["aggressive_motion_score"] >= VIOLENCE_MOTION_GATE_THRESHOLD
                            and keypoint_violence_score >= KEYPOINT_VIOLENCE_THRESHOLD
                        )
                        violence_momentum_buffer.append(frame_has_violence)
                        if len(violence_momentum_buffer) > MOMENTUM_WINDOW:
                            violence_momentum_buffer.pop(0)
                        sustained_count = sum(violence_momentum_buffer)

                        # Always run CNN-LSTM after temporal interaction is confirmed.
                        cnn_seq = np.array(cnn_buffer[-NUM_FRAMES:], dtype=np.float32)
                        lstm_start = time.perf_counter()
                        probability = det.predict_ensemble(cnn_seq)
                        lstm_elapsed = time.perf_counter() - lstm_start
                        perf_stats["lstm_times"].append(lstm_elapsed)
                        model_probability = probability

                        passive_candidate = is_passive_realtime_interaction(
                            pose_support_ratio,
                            keypoint_violence_score,
                            recent_contact_density,
                            motion_context,
                            recent_kick_contact,
                        )
                        passive_proximity_buffer.append(passive_candidate)
                        if len(passive_proximity_buffer) > PASSIVE_PROXIMITY_WINDOW:
                            passive_proximity_buffer.pop(0)
                        passive_sustained = (
                            len(passive_proximity_buffer) >= PASSIVE_PROXIMITY_MIN_COUNT
                            and sum(passive_proximity_buffer) >= PASSIVE_PROXIMITY_MIN_COUNT
                        )

                        if passive_sustained:
                            probability = 0.0
                            decision_mode = "PassiveProximity"
                            filter_reason = "Passive Proximity"
                        elif not frame_has_violence:
                            decision_mode = "InfraredRGBFallback" if infrared_mode else "RGBFallback"
                            filter_reason = ""
                        elif sustained_count < SUSTAINED_VIOLENCE_COUNT:
                            decision_mode = "InfraredRGBFallback" if infrared_mode else "RGBFallback"
                            filter_reason = ""
                        else:
                            decision_mode = (
                                "InfraredRGBKeypointSupport"
                                if infrared_mode else "RGBKeypointSupport"
                            )
                        effective_threshold = (
                            REALTIME_INFRARED_THRESHOLD if infrared_mode else det.threshold
                        )
                        
                        # Realtime fallback uses the trained RGB model directly.
                        # Pose/contact heuristics remain diagnostic-only here.
                        motion_context["effective_threshold"] = effective_threshold
                    else:
                        # Not enough people or not interacting → safe
                        probability = 0.0
                        passive_proximity_buffer.clear()
                        if persons_sustained:
                            filter_reason = "Insufficient Temporal Interaction"
                            decision_mode = "TemporalInteractionGuard"

                        # Reset only when the whole temporal window has no valid interaction.
                        # This keeps stream decisions stable when YOLO misses people briefly.
                        if recent_persons < MIN_PERSONS:
                            det.prediction_buffer.clear()
                            det.is_alert = False
                            det.current_probability = 0.0
                            det.stage1_result = "No interaction"
                            det.stage2_result = "Safe"
                            cnn_buffer.clear()
                            contact_buffer.clear()
                            keypoint_buffer.clear()
                            person_buffer.clear()
                            interaction_buffer.clear()
                            kick_contact_buffer.clear()
                            violence_momentum_buffer.clear()
                            passive_proximity_buffer.clear()
                            best_interaction_detail = None

                    (
                        probability,
                        effective_threshold,
                        min_fight_count,
                        decision_mode,
                        filter_reason,
                    ) = apply_runtime_action_policy(
                        probability=probability,
                        effective_threshold=effective_threshold,
                        min_fight_count=min_fight_count,
                        decision_mode=decision_mode,
                        filter_reason=filter_reason,
                        recent_contact=recent_contact,
                        contact_density=recent_contact_density,
                        motion_context=motion_context,
                        keypoint_violence_score=keypoint_violence_score,
                        latest_keypoint_violence_score=latest_keypoint_violence_score,
                        kick_contact=recent_kick_contact,
                        passive_sustained=passive_sustained,
                    )
                    previous_total_alerts = det.total_alerts
                    det.update(probability, threshold=effective_threshold, min_fight=min_fight_count)
                    new_alert_triggered = det.total_alerts > previous_total_alerts
                    if new_alert_triggered:
                        alert_decision_delay = time.perf_counter() - pipeline_start
                        perf_stats["alert_decision_delay_times"].append(alert_decision_delay)
                    if len(cnn_buffer) > NUM_FRAMES:
                        cnn_buffer.pop(0)
                        contact_buffer.pop(0)
                        keypoint_buffer.pop(0)
                        person_buffer.pop(0)
                        interaction_buffer.pop(0)
                        kick_contact_buffer.pop(0)

                    # KDRT alert → save to Firebase (always, regardless of monitoring)
                    if det.is_alert and now - last_alert_save > COOLDOWN_SECS:
                        last_alert_save = now
                        alert_save_start = time.perf_counter()
                        result = save_to_firebase(frame, camera_name, camera_ip, probability)
                        perf_stats["firebase_saves"] += 1
                        evidence_path = evidence_recorder.start_alert(
                            probability,
                            firestore_id=result.get('id') if result else None,
                        )
                        if result:
                            result['evidence_video_local_path'] = evidence_path
                            if evidence_path:
                                result['evidence_video_local_url'] = f"/evidence_clip/{os.path.basename(evidence_path)}"
                                result['evidence_video_download_url'] = f"/evidence_download/{os.path.basename(evidence_path)}"
                                result['evidence_video_filename'] = os.path.basename(evidence_path)
                            socketio.emit('history_update', result)
                        alert_save_delay = time.perf_counter() - alert_save_start
                        perf_stats["alert_save_delay_times"].append(alert_save_delay)

                display_probability = build_video_display_probability(
                    raw_probability=det.current_probability,
                    n_persons=n_persons,
                    interaction=interaction,
                    has_contact=recent_contact,
                    sequence_fill_ratio=sequence_fill_ratio,
                    threshold=effective_threshold
                )
                det.update_display_probability(display_probability)

                # Emit detection status (throttled) — always broadcast
                if now - last_detect_emit > DETECT_EMIT_INTERVAL:
                    last_detect_emit = now
                    pct = int(round(float(det.display_probability) * 100))
                    is_kdrt = bool(det.is_alert)
                    socketio.emit('detection_update', {
                        'ip': camera_ip,
                        'camera_name': camera_name,
                        'status': 'KDRT' if is_kdrt else 'Normal',
                        'probability': round(float(det.display_probability), 4),
                        'raw_probability': round(float(det.current_probability), 4),
                        'percentage': pct,
                        'is_alert': is_kdrt,
                        'persons': n_persons,
                        'interaction': interaction,
                        'low_light': bool(low_light),
                        'aggressive_motion': motion_context["aggressive_motion_score"],
                        'decision_mode': decision_mode,
                        'stage1': str(det.stage1_result),
                        'stage2': str(det.stage2_result),
                    })

                # ── Heuristic detail row (one per YOLO iteration; matches video_processing_loop) ──
                samples_per_second = camera_fps / max(YOLO_EVERY, 1)
                fight_count = int(sum(det.prediction_buffer))
                cooldown_remaining = 0.0
                if det.last_alert_time > 0:
                    cooldown_remaining = max(0.0, det.cooldown - (now - det.last_alert_time))
                closing_speed_px = interaction_detail["closing_speed_px"]
                heuristic_rows.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "video_id": session_id,
                    "frame": total_camera_frames,
                    "second": round(total_camera_frames / max(camera_fps, 1), 4),
                    "persons": n_persons,
                    "min_person_passed": n_persons >= MIN_PERSONS,
                    "proximity_score": interaction_detail["proximity_score"],
                    "velocity_score": interaction_detail["velocity_score"],
                    "overlap_score": interaction_detail["overlap_score"],
                    "interaction_score": interaction_detail["interaction_score"],
                    "interaction_threshold": MIN_INTERACTION_SCORE,
                    "interaction_passed": interaction >= MIN_INTERACTION_SCORE,
                    "interaction_positive_count": int(interaction_positive_count),
                    "person_positive_count": int(person_positive_count),
                    "center_distance_px": interaction_detail["center_distance_px"],
                    "avg_bbox_diagonal_px": interaction_detail["avg_bbox_diagonal_px"],
                    "previous_distance_px": interaction_detail["previous_distance_px"],
                    "closing_speed_px": closing_speed_px,
                    "closing_velocity_px_per_second": (
                        round(float(closing_speed_px) * samples_per_second, 4)
                        if closing_speed_px != "" else ""
                    ),
                    "overlap_iou": interaction_detail["overlap_iou"],
                    "interaction_pair": interaction_detail["interaction_pair"],
                    "physical_contact": bool(contact),
                    "contact_person": contact_detail["contact_person"],
                    "core_person": contact_detail["core_person"],
                    "extremity_keypoint": contact_detail["extremity_keypoint"],
                    "core_keypoint": contact_detail["core_keypoint"],
                    "contact_distance_px": contact_detail["contact_distance_px"],
                    "contact_threshold_px": contact_detail["contact_threshold_px"],
                    "low_light": bool(low_light),
                    "avg_extremity_velocity": motion_context["avg_extremity_velocity"],
                    "max_extremity_velocity": motion_context["max_extremity_velocity"],
                    "max_leg_velocity": motion_context["max_leg_velocity"],
                    "avg_extremity_velocity_per_second": round(
                        motion_context["avg_extremity_velocity"] * samples_per_second, 5
                    ),
                    "max_extremity_velocity_per_second": round(
                        motion_context["max_extremity_velocity"] * samples_per_second, 5
                    ),
                    "max_leg_velocity_per_second": round(
                        motion_context["max_leg_velocity"] * samples_per_second, 5
                    ),
                    "closing_speed_norm": motion_context["closing_speed_norm"],
                    "aggressive_motion_score": motion_context["aggressive_motion_score"],
                    "keypoint_violence_score": round(float(keypoint_violence_score), 5),
                    "pose_support_ratio": round(float(pose_support_ratio), 5),
                    "passive_candidate": bool(passive_candidate),
                    "passive_sustained": bool(passive_sustained),
                    "motion_passed": motion_context["motion_passed"],
                    "static_contact": motion_context["static_contact"],
                    "passive_overlap": motion_context["passive_overlap"],
                    "effective_threshold": round(float(effective_threshold), 4),
                    "decision_mode": decision_mode,
                    "kick_contact": bool(kick_contact),
                    "model_probability": round(float(model_probability), 4),
                    "filter_reason": filter_reason,
                    "raw_probability": round(float(det.current_probability), 4),
                    "display_probability": round(float(det.display_probability), 4),
                    "buffer_fight_count": fight_count,
                    "buffer_size": len(det.prediction_buffer),
                    "buffer_requirement": det.min_fight,
                    "alert": bool(det.is_alert),
                    "cooldown_remaining_seconds": round(float(cooldown_remaining), 4),
                    "new_alert_triggered": bool(new_alert_triggered),
                    "total_alerts": int(det.total_alerts),
                    "stage1": str(det.stage1_result),
                    "stage2": str(det.stage2_result),
                })
                pipeline_elapsed = time.perf_counter() - pipeline_start
                perf_stats["pipeline_times"].append(pipeline_elapsed)
                perf_stats["processed_frames"] += 1
                wall_elapsed = max(time.perf_counter() - stream_started_at, 1e-6)
                active_elapsed = max(sum(perf_stats["pipeline_times"]), 1e-6)
                pipeline_ms = round(pipeline_elapsed * 1000.0, 4)
                if time.perf_counter() - last_ping_check >= 5.0:
                    current_ping_ms = ping_host_ms(camera_ip)
                    last_ping_check = time.perf_counter()
                encoded_frame_bytes = (
                    len(latest_frames_raw.get(camera_ip, b"")) +
                    len(latest_frames_anno.get(camera_ip, b""))
                )
                qos_metrics = compute_qos_network_metrics(
                    "realtime",
                    pipeline_ms,
                    previous_qos_pipeline_ms,
                    encoded_bytes=encoded_frame_bytes,
                    elapsed_seconds=pipeline_elapsed,
                    ping_ms=current_ping_ms,
                )
                previous_qos_pipeline_ms = pipeline_ms
                qos_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_id": session_id,
                    "source_type": "realtime",
                    "frame": total_camera_frames,
                    "second": round(total_camera_frames / max(camera_fps, 1), 4),
                    "source_fps": round(float(camera_fps), 4),
                    "yolo_every": YOLO_EVERY,
                    "processed_frames": perf_stats["processed_frames"],
                    "wall_fps": round(total_camera_frames / wall_elapsed, 4),
                    "active_pipeline_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                    "yolo_ms": round(yolo_elapsed * 1000.0, 4),
                    "heuristic_ms": round(heuristic_elapsed * 1000.0, 4),
                    "mobilenet_ms": round(mobilenet_elapsed * 1000.0, 4),
                    "cnn_lstm_ms": round(lstm_elapsed * 1000.0, 4),
                    "encode_ms": "",
                    "pipeline_ms": pipeline_ms,
                    "alert_decision_delay_ms": round(alert_decision_delay * 1000.0, 4),
                    "alert_save_delay_ms": round(alert_save_delay * 1000.0, 4),
                    "is_alert": bool(det.is_alert),
                    "new_alert_triggered": bool(new_alert_triggered),
                    "decision_mode": decision_mode,
                }
                qos_row.update(qos_metrics)
                qos_rows.append(qos_row)
                metadata_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_id": session_id,
                    "source_type": "realtime",
                    "source_name": camera_name,
                    "camera_ip": camera_ip,
                    "frame": total_camera_frames,
                    "second": round(total_camera_frames / max(camera_fps, 1), 4),
                    "resolution": f"{w}x{h}",
                    "source_fps": round(float(camera_fps), 4),
                    "target_sample_rate": round(float(TARGET_SAMPLE_RATE), 4),
                    "actual_sample_rate": round(float(camera_fps) / max(YOLO_EVERY, 1), 4),
                    "sequence_window_seconds": round(NUM_FRAMES / max(float(camera_fps) / max(YOLO_EVERY, 1), 0.1), 4),
                    "processed_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                    "yolo_every": YOLO_EVERY,
                    "base_threshold": round(float(det.threshold), 4),
                    "threshold": round(float(effective_threshold), 4),
                    "general_threshold": round(float(GENERAL_STAGE1_THRESHOLD), 4),
                    "low_light_threshold": round(float(LOW_LIGHT_GENERAL_THRESHOLD), 4),
                    "model_override_threshold": round(float(MODEL_OVERRIDE_THRESHOLD), 4),
                    "min_persons": MIN_PERSONS,
                    "min_interaction_score": MIN_INTERACTION_SCORE,
                    "buffer_capacity": det.buffer_size,
                    "buffer_size": len(det.prediction_buffer),
                    "buffer_requirement": min_fight_count,
                    "buffer_fight_count": int(sum(det.prediction_buffer)),
                    "persons": n_persons,
                    "min_person_passed": n_persons >= MIN_PERSONS,
                    "interaction_score": round(float(interaction), 4),
                    "interaction_threshold": MIN_INTERACTION_SCORE,
                    "interaction_passed": interaction >= MIN_INTERACTION_SCORE,
                    "interaction_positive_count": int(interaction_positive_count),
                    "person_positive_count": int(person_positive_count),
                    "physical_contact": bool(contact),
                    "contact_density": int(recent_contact_density),
                    "brightness": lighting_metrics["brightness"],
                    "saturation": lighting_metrics["saturation"],
                    "brightness_threshold": round(float(DARK_BRIGHTNESS_THRESHOLD), 4),
                    "saturation_threshold": round(float(DARK_SATURATION_THRESHOLD), 4),
                    "low_light": bool(low_light),
                    "keypoint_violence_score": round(float(keypoint_violence_score), 5),
                    "aggressive_motion_score": motion_context["aggressive_motion_score"],
                    "pose_support_ratio": round(float(pose_support_ratio), 5),
                    "passive_candidate": bool(passive_candidate),
                    "passive_sustained": bool(passive_sustained),
                    "raw_probability": round(float(det.current_probability), 4),
                    "display_probability": round(float(det.display_probability), 4),
                    "model_probability": round(float(model_probability), 4),
                    "decision_mode": decision_mode,
                    "filter_reason": filter_reason,
                    "stage1": str(det.stage1_result),
                    "stage2": str(det.stage2_result),
                    "final_decision": "KDRT" if det.is_alert else "Normal",
                    "is_alert": bool(det.is_alert),
                    "new_alert_triggered": bool(new_alert_triggered),
                }
                metadata_rows.append(metadata_row)
                if realtime_csv_writer is not None:
                    try:
                        realtime_csv_writer.append(qos_row, metadata_row)
                    except Exception as csv_error:
                        print(f"[CSV] Real-time append failed: {csv_error}")
                        try:
                            realtime_csv_writer.close()
                        except Exception:
                            pass
                        realtime_csv_writer = None

            # ── Submit frame for async MJPEG streaming ──
            # B2: encoding happens in AsyncMJPEGEncoder thread; main loop just hands off.
            if now - last_frame_emit > FRAME_EMIT_INTERVAL:
                last_frame_emit = now
                mjpeg_encoder.submit(frame, current_plotted)

    except Exception as e:
        print(f"[Stream] Error: {e}")
    finally:
        if realtime_csv_writer is not None:
            try:
                realtime_csv_writer.close()
            except Exception as csv_error:
                print(f"[CSV] Failed to close real-time writer: {csv_error}")
        evidence_recorder.close()
        if recording_writer is not None:
            recording_writer.release()
            print(f"[Recording] Saved: {recording_path}")
        # B2: stop MJPEG encoder thread and merge its timings into perf_stats
        mjpeg_encoder.release()
        perf_stats["encode_times"].extend(mjpeg_encoder.encode_times)
        reader.release()
        # Clean up frame buffer
        if camera_ip in latest_frames_raw:
            del latest_frames_raw[camera_ip]
        if camera_ip in latest_frames_anno:
            del latest_frames_anno[camera_ip]
        # Flush diagnostic CSVs (matches video_processing_loop output schema)
        try:
            perf_stats["wall_time_seconds"] = time.perf_counter() - stream_started_at
            perf_stats["total_frames"] = total_camera_frames
            perf_stats["processed_frames"] = yolo_process_count
            if perf_stats["wall_time_seconds"] > 0:
                perf_stats["wall_fps"] = total_camera_frames / perf_stats["wall_time_seconds"]
            active_seconds = sum(perf_stats["pipeline_times"])
            if active_seconds > 0:
                perf_stats["active_pipeline_fps"] = perf_stats["processed_frames"] / active_seconds
            perf_csv = write_video_performance_summary(session_id, "", perf_stats)
            print(f"[Performance] Real-time summary saved: {perf_csv}")
        except Exception as perf_error:
            print(f"[Performance] Failed to save real-time summary: {perf_error}")
        try:
            qos_csv = write_qos_detail_rows(session_id, "", qos_rows)
            if qos_csv:
                print(f"[QoS] Real-time detail saved: {qos_csv}")
        except Exception as qos_error:
            print(f"[QoS] Failed to save real-time detail: {qos_error}")
        try:
            metadata_csv = write_test_metadata_rows(session_id, "", metadata_rows)
            if metadata_csv:
                print(f"[Metadata] Real-time detail saved: {metadata_csv}")
            actual_sample_rate = float(camera_fps) / max(YOLO_EVERY, 1)
            sync_csv = write_sync_summary(session_id, "", {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_id": session_id,
                "source_type": "realtime",
                "source_name": camera_name,
                "resolution": f"{w}x{h}",
                "source_fps": round(float(camera_fps), 4),
                "target_sample_rate": round(float(TARGET_SAMPLE_RATE), 4),
                "actual_sample_rate": round(actual_sample_rate, 4),
                "yolo_every": YOLO_EVERY,
                "sequence_length": NUM_FRAMES,
                "sequence_window_seconds": round(NUM_FRAMES / max(actual_sample_rate, 0.1), 4),
                "base_threshold": round(float(det.threshold), 4),
                "general_threshold": round(float(GENERAL_STAGE1_THRESHOLD), 4),
                "low_light_threshold": round(float(LOW_LIGHT_GENERAL_THRESHOLD), 4),
                "model_override_threshold": round(float(MODEL_OVERRIDE_THRESHOLD), 4),
                "buffer_capacity": det.buffer_size,
                "default_buffer_requirement": MIN_FIGHT_COUNT,
                "min_persons": MIN_PERSONS,
                "min_interaction_score": MIN_INTERACTION_SCORE,
                "contact_density_threshold": CONTACT_DENSITY_THRESHOLD,
                "realtime_person_min_count": REALTIME_PERSON_MIN_COUNT,
                "realtime_interaction_min_count": REALTIME_INTERACTION_MIN_COUNT,
                "passive_proximity_window": PASSIVE_PROXIMITY_WINDOW,
                "passive_proximity_min_count": PASSIVE_PROXIMITY_MIN_COUNT,
                "passive_min_pose_support_ratio": PASSIVE_MIN_POSE_SUPPORT_RATIO,
                "passive_max_keypoint_violence": PASSIVE_MAX_KEYPOINT_VIOLENCE,
                "filters_logged": (
                    "filter_reason,decision_mode,stage1,stage2,contact_density,"
                    "keypoint_violence_score,pose_support_ratio,passive_sustained"
                ),
            })
            print(f"[Metadata] Real-time sync summary saved: {sync_csv}")
        except Exception as metadata_error:
            print(f"[Metadata] Failed to save real-time detail: {metadata_error}")
        try:
            heuristic_csv = write_heuristic_detail_rows(session_id, "", heuristic_rows)
            if heuristic_csv:
                print(f"[Heuristic] Real-time detail saved: {heuristic_csv}")
        except Exception as heuristic_error:
            print(f"[Heuristic] Failed to save real-time detail: {heuristic_error}")
        print(f"[Stream] Closed: {camera_name}")
        socketio.emit('stream_closed', {'ip': camera_ip})


# ── MJPEG Streaming ───────────────────────────────────────────────────────────
def mjpeg_generator(camera_ip, frame_dict):
    """Yield JPEG frames as multipart HTTP response."""
    while camera_ip in stop_flags and not stop_flags[camera_ip].is_set():
        if camera_ip in frame_dict:
            frame_bytes = frame_dict[camera_ip]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # ~20 FPS max


@app.route('/video_feed/raw/<path:camera_ip>')
def video_feed_raw(camera_ip):
    """MJPEG stream endpoint for raw camera feed."""
    return Response(
        mjpeg_generator(camera_ip, latest_frames_raw),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video_feed/anno/<path:camera_ip>')
def video_feed_anno(camera_ip):
    """MJPEG stream endpoint for annotated camera feed."""
    return Response(
        mjpeg_generator(camera_ip, latest_frames_anno),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ── Video Mode: MJPEG Generator ───────────────────────────────────────────────
def video_mjpeg_generator(video_id, frame_dict):
    """Yield JPEG frames for video mode as multipart HTTP response."""
    while video_id in video_stop_flags and not video_stop_flags[video_id].is_set():
        if video_id in frame_dict:
            frame_bytes = frame_dict[video_id]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # ~20 FPS
    # Send last frame after done (so display doesn't freeze on empty)
    if video_id in frame_dict:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_dict[video_id] + b'\r\n')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def avg_ms(values):
    if not values:
        return 0.0
    return round((sum(values) / len(values)) * 1000.0, 4)


def ping_host_ms(host, timeout_ms=300):
    """Return ICMP ping latency in ms, or blank string when unavailable."""
    if not host or str(host).isdigit():
        return ""
    host = str(host).strip()
    if "://" in host or "/" in host:
        return ""
    try:
        if platform.system().lower().startswith("win"):
            cmd = ["ping", "-n", "1", "-w", str(int(timeout_ms)), host]
        else:
            cmd = ["ping", "-c", "1", "-W", str(max(1, int(timeout_ms / 1000))), host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout_ms / 1000.0 + 0.5))
        output = (result.stdout or "") + (result.stderr or "")
        match = re.search(r"(?:time[=<]\s*|time=)(\d+(?:\.\d+)?)\s*ms", output, flags=re.IGNORECASE)
        if match:
            return round(float(match.group(1)), 4)
    except Exception:
        pass
    return ""


def compute_qos_network_metrics(source_type, pipeline_ms, previous_pipeline_ms,
                                encoded_bytes=0, elapsed_seconds=0.0, ping_ms=""):
    """Derive QoS fields from pipeline timing and encoded output size."""
    latency_ms = round(float(pipeline_ms or 0.0), 4)
    jitter_ms = ""
    if previous_pipeline_ms not in ("", None):
        try:
            jitter_ms = round(abs(float(pipeline_ms or 0.0) - float(previous_pipeline_ms)), 4)
        except Exception:
            jitter_ms = ""
    throughput_kbps = 0.0
    bitrate_kbps = 0.0
    if elapsed_seconds and elapsed_seconds > 0 and encoded_bytes:
        throughput_kbps = round((float(encoded_bytes) * 8.0) / float(elapsed_seconds) / 1000.0, 4)
        bitrate_kbps = throughput_kbps
    return {
        "delay_ms": latency_ms,
        "latency_ms": latency_ms,
        "jitter_ms": jitter_ms,
        "ping_ms": ping_ms if source_type == "realtime" else "",
        "throughput_kbps": throughput_kbps,
        "bitrate_kbps": bitrate_kbps,
        "encoded_frame_bytes": int(encoded_bytes or 0),
    }


def get_video_output_name(video_id, video_path):
    original_name = video_original_names.get(video_id)
    if original_name:
        return secure_filename(os.path.splitext(original_name)[0])
    if video_id:
        return secure_filename(str(video_id))
    return secure_filename(os.path.splitext(os.path.basename(video_path))[0])


def write_video_performance_summary(video_id, video_path, stats):
    csv_base_name = get_video_output_name(video_id, video_path)
    output_path = os.path.join(PERFORMANCE_FOLDER, f"performance_{csv_base_name}.csv")
    file_exists = os.path.exists(output_path)
    fieldnames = [
        "timestamp",
        "video_id",
        "video_name",
        "resolution",
        "source_fps",
        "total_frames",
        "processed_frames",
        "yolo_every",
        "wall_time_seconds",
        "wall_fps",
        "active_pipeline_fps",
        "avg_yolo_ms",
        "avg_heuristic_ms",
        "avg_mobilenet_ms",
        "avg_cnn_lstm_ms",
        "avg_encode_ms",
        "avg_detection_latency_ms",
        "end_to_end_latency_ms",
        "avg_pipeline_ms",
        "avg_alert_decision_delay_ms",
        "max_alert_decision_delay_ms",
        "avg_alert_save_delay_ms",
        "max_alert_save_delay_ms",
        "alert_events",
        "cnn_lstm_runs",
        "firebase_saves",
    ]
    header_matches = False
    if file_exists:
        try:
            with open(output_path, "r", encoding="utf-8") as existing:
                header_matches = existing.readline().strip().split(",") == fieldnames
        except OSError:
            header_matches = False
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or not header_matches:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": video_id,
            "video_name": video_original_names.get(video_id, os.path.basename(video_path)),
            "resolution": f"{stats['frame_w']}x{stats['frame_h']}",
            "source_fps": round(float(stats["src_fps"]), 4),
            "total_frames": stats["total_frames"],
            "processed_frames": stats["processed_frames"],
            "yolo_every": stats["yolo_every"],
            "wall_time_seconds": round(stats["wall_time_seconds"], 4),
            "wall_fps": round(stats["wall_fps"], 4),
            "active_pipeline_fps": round(stats["active_pipeline_fps"], 4),
            "avg_yolo_ms": avg_ms(stats["yolo_times"]),
            "avg_heuristic_ms": avg_ms(stats["heuristic_times"]),
            "avg_mobilenet_ms": avg_ms(stats["mobilenet_times"]),
            "avg_cnn_lstm_ms": avg_ms(stats["lstm_times"]),
            "avg_encode_ms": avg_ms(stats["encode_times"]),
            "avg_detection_latency_ms": avg_ms(stats["pipeline_times"]),
            "end_to_end_latency_ms": round((stats["wall_time_seconds"] / stats["processed_frames"]) * 1000.0, 4) if stats["processed_frames"] > 0 else 0.0,
            "avg_pipeline_ms": avg_ms(stats["pipeline_times"]),
            "avg_alert_decision_delay_ms": avg_ms(stats.get("alert_decision_delay_times", [])),
            "max_alert_decision_delay_ms": round(max(stats.get("alert_decision_delay_times", [0.0])) * 1000.0, 4) if stats.get("alert_decision_delay_times") else 0.0,
            "avg_alert_save_delay_ms": avg_ms(stats.get("alert_save_delay_times", [])),
            "max_alert_save_delay_ms": round(max(stats.get("alert_save_delay_times", [0.0])) * 1000.0, 4) if stats.get("alert_save_delay_times") else 0.0,
            "alert_events": len(stats.get("alert_decision_delay_times", [])),
            "cnn_lstm_runs": len(stats["lstm_times"]),
            "firebase_saves": stats["firebase_saves"],
        })
    return output_path


QOS_DETAIL_FIELDNAMES = [
    "timestamp",
    "video_id",
    "source_type",
    "frame",
    "second",
    "source_fps",
    "yolo_every",
    "processed_frames",
    "wall_fps",
    "active_pipeline_fps",
    "delay_ms",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "throughput_kbps",
    "bitrate_kbps",
    "encoded_frame_bytes",
    "yolo_ms",
    "heuristic_ms",
    "mobilenet_ms",
    "cnn_lstm_ms",
    "encode_ms",
    "pipeline_ms",
    "alert_decision_delay_ms",
    "alert_save_delay_ms",
    "is_alert",
    "new_alert_triggered",
    "decision_mode",
]


def write_qos_detail_rows(video_id, video_path, rows):
    if not rows:
        return None
    csv_base_name = get_video_output_name(video_id, video_path)
    output_path = os.path.join(QOS_FOLDER, f"qos_{csv_base_name}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QOS_DETAIL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


TEST_METADATA_FIELDNAMES = [
    "timestamp",
    "video_id",
    "source_type",
    "source_name",
    "camera_ip",
    "frame",
    "second",
    "resolution",
    "source_fps",
    "target_sample_rate",
    "actual_sample_rate",
    "sequence_window_seconds",
    "processed_fps",
    "yolo_every",
    "base_threshold",
    "threshold",
    "general_threshold",
    "low_light_threshold",
    "model_override_threshold",
    "min_persons",
    "min_interaction_score",
    "buffer_capacity",
    "buffer_size",
    "buffer_requirement",
    "buffer_fight_count",
    "persons",
    "min_person_passed",
    "interaction_score",
    "interaction_threshold",
    "interaction_passed",
    "interaction_positive_count",
    "person_positive_count",
    "physical_contact",
    "contact_density",
    "brightness",
    "saturation",
    "brightness_threshold",
    "saturation_threshold",
    "low_light",
    "keypoint_violence_score",
    "aggressive_motion_score",
    "pose_support_ratio",
    "passive_candidate",
    "passive_sustained",
    "raw_probability",
    "display_probability",
    "model_probability",
    "decision_mode",
    "filter_reason",
    "stage1",
    "stage2",
    "final_decision",
    "is_alert",
    "new_alert_triggered",
]


def write_test_metadata_rows(video_id, video_path, rows):
    if not rows:
        return None
    csv_base_name = get_video_output_name(video_id, video_path)
    output_path = os.path.join(TEST_METADATA_FOLDER, f"metadata_{csv_base_name}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


class RealtimeCSVWriter:
    """Persist QoS and test metadata throughout an active monitoring session."""

    def __init__(self, session_id):
        csv_base_name = get_video_output_name(session_id, "")
        self.qos_path = os.path.join(QOS_FOLDER, f"qos_{csv_base_name}.csv")
        self.metadata_path = os.path.join(
            TEST_METADATA_FOLDER,
            f"metadata_{csv_base_name}.csv",
        )
        self.qos_file = None
        self.metadata_file = None
        self.rows_since_flush = 0
        self.flush_every = max(1, int(round(TARGET_SAMPLE_RATE)))

        try:
            self.qos_file = open(self.qos_path, "w", newline="", encoding="utf-8")
            self.metadata_file = open(
                self.metadata_path,
                "w",
                newline="",
                encoding="utf-8",
            )
            self.qos_writer = csv.DictWriter(
                self.qos_file,
                fieldnames=QOS_DETAIL_FIELDNAMES,
            )
            self.metadata_writer = csv.DictWriter(
                self.metadata_file,
                fieldnames=TEST_METADATA_FIELDNAMES,
            )
            self.qos_writer.writeheader()
            self.metadata_writer.writeheader()
            self.flush()
        except Exception:
            self.close()
            raise

    def append(self, qos_row, metadata_row):
        self.qos_writer.writerow(qos_row)
        self.metadata_writer.writerow(metadata_row)
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        if self.qos_file is not None:
            self.qos_file.flush()
        if self.metadata_file is not None:
            self.metadata_file.flush()
        self.rows_since_flush = 0

    def close(self):
        self.flush()
        if self.qos_file is not None:
            self.qos_file.close()
            self.qos_file = None
        if self.metadata_file is not None:
            self.metadata_file.close()
            self.metadata_file = None


def write_sync_summary(video_id, video_path, row):
    csv_base_name = get_video_output_name(video_id, video_path)
    output_path = os.path.join(TEST_METADATA_FOLDER, f"sync_summary_{csv_base_name}.csv")
    fieldnames = [
        "timestamp",
        "video_id",
        "source_type",
        "source_name",
        "resolution",
        "source_fps",
        "target_sample_rate",
        "actual_sample_rate",
        "yolo_every",
        "sequence_length",
        "sequence_window_seconds",
        "base_threshold",
        "general_threshold",
        "low_light_threshold",
        "model_override_threshold",
        "buffer_capacity",
        "default_buffer_requirement",
        "min_persons",
        "min_interaction_score",
        "contact_density_threshold",
        "realtime_person_min_count",
        "realtime_interaction_min_count",
        "passive_proximity_window",
        "passive_proximity_min_count",
        "passive_min_pose_support_ratio",
        "passive_max_keypoint_violence",
        "filters_logged",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return output_path


def write_heuristic_detail_rows(video_id, video_path, rows):
    if not rows:
        return None
    csv_base_name = get_video_output_name(video_id, video_path)
    output_path = os.path.join(HEURISTIC_FOLDER, f"{csv_base_name}.csv")
    fieldnames = [
        "timestamp",
        "video_id",
        "frame",
        "second",
        "persons",
        "min_person_passed",
        "proximity_score",
        "velocity_score",
        "overlap_score",
        "interaction_score",
        "interaction_threshold",
        "interaction_passed",
        "interaction_positive_count",
        "person_positive_count",
        "center_distance_px",
        "avg_bbox_diagonal_px",
        "previous_distance_px",
        "closing_speed_px",
        "closing_velocity_px_per_second",
        "overlap_iou",
        "interaction_pair",
        "physical_contact",
        "contact_person",
        "core_person",
        "extremity_keypoint",
        "core_keypoint",
        "contact_distance_px",
        "contact_threshold_px",
        "low_light",
        "avg_extremity_velocity",
        "max_extremity_velocity",
        "max_leg_velocity",
        "avg_extremity_velocity_per_second",
        "max_extremity_velocity_per_second",
        "max_leg_velocity_per_second",
        "closing_speed_norm",
        "aggressive_motion_score",
        "keypoint_violence_score",
        "pose_support_ratio",
        "passive_candidate",
        "passive_sustained",
        "motion_passed",
        "static_contact",
        "passive_overlap",
        "effective_threshold",
        "decision_mode",
        "kick_contact",
        "model_probability",
        "filter_reason",
        "raw_probability",
        "display_probability",
        "buffer_fight_count",
        "buffer_size",
        "buffer_requirement",
        "alert",
        "cooldown_remaining_seconds",
        "new_alert_triggered",
        "total_alerts",
        "stage1",
        "stage2",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def create_evidence_clip_from_video(video_path, output_name, video_id, alert_second,
                                    pre_seconds=EVIDENCE_PRE_SECONDS,
                                    post_seconds=EVIDENCE_POST_SECONDS):
    """Cut an uploaded source video around the alert timestamp."""
    if not video_path or not os.path.exists(video_path):
        return None, 0.0

    source = cv2.VideoCapture(video_path)
    if not source.isOpened():
        print(f"[Evidence] Failed to open source video: {video_path}")
        return None, 0.0

    try:
        fps = float(source.get(cv2.CAP_PROP_FPS) or 30.0)
        fps = fps if fps > 1 else 30.0
        total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            print(f"[Evidence] Invalid source video dimensions: {video_path}")
            return None, 0.0

        start_second = max(0.0, float(alert_second) - float(pre_seconds))
        end_second = float(alert_second) + float(post_seconds)
        start_frame = max(0, int(round(start_second * fps)))
        if total_frames > 0:
            end_frame = min(total_frames - 1, int(round(end_second * fps)))
        else:
            end_frame = int(round(end_second * fps))

        if end_frame < start_frame:
            end_frame = start_frame

        safe_name = secure_filename(output_name) or secure_filename(str(video_id)) or 'video'
        alert_ms = int(round(float(alert_second) * 1000.0))
        output_stem = os.path.join(
            EVIDENCE_FOLDER,
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_{safe_name}_{alert_ms}ms_upload_evidence"
        )
        writer, output_path = create_browser_video_writer(output_stem, width, height, fps)
        if writer is None:
            print(f"[Evidence] Failed to create upload clip: {output_stem}")
            return None, 0.0

        source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        frames_written = 0
        while current_frame <= end_frame:
            ret, frame = source.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frames_written += 1
            current_frame += 1

        writer.release()
        duration = frames_written / fps if fps > 0 else 0.0
        if frames_written == 0:
            try:
                os.remove(output_path)
            except OSError:
                pass
            print(f"[Evidence] Upload clip empty for alert at {alert_second:.2f}s")
            return None, 0.0

        print(f"[Evidence] Upload clip saved: {output_path} ({duration:.2f}s)")
        return output_path, duration
    finally:
        source.release()


def video_processing_loop(video_id, video_path, stop_event):
    """
    Background thread: process uploaded video file through the same pipeline
    as detection_loop (YOLO → features → ensemble → TwoStageDetector).
    Emits frames to MJPEG buffers and detection_update/video_progress/video_done
    via WebSocket.
    """
    global video_detectors, video_frames_raw, video_frames_anno

    output_name = get_video_output_name(video_id, video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit('video_error', {'video_id': video_id, 'error': 'Failed to open video file'})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    DISPLAY_W    = 640

    print(f"[Video] Processing {video_id}: {frame_w}x{frame_h} @ {src_fps:.0f}fps, {total_frames} frames")

    # Create detector for this video
    video_detectors[video_id] = TwoStageDetector(ensemble_models)
    det = video_detectors[video_id]
    det.display_probability = 0.0

    cnn_buffer      = []
    contact_buffer  = []
    keypoint_buffer = []
    person_buffer = []
    interaction_buffer = []
    kick_contact_buffer = []
    best_interaction_detail = None
    frame_count     = 0
    last_detect_at  = -1   # frame index of last YOLO run
    # Short clips must still produce enough observations for the 20-frame LSTM
    # sequence. Long videos remain capped near 6 inference samples per second.
    YOLO_EVERY = max(1, int(round(float(src_fps) / max(TARGET_SAMPLE_RATE, 0.1))))
    prev_boxes      = None
    current_plotted = None
    last_emit_at    = -1
    EMIT_EVERY      = max(1, int(src_fps / 20))  # emit frames at ~20 fps
    last_alert_save = 0
    process_started_at = time.perf_counter()
    perf_stats = {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "src_fps": src_fps,
        "total_frames": total_frames,
        "processed_frames": 0,
        "yolo_every": YOLO_EVERY,
        "wall_time_seconds": 0.0,
        "wall_fps": 0.0,
        "active_pipeline_fps": 0.0,
        "yolo_times": [],
        "heuristic_times": [],
        "mobilenet_times": [],
        "lstm_times": [],
        "encode_times": [],
        "pipeline_times": [],
        "alert_decision_delay_times": [],
        "alert_save_delay_times": [],
        "firebase_saves": 0,
    }

    n_persons    = 0
    interaction  = 0.0
    low_light    = False
    kick_contact = False
    effective_threshold = det.threshold
    min_fight_count = MIN_FIGHT_COUNT
    decision_mode = "General"
    motion_context = compute_temporal_motion_context([], 1.0, None, False)
    heuristic_rows = []
    qos_rows = []
    metadata_rows = []
    previous_qos_pipeline_ms = None
    last_valid_frame = None

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1
            last_valid_frame = frame

            # ── YOLO Processing (every YOLO_EVERY frames) ──
            if (frame_count - last_detect_at) >= YOLO_EVERY:
                pipeline_start = time.perf_counter()
                last_detect_at = frame_count
                yolo_elapsed = 0.0
                heuristic_elapsed = 0.0
                mobilenet_elapsed = 0.0
                lstm_elapsed = 0.0
                alert_decision_delay = 0.0
                alert_save_delay = 0.0

                yolo_scale = YOLO_INFER_SIZE / max(frame_w, frame_h)
                yolo_frame = cv2.resize(frame, (int(frame_w * yolo_scale), int(frame_h * yolo_scale)))

                yolo_start = time.perf_counter()
                results = yolo_model.predict(yolo_frame, verbose=False, conf=0.5)
                yolo_elapsed = time.perf_counter() - yolo_start
                perf_stats["yolo_times"].append(yolo_elapsed)

                heuristic_start = time.perf_counter()
                yolo_area = yolo_frame.shape[0] * yolo_frame.shape[1]
                n_persons = count_persons(results[0], frame_area=yolo_area)
                interaction, cur_boxes, interaction_detail = compute_interaction_score(
                    results[0],
                    prev_boxes,
                    frame_area=yolo_area,
                    return_details=True
                )
                prev_boxes = cur_boxes
                if (
                    best_interaction_detail is None or
                    interaction_detail["interaction_score"] > best_interaction_detail["interaction_score"]
                ):
                    best_interaction_detail = interaction_detail.copy()
                keypoint_buffer.append(extract_keypoint_snapshot(results[0]))
                current_plotted = results[0].plot()

                yw, yh = yolo_frame.shape[1], yolo_frame.shape[0]
                
                # Check absolute physical contact
                contact, contact_detail = check_physical_contact(results[0], yw, yh, return_details=True)
                kick_contact = contact and str(contact_detail.get("extremity_keypoint", "")) in {"15", "16"}
                contact_buffer.append(contact)
                person_buffer.append(n_persons)
                interaction_buffer.append(interaction)
                kick_contact_buffer.append(kick_contact)
                heuristic_elapsed = time.perf_counter() - heuristic_start
                perf_stats["heuristic_times"].append(heuristic_elapsed)
                
                # Extract CNN features (MobileNetV2) — same as CCTV detection_loop
                mobilenet_start = time.perf_counter()
                if n_persons > 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype(np.float32)
                    frame_pre = preprocess_input(np.expand_dims(frame_resized, axis=0))
                    cnn_feat = cnn_feature_extractor.predict(frame_pre, verbose=0)[0]
                else:
                    cnn_feat = np.zeros((CNN_FEATURE_SIZE,), dtype=np.float32)
                mobilenet_elapsed = time.perf_counter() - mobilenet_start
                perf_stats["mobilenet_times"].append(mobilenet_elapsed)
                    
                cnn_buffer.append(cnn_feat)
                sequence_fill_ratio = min(1.0, len(cnn_buffer) / float(NUM_FRAMES))
                recent_contact = any(contact_buffer[-NUM_FRAMES:]) if contact_buffer else False
                lighting_metrics = measure_lighting_frame(frame)
                low_light = lighting_metrics["is_low_light"]
                frame_diag = float(np.sqrt((yolo_frame.shape[1] ** 2) + (yolo_frame.shape[0] ** 2)))
                motion_context = compute_temporal_motion_context(
                    keypoint_buffer[-NUM_FRAMES:],
                    frame_diag,
                    interaction_detail,
                    recent_contact,
                )
                keypoint_violence_score = compute_keypoint_violence_score(
                    keypoint_buffer[-NUM_FRAMES:],
                    frame_diag,
                )
                latest_keypoint_violence_score = compute_keypoint_violence_score(
                    keypoint_buffer[-2:],
                    frame_diag,
                )
                effective_threshold = det.threshold
                min_fight_count = MIN_FIGHT_COUNT
                decision_mode = "General"
                motion_context["low_light"] = bool(low_light)
                new_alert_triggered = False
                model_probability = 0.0
                filter_reason = ""

                # ── Ensemble prediction when 20 frames collected ──
                if len(cnn_buffer) >= NUM_FRAMES:
                    recent_persons = max(person_buffer[-NUM_FRAMES:], default=n_persons)
                    recent_interaction = max(interaction_buffer[-NUM_FRAMES:], default=interaction)
                    if recent_persons >= MIN_PERSONS and recent_interaction >= MIN_INTERACTION_SCORE:
                        cnn_seq = np.array(cnn_buffer[-NUM_FRAMES:], dtype=np.float32)
                        lstm_start = time.perf_counter()
                        probability = det.predict_ensemble(cnn_seq)
                        model_probability = probability
                        lstm_elapsed = time.perf_counter() - lstm_start
                        perf_stats["lstm_times"].append(lstm_elapsed)
                        
                        # Temporal validation: suppress static contact/passive overlap,
                        # and use a lower threshold only for low-light frames with aggressive motion.
                        probability, effective_threshold, filter_reason, min_fight_count, decision_mode = apply_decision_filters(
                            probability,
                            det.threshold,
                            recent_contact,
                            motion_context,
                            low_light,
                            kick_contact,
                            contact_density=sum(contact_buffer[-NUM_FRAMES:]) if contact_buffer else 0,
                        )
                        (
                            probability,
                            effective_threshold,
                            min_fight_count,
                            decision_mode,
                            filter_reason,
                        ) = apply_runtime_action_policy(
                            probability=probability,
                            effective_threshold=effective_threshold,
                            min_fight_count=min_fight_count,
                            decision_mode=decision_mode,
                            filter_reason=filter_reason,
                            recent_contact=recent_contact,
                            contact_density=sum(contact_buffer[-NUM_FRAMES:]) if contact_buffer else 0,
                            motion_context=motion_context,
                            keypoint_violence_score=keypoint_violence_score,
                            latest_keypoint_violence_score=latest_keypoint_violence_score,
                            kick_contact=kick_contact,
                        )
                        motion_context["effective_threshold"] = effective_threshold
                        if filter_reason:
                            det.stage1_result = filter_reason
                    else:
                        probability = 0.0
                        if n_persons < MIN_PERSONS:
                            det.prediction_buffer.clear()
                            det.is_alert = False
                            det.current_probability = 0.0
                            det.stage1_result = "No interaction"
                            det.stage2_result = "Safe"
                            cnn_buffer.clear()
                            contact_buffer.clear()
                            keypoint_buffer.clear()
                            person_buffer.clear()
                            interaction_buffer.clear()
                            kick_contact_buffer.clear()

                    previous_total_alerts = det.total_alerts
                    det.update(probability, threshold=effective_threshold, min_fight=min_fight_count)
                    new_alert_triggered = det.total_alerts > previous_total_alerts
                    if new_alert_triggered:
                        alert_decision_delay = time.perf_counter() - pipeline_start
                        perf_stats["alert_decision_delay_times"].append(alert_decision_delay)
                    if len(cnn_buffer) > NUM_FRAMES:
                        cnn_buffer.pop(0)
                        contact_buffer.pop(0)
                        keypoint_buffer.pop(0)
                        person_buffer.pop(0)
                        interaction_buffer.pop(0)
                        kick_contact_buffer.pop(0)

                    if det.is_alert and time.time() - last_alert_save > COOLDOWN_SECS:
                        last_alert_save = time.time()
                        alert_second = frame_count / max(src_fps, 1)
                        alert_save_start = time.perf_counter()
                        result = save_to_firebase(
                            frame,
                            output_name,
                            video_id,
                            det.current_probability,
                            source_type='video'
                        )
                        perf_stats["firebase_saves"] += 1
                        if result:
                            enqueue_upload_evidence_clip(
                                video_path=video_path,
                                output_name=output_name,
                                video_id=video_id,
                                alert_second=alert_second,
                                firestore_id=result.get('id'),
                                probability=det.current_probability,
                            )
                            socketio.emit('history_update', result)
                        alert_save_delay = time.perf_counter() - alert_save_start
                        perf_stats["alert_save_delay_times"].append(alert_save_delay)

                display_probability = build_video_display_probability(
                    raw_probability=det.current_probability,
                    n_persons=n_persons,
                    interaction=interaction,
                    has_contact=recent_contact,
                    sequence_fill_ratio=sequence_fill_ratio,
                    threshold=effective_threshold
                )
                det.update_display_probability(display_probability)

                # Emit detection status
                pct = int(round(float(det.display_probability) * 100))
                fight_count = int(sum(det.prediction_buffer))
                cooldown_remaining = 0.0
                if det.last_alert_time > 0:
                    cooldown_remaining = max(0.0, det.cooldown - (time.time() - det.last_alert_time))
                samples_per_second = src_fps / max(YOLO_EVERY, 1)
                closing_speed_px = interaction_detail["closing_speed_px"]
                heuristic_rows.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_id": video_id,
                    "frame": frame_count,
                    "second": round(frame_count / max(src_fps, 1), 4),
                    "persons": n_persons,
                    "min_person_passed": n_persons >= MIN_PERSONS,
                    "proximity_score": interaction_detail["proximity_score"],
                    "velocity_score": interaction_detail["velocity_score"],
                    "overlap_score": interaction_detail["overlap_score"],
                    "interaction_score": interaction_detail["interaction_score"],
                    "interaction_threshold": MIN_INTERACTION_SCORE,
                    "interaction_passed": interaction >= MIN_INTERACTION_SCORE,
                    "center_distance_px": interaction_detail["center_distance_px"],
                    "avg_bbox_diagonal_px": interaction_detail["avg_bbox_diagonal_px"],
                    "previous_distance_px": interaction_detail["previous_distance_px"],
                    "closing_speed_px": closing_speed_px,
                    "closing_velocity_px_per_second": (
                        round(float(closing_speed_px) * samples_per_second, 4)
                        if closing_speed_px != "" else ""
                    ),
                    "overlap_iou": interaction_detail["overlap_iou"],
                    "interaction_pair": interaction_detail["interaction_pair"],
                    "physical_contact": bool(contact),
                    "contact_person": contact_detail["contact_person"],
                    "core_person": contact_detail["core_person"],
                    "extremity_keypoint": contact_detail["extremity_keypoint"],
                    "core_keypoint": contact_detail["core_keypoint"],
                    "contact_distance_px": contact_detail["contact_distance_px"],
                    "contact_threshold_px": contact_detail["contact_threshold_px"],
                    "low_light": bool(low_light),
                    "avg_extremity_velocity": motion_context["avg_extremity_velocity"],
                    "max_extremity_velocity": motion_context["max_extremity_velocity"],
                    "max_leg_velocity": motion_context["max_leg_velocity"],
                    "avg_extremity_velocity_per_second": round(
                        motion_context["avg_extremity_velocity"] * samples_per_second, 5
                    ),
                    "max_extremity_velocity_per_second": round(
                        motion_context["max_extremity_velocity"] * samples_per_second, 5
                    ),
                    "max_leg_velocity_per_second": round(
                        motion_context["max_leg_velocity"] * samples_per_second, 5
                    ),
                    "closing_speed_norm": motion_context["closing_speed_norm"],
                    "aggressive_motion_score": motion_context["aggressive_motion_score"],
                    "motion_passed": motion_context["motion_passed"],
                    "static_contact": motion_context["static_contact"],
                    "passive_overlap": motion_context["passive_overlap"],
                    "effective_threshold": round(float(effective_threshold), 4),
                    "decision_mode": decision_mode,
                    "kick_contact": bool(kick_contact),
                    "model_probability": round(float(model_probability), 4),
                    "filter_reason": filter_reason,
                    "raw_probability": round(float(det.current_probability), 4),
                    "display_probability": round(float(det.display_probability), 4),
                    "buffer_fight_count": fight_count,
                    "buffer_size": len(det.prediction_buffer),
                    "buffer_requirement": det.min_fight,
                    "alert": bool(det.is_alert),
                    "cooldown_remaining_seconds": round(float(cooldown_remaining), 4),
                    "new_alert_triggered": bool(new_alert_triggered),
                    "total_alerts": int(det.total_alerts),
                    "stage1": str(det.stage1_result),
                    "stage2": str(det.stage2_result),
                })
                socketio.emit('video_detection_update', {
                    'video_id': video_id,
                    'status': 'KDRT' if det.is_alert else 'Normal',
                    'probability': round(float(det.display_probability), 4),
                    'raw_probability': round(float(det.current_probability), 4),
                    'percentage': pct,
                    'is_alert': bool(det.is_alert),
                    'persons': n_persons,
                    'interaction': round(float(interaction), 3),
                    'low_light': bool(low_light),
                    'aggressive_motion': motion_context["aggressive_motion_score"],
                    'decision_mode': decision_mode,
                    'stage1': str(det.stage1_result),
                    'stage2': str(det.stage2_result),
                })
                pipeline_elapsed = time.perf_counter() - pipeline_start
                perf_stats["pipeline_times"].append(pipeline_elapsed)
                perf_stats["processed_frames"] += 1
                wall_elapsed = max(time.perf_counter() - process_started_at, 1e-6)
                active_elapsed = max(sum(perf_stats["pipeline_times"]), 1e-6)
                pipeline_ms = round(pipeline_elapsed * 1000.0, 4)
                encoded_frame_bytes = (
                    len(video_frames_raw.get(video_id, b"")) +
                    len(video_frames_anno.get(video_id, b""))
                )
                qos_metrics = compute_qos_network_metrics(
                    "upload_video",
                    pipeline_ms,
                    previous_qos_pipeline_ms,
                    encoded_bytes=encoded_frame_bytes,
                    elapsed_seconds=pipeline_elapsed,
                    ping_ms="",
                )
                previous_qos_pipeline_ms = pipeline_ms
                qos_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_id": video_id,
                    "source_type": "upload_video",
                    "frame": frame_count,
                    "second": round(frame_count / max(src_fps, 1), 4),
                    "source_fps": round(float(src_fps), 4),
                    "yolo_every": YOLO_EVERY,
                    "processed_frames": perf_stats["processed_frames"],
                    "wall_fps": round(frame_count / wall_elapsed, 4),
                    "active_pipeline_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                    "yolo_ms": round(yolo_elapsed * 1000.0, 4),
                    "heuristic_ms": round(heuristic_elapsed * 1000.0, 4),
                    "mobilenet_ms": round(mobilenet_elapsed * 1000.0, 4),
                    "cnn_lstm_ms": round(lstm_elapsed * 1000.0, 4),
                    "encode_ms": "",
                    "pipeline_ms": pipeline_ms,
                    "alert_decision_delay_ms": round(alert_decision_delay * 1000.0, 4),
                    "alert_save_delay_ms": round(alert_save_delay * 1000.0, 4),
                    "is_alert": bool(det.is_alert),
                    "new_alert_triggered": bool(new_alert_triggered),
                    "decision_mode": decision_mode,
                }
                qos_row.update(qos_metrics)
                qos_rows.append(qos_row)
                metadata_rows.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_id": video_id,
                    "source_type": "upload_video",
                    "source_name": video_original_names.get(video_id, os.path.basename(video_path)),
                    "camera_ip": "",
                    "frame": frame_count,
                    "second": round(frame_count / max(src_fps, 1), 4),
                    "resolution": f"{frame_w}x{frame_h}",
                    "source_fps": round(float(src_fps), 4),
                    "target_sample_rate": round(float(TARGET_SAMPLE_RATE), 4),
                    "actual_sample_rate": round(float(src_fps) / max(YOLO_EVERY, 1), 4),
                    "sequence_window_seconds": round(NUM_FRAMES / max(float(src_fps) / max(YOLO_EVERY, 1), 0.1), 4),
                    "processed_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                    "yolo_every": YOLO_EVERY,
                    "base_threshold": round(float(det.threshold), 4),
                    "threshold": round(float(effective_threshold), 4),
                    "general_threshold": round(float(GENERAL_STAGE1_THRESHOLD), 4),
                    "low_light_threshold": round(float(LOW_LIGHT_GENERAL_THRESHOLD), 4),
                    "model_override_threshold": round(float(MODEL_OVERRIDE_THRESHOLD), 4),
                    "min_persons": MIN_PERSONS,
                    "min_interaction_score": MIN_INTERACTION_SCORE,
                    "buffer_capacity": det.buffer_size,
                    "buffer_size": len(det.prediction_buffer),
                    "buffer_requirement": min_fight_count,
                    "buffer_fight_count": int(sum(det.prediction_buffer)),
                    "persons": n_persons,
                    "min_person_passed": n_persons >= MIN_PERSONS,
                    "interaction_score": round(float(interaction), 4),
                    "interaction_threshold": MIN_INTERACTION_SCORE,
                    "interaction_passed": interaction >= MIN_INTERACTION_SCORE,
                    "physical_contact": bool(contact),
                    "contact_density": int(sum(contact_buffer[-NUM_FRAMES:])) if contact_buffer else 0,
                    "brightness": lighting_metrics["brightness"],
                    "saturation": lighting_metrics["saturation"],
                    "brightness_threshold": round(float(DARK_BRIGHTNESS_THRESHOLD), 4),
                    "saturation_threshold": round(float(DARK_SATURATION_THRESHOLD), 4),
                    "low_light": bool(low_light),
                    "keypoint_violence_score": round(float(keypoint_violence_score), 5),
                    "aggressive_motion_score": motion_context["aggressive_motion_score"],
                    "raw_probability": round(float(det.current_probability), 4),
                    "display_probability": round(float(det.display_probability), 4),
                    "model_probability": round(float(model_probability), 4),
                    "decision_mode": decision_mode,
                    "filter_reason": filter_reason,
                    "stage1": str(det.stage1_result),
                    "stage2": str(det.stage2_result),
                    "final_decision": "KDRT" if det.is_alert else "Normal",
                    "is_alert": bool(det.is_alert),
                    "new_alert_triggered": bool(new_alert_triggered),
                })

            # ── Store MJPEG frames ──
            if (frame_count - last_emit_at) >= EMIT_EVERY:
                encode_start = time.perf_counter()
                last_emit_at = frame_count
                scale     = DISPLAY_W / frame_w
                small_raw = cv2.resize(frame, (DISPLAY_W, int(frame_h * scale)))
                _, jpg_raw = cv2.imencode('.jpg', small_raw, [cv2.IMWRITE_JPEG_QUALITY, 70])
                video_frames_raw[video_id] = jpg_raw.tobytes()

                if current_plotted is not None:
                    ah, aw = current_plotted.shape[:2]
                    a_scale   = DISPLAY_W / aw
                    small_anno = cv2.resize(current_plotted, (DISPLAY_W, int(ah * a_scale)))
                    _, jpg_anno = cv2.imencode('.jpg', small_anno, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    video_frames_anno[video_id] = jpg_anno.tobytes()
                perf_stats["encode_times"].append(time.perf_counter() - encode_start)

            # ── Progress update every 30 frames ──
            if frame_count % 30 == 0 and total_frames > 0:
                pct_done = int(frame_count / total_frames * 100)
                socketio.emit('video_progress', {
                    'video_id': video_id,
                    'frame': frame_count,
                    'total': total_frames,
                    'percent': pct_done,
                })

            # Throttle to avoid hammering CPU when video is faster than realtime
            time.sleep(1.0 / max(src_fps, 1) * 0.5)

        # ── Final Flush for Short Videos ──
        # If the video finished but the number of collected CNN features is still less than 20 (i.e. for videos ~1-3 seconds long)
        if not stop_event.is_set() and 5 < len(cnn_buffer) < NUM_FRAMES:
            final_flush_start = time.perf_counter()
            final_alert_decision_delay = 0.0
            final_alert_save_delay = 0.0
            final_lstm_elapsed = 0.0
            # Pad the buffer with the last feature up to 20 frames (so the LSTM can process it)
            last_feat = cnn_buffer[-1]
            last_contact = contact_buffer[-1] if len(contact_buffer) > 0 else False
            last_keypoints = keypoint_buffer[-1] if len(keypoint_buffer) > 0 else None
            while len(cnn_buffer) < NUM_FRAMES:
                cnn_buffer.append(last_feat)
                contact_buffer.append(last_contact)
                keypoint_buffer.append(last_keypoints)

            recent_contact = any(contact_buffer[-NUM_FRAMES:]) if contact_buffer else False
            final_n_persons = max(person_buffer[-NUM_FRAMES:], default=n_persons)
            final_interaction = max(interaction_buffer[-NUM_FRAMES:], default=interaction)
            kick_contact = any(kick_contact_buffer[-NUM_FRAMES:]) if kick_contact_buffer else False
            frame_diag = float(np.sqrt((frame_w ** 2) + (frame_h ** 2)))
            motion_context = compute_temporal_motion_context(
                keypoint_buffer[-NUM_FRAMES:],
                frame_diag,
                best_interaction_detail,
                recent_contact,
            )
            keypoint_violence_score = compute_keypoint_violence_score(
                keypoint_buffer[-NUM_FRAMES:],
                frame_diag,
            )
            latest_keypoint_violence_score = compute_keypoint_violence_score(
                keypoint_buffer[-2:],
                frame_diag,
            )
            final_frame = last_valid_frame
            lighting_metrics = measure_lighting_frame(final_frame)
            low_light = lighting_metrics["is_low_light"]
            motion_context["low_light"] = bool(low_light)
            effective_threshold = det.threshold
            min_fight_count = MIN_FIGHT_COUNT
            decision_mode = "General"
                
            if final_n_persons >= MIN_PERSONS and final_interaction >= MIN_INTERACTION_SCORE:
                cnn_seq = np.array(cnn_buffer[-NUM_FRAMES:], dtype=np.float32)
                lstm_start = time.perf_counter()
                probability = det.predict_ensemble(cnn_seq)
                final_lstm_elapsed = time.perf_counter() - lstm_start
                perf_stats["lstm_times"].append(final_lstm_elapsed)
                
                probability, effective_threshold, filter_reason, min_fight_count, decision_mode = apply_decision_filters(
                    probability,
                    det.threshold,
                    recent_contact,
                    motion_context,
                    low_light,
                    kick_contact,
                    contact_density=sum(contact_buffer[-NUM_FRAMES:]) if contact_buffer else 0,
                )
                (
                    probability,
                    effective_threshold,
                    min_fight_count,
                    decision_mode,
                    filter_reason,
                ) = apply_runtime_action_policy(
                    probability=probability,
                    effective_threshold=effective_threshold,
                    min_fight_count=min_fight_count,
                    decision_mode=decision_mode,
                    filter_reason=filter_reason,
                    recent_contact=recent_contact,
                    contact_density=sum(contact_buffer[-NUM_FRAMES:]) if contact_buffer else 0,
                    motion_context=motion_context,
                    keypoint_violence_score=keypoint_violence_score,
                    latest_keypoint_violence_score=latest_keypoint_violence_score,
                    kick_contact=kick_contact,
                )
                motion_context["effective_threshold"] = effective_threshold
                if filter_reason:
                    det.stage1_result = filter_reason
            else:
                probability = 0.0
                
            det.update(probability, threshold=effective_threshold, min_fight=min_fight_count)
            if det.is_alert:
                final_alert_decision_delay = time.perf_counter() - final_flush_start
                perf_stats["alert_decision_delay_times"].append(final_alert_decision_delay)
            display_probability = build_video_display_probability(
                raw_probability=det.current_probability,
                n_persons=final_n_persons,
                interaction=final_interaction,
                has_contact=recent_contact,
                sequence_fill_ratio=1.0,
                threshold=effective_threshold
            )
            det.update_display_probability(display_probability)

            if det.is_alert and time.time() - last_alert_save > COOLDOWN_SECS:
                last_alert_save = time.time()
                alert_second = frame_count / max(src_fps, 1)
                final_alert_save_start = time.perf_counter()
                result = save_to_firebase(
                    final_frame,
                    output_name,
                    video_id,
                    det.current_probability,
                    source_type='video'
                )
                perf_stats["firebase_saves"] += 1
                if result:
                    enqueue_upload_evidence_clip(
                        video_path=video_path,
                        output_name=output_name,
                        video_id=video_id,
                        alert_second=alert_second,
                        firestore_id=result.get('id'),
                        probability=det.current_probability,
                    )
                    socketio.emit('history_update', result)
                final_alert_save_delay = time.perf_counter() - final_alert_save_start
                perf_stats["alert_save_delay_times"].append(final_alert_save_delay)

            pct = int(round(float(det.display_probability) * 100))
            samples_per_second = src_fps / max(YOLO_EVERY, 1)
            final_interaction_detail = best_interaction_detail or {}
            final_closing_speed_px = final_interaction_detail.get("closing_speed_px", "")
            heuristic_rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_id": video_id,
                "frame": frame_count,
                "second": round(frame_count / max(src_fps, 1), 4),
                "persons": final_n_persons,
                "min_person_passed": final_n_persons >= MIN_PERSONS,
                "proximity_score": final_interaction_detail.get("proximity_score", ""),
                "velocity_score": final_interaction_detail.get("velocity_score", ""),
                "overlap_score": final_interaction_detail.get("overlap_score", ""),
                "interaction_score": round(float(final_interaction), 4),
                "interaction_threshold": MIN_INTERACTION_SCORE,
                "interaction_passed": final_interaction >= MIN_INTERACTION_SCORE,
                "center_distance_px": final_interaction_detail.get("center_distance_px", ""),
                "avg_bbox_diagonal_px": final_interaction_detail.get("avg_bbox_diagonal_px", ""),
                "previous_distance_px": final_interaction_detail.get("previous_distance_px", ""),
                "closing_speed_px": final_closing_speed_px,
                "closing_velocity_px_per_second": (
                    round(float(final_closing_speed_px) * samples_per_second, 4)
                    if final_closing_speed_px != "" else ""
                ),
                "overlap_iou": final_interaction_detail.get("overlap_iou", ""),
                "interaction_pair": final_interaction_detail.get("interaction_pair", ""),
                "physical_contact": bool(recent_contact),
                "low_light": bool(low_light),
                "avg_extremity_velocity": motion_context["avg_extremity_velocity"],
                "max_extremity_velocity": motion_context["max_extremity_velocity"],
                "max_leg_velocity": motion_context["max_leg_velocity"],
                "avg_extremity_velocity_per_second": round(
                    motion_context["avg_extremity_velocity"] * samples_per_second, 5
                ),
                "max_extremity_velocity_per_second": round(
                    motion_context["max_extremity_velocity"] * samples_per_second, 5
                ),
                "max_leg_velocity_per_second": round(
                    motion_context["max_leg_velocity"] * samples_per_second, 5
                ),
                "closing_speed_norm": motion_context["closing_speed_norm"],
                "aggressive_motion_score": motion_context["aggressive_motion_score"],
                "motion_passed": motion_context["motion_passed"],
                "static_contact": motion_context["static_contact"],
                "passive_overlap": motion_context["passive_overlap"],
                "effective_threshold": round(float(effective_threshold), 4),
                "decision_mode": f"{decision_mode}FinalFlush",
                "kick_contact": bool(kick_contact),
                "model_probability": round(float(probability), 4),
                "filter_reason": filter_reason,
                "raw_probability": round(float(det.current_probability), 4),
                "display_probability": round(float(det.display_probability), 4),
                "buffer_fight_count": int(sum(det.prediction_buffer)),
                "buffer_size": len(det.prediction_buffer),
                "buffer_requirement": min_fight_count,
                "alert": bool(det.is_alert),
                "new_alert_triggered": bool(det.is_alert),
                "total_alerts": int(det.total_alerts),
                "stage1": str(det.stage1_result),
                "stage2": str(det.stage2_result),
            })
            final_pipeline_elapsed = time.perf_counter() - final_flush_start
            perf_stats["pipeline_times"].append(final_pipeline_elapsed)
            perf_stats["processed_frames"] += 1
            wall_elapsed = max(time.perf_counter() - process_started_at, 1e-6)
            active_elapsed = max(sum(perf_stats["pipeline_times"]), 1e-6)
            final_pipeline_ms = round(final_pipeline_elapsed * 1000.0, 4)
            encoded_frame_bytes = (
                len(video_frames_raw.get(video_id, b"")) +
                len(video_frames_anno.get(video_id, b""))
            )
            qos_metrics = compute_qos_network_metrics(
                "upload_video_final_flush",
                final_pipeline_ms,
                previous_qos_pipeline_ms,
                encoded_bytes=encoded_frame_bytes,
                elapsed_seconds=final_pipeline_elapsed,
                ping_ms="",
            )
            previous_qos_pipeline_ms = final_pipeline_ms
            qos_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_id": video_id,
                "source_type": "upload_video_final_flush",
                "frame": frame_count,
                "second": round(frame_count / max(src_fps, 1), 4),
                "source_fps": round(float(src_fps), 4),
                "yolo_every": YOLO_EVERY,
                "processed_frames": perf_stats["processed_frames"],
                "wall_fps": round(frame_count / wall_elapsed, 4),
                "active_pipeline_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                "yolo_ms": 0.0,
                "heuristic_ms": 0.0,
                "mobilenet_ms": 0.0,
                "cnn_lstm_ms": round(final_lstm_elapsed * 1000.0, 4),
                "encode_ms": "",
                "pipeline_ms": final_pipeline_ms,
                "alert_decision_delay_ms": round(final_alert_decision_delay * 1000.0, 4),
                "alert_save_delay_ms": round(final_alert_save_delay * 1000.0, 4),
                "is_alert": bool(det.is_alert),
                "new_alert_triggered": bool(det.is_alert),
                "decision_mode": f"{decision_mode}FinalFlush",
            }
            qos_row.update(qos_metrics)
            qos_rows.append(qos_row)
            metadata_rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_id": video_id,
                "source_type": "upload_video_final_flush",
                "source_name": video_original_names.get(video_id, os.path.basename(video_path)),
                "camera_ip": "",
                "frame": frame_count,
                "second": round(frame_count / max(src_fps, 1), 4),
                "resolution": f"{frame_w}x{frame_h}",
                "source_fps": round(float(src_fps), 4),
                "target_sample_rate": round(float(TARGET_SAMPLE_RATE), 4),
                "actual_sample_rate": round(float(src_fps) / max(YOLO_EVERY, 1), 4),
                "sequence_window_seconds": round(NUM_FRAMES / max(float(src_fps) / max(YOLO_EVERY, 1), 0.1), 4),
                "processed_fps": round(perf_stats["processed_frames"] / active_elapsed, 4),
                "yolo_every": YOLO_EVERY,
                "base_threshold": round(float(det.threshold), 4),
                "threshold": round(float(effective_threshold), 4),
                "general_threshold": round(float(GENERAL_STAGE1_THRESHOLD), 4),
                "low_light_threshold": round(float(LOW_LIGHT_GENERAL_THRESHOLD), 4),
                "model_override_threshold": round(float(MODEL_OVERRIDE_THRESHOLD), 4),
                "min_persons": MIN_PERSONS,
                "min_interaction_score": MIN_INTERACTION_SCORE,
                "buffer_capacity": det.buffer_size,
                "buffer_size": len(det.prediction_buffer),
                "buffer_requirement": min_fight_count,
                "buffer_fight_count": int(sum(det.prediction_buffer)),
                "persons": final_n_persons,
                "min_person_passed": final_n_persons >= MIN_PERSONS,
                "interaction_score": round(float(final_interaction), 4),
                "interaction_threshold": MIN_INTERACTION_SCORE,
                "interaction_passed": final_interaction >= MIN_INTERACTION_SCORE,
                "physical_contact": bool(recent_contact),
                "contact_density": int(sum(contact_buffer[-NUM_FRAMES:])) if contact_buffer else 0,
                "brightness": lighting_metrics["brightness"],
                "saturation": lighting_metrics["saturation"],
                "brightness_threshold": round(float(DARK_BRIGHTNESS_THRESHOLD), 4),
                "saturation_threshold": round(float(DARK_SATURATION_THRESHOLD), 4),
                "low_light": bool(low_light),
                "keypoint_violence_score": round(float(keypoint_violence_score), 5),
                "aggressive_motion_score": motion_context["aggressive_motion_score"],
                "raw_probability": round(float(det.current_probability), 4),
                "display_probability": round(float(det.display_probability), 4),
                "model_probability": round(float(probability), 4),
                "decision_mode": f"{decision_mode}FinalFlush",
                "filter_reason": filter_reason,
                "stage1": str(det.stage1_result),
                "stage2": str(det.stage2_result),
                "final_decision": "KDRT" if det.is_alert else "Normal",
                "is_alert": bool(det.is_alert),
                "new_alert_triggered": bool(det.is_alert),
            })
            socketio.emit('video_detection_update', {
                'video_id': video_id,
                'status': 'KDRT' if det.is_alert else 'Normal',
                'probability': round(float(det.display_probability), 4),
                'raw_probability': round(float(det.current_probability), 4),
                'percentage': pct,
                'is_alert': bool(det.is_alert),
                'persons': final_n_persons,
                'interaction': round(float(final_interaction), 3),
                'low_light': bool(low_light),
                'aggressive_motion': motion_context["aggressive_motion_score"],
                'decision_mode': decision_mode,
                'stage1': str(det.stage1_result),
                'stage2': str(det.stage2_result),
            })

    except Exception as e:
        print(f"[Video] Error processing {video_id}: {e}")
        socketio.emit('video_error', {'video_id': video_id, 'error': str(e)})
    finally:
        perf_stats["wall_time_seconds"] = time.perf_counter() - process_started_at
        if perf_stats["wall_time_seconds"] > 0:
            perf_stats["wall_fps"] = frame_count / perf_stats["wall_time_seconds"]
        active_seconds = sum(perf_stats["pipeline_times"])
        if active_seconds > 0:
            perf_stats["active_pipeline_fps"] = perf_stats["processed_frames"] / active_seconds
        try:
            perf_csv = write_video_performance_summary(video_id, video_path, perf_stats)
            print(f"[Performance] Summary saved: {perf_csv}")
        except Exception as perf_error:
            print(f"[Performance] Failed to save summary: {perf_error}")
        try:
            qos_csv = write_qos_detail_rows(video_id, video_path, qos_rows)
            if qos_csv:
                print(f"[QoS] Detail saved: {qos_csv}")
        except Exception as qos_error:
            print(f"[QoS] Failed to save detail: {qos_error}")
        try:
            metadata_csv = write_test_metadata_rows(video_id, video_path, metadata_rows)
            if metadata_csv:
                print(f"[Metadata] Detail saved: {metadata_csv}")
            actual_sample_rate = float(src_fps) / max(YOLO_EVERY, 1)
            sync_csv = write_sync_summary(video_id, video_path, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_id": video_id,
                "source_type": "upload_video",
                "source_name": video_original_names.get(video_id, os.path.basename(video_path)),
                "resolution": f"{frame_w}x{frame_h}",
                "source_fps": round(float(src_fps), 4),
                "target_sample_rate": round(float(TARGET_SAMPLE_RATE), 4),
                "actual_sample_rate": round(actual_sample_rate, 4),
                "yolo_every": YOLO_EVERY,
                "sequence_length": NUM_FRAMES,
                "sequence_window_seconds": round(NUM_FRAMES / max(actual_sample_rate, 0.1), 4),
                "base_threshold": round(float(det.threshold), 4),
                "general_threshold": round(float(GENERAL_STAGE1_THRESHOLD), 4),
                "low_light_threshold": round(float(LOW_LIGHT_GENERAL_THRESHOLD), 4),
                "model_override_threshold": round(float(MODEL_OVERRIDE_THRESHOLD), 4),
                "buffer_capacity": det.buffer_size,
                "default_buffer_requirement": MIN_FIGHT_COUNT,
                "min_persons": MIN_PERSONS,
                "min_interaction_score": MIN_INTERACTION_SCORE,
                "contact_density_threshold": CONTACT_DENSITY_THRESHOLD,
                "filters_logged": "filter_reason,decision_mode,stage1,stage2,contact_density",
            })
            print(f"[Metadata] Sync summary saved: {sync_csv}")
        except Exception as metadata_error:
            print(f"[Metadata] Failed to save detail: {metadata_error}")
        try:
            heuristic_csv = write_heuristic_detail_rows(video_id, video_path, heuristic_rows)
            if heuristic_csv:
                print(f"[Heuristic] Detail saved: {heuristic_csv}")
        except Exception as heuristic_error:
            print(f"[Heuristic] Failed to save detail: {heuristic_error}")

        cap.release()
        # Cleanup
        if video_id in video_detectors:
            del video_detectors[video_id]
        if video_id in video_stop_flags:
            del video_stop_flags[video_id]
        video_original_names.pop(video_id, None)
        # Clean up temp file
        try:
            upload_root = os.path.realpath(UPLOAD_FOLDER)
            candidate_path = os.path.realpath(video_path)
            is_temporary_upload = (
                os.path.commonpath([upload_root, candidate_path]) == upload_root
            )
            if is_temporary_upload and os.path.exists(candidate_path):
                os.remove(video_path)
            elif os.path.exists(candidate_path):
                print(f"[Video] Skip cleanup outside tmp_uploads: {candidate_path}")
        except Exception:
            pass
        print(f"[Video] Done: {video_id}")
        socketio.emit('video_done', {'video_id': video_id})


# ── Video Routes ───────────────────────────────────────────────────────────────
@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Accept video file upload, return video_id for processing."""
    if not is_admin_session():
        return jsonify({'error': 'Only an admin can upload videos'}), 403
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error': 'Unsupported file format. Use: mp4, avi, mov, mkv, wmv, flv, webm'}), 400

    video_id  = str(uuid.uuid4())
    filename  = secure_filename(f.filename)
    _, ext    = os.path.splitext(filename)
    save_path = os.path.join(UPLOAD_FOLDER, f"{video_id}{ext}")
    f.save(save_path)
    video_original_names[video_id] = filename
    print(f"[Video] Uploaded: {filename} → {save_path}")
    return jsonify({'video_id': video_id, 'filename': filename})


@app.route('/process_video/<video_id>', methods=['POST'])
def process_video(video_id):
    """Start processing an uploaded video in background thread."""
    if not is_admin_session():
        return jsonify({'error': 'Only an admin can process videos'}), 403
    # Find file
    video_path = None
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(UPLOAD_FOLDER, f"{video_id}.{ext}")
        if os.path.exists(p):
            video_path = p
            break
    if not video_path:
        return jsonify({'error': 'Video not found'}), 404

    # Stop existing processing for this video_id if any
    if video_id in video_stop_flags:
        video_stop_flags[video_id].set()
        time.sleep(0.3)

    stop_event = threading.Event()
    video_stop_flags[video_id] = stop_event

    t = threading.Thread(
        target=video_processing_loop,
        args=(video_id, video_path, stop_event),
        daemon=True
    )
    t.start()
    print(f"[Video] Started processing: {video_id}")
    return jsonify({'status': 'started', 'video_id': video_id})


@app.route('/video_feed/vid_raw/<video_id>')
def video_feed_vid_raw(video_id):
    """MJPEG stream for raw video frames (no skeleton)."""
    return Response(
        video_mjpeg_generator(video_id, video_frames_raw),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/video_feed/vid_anno/<video_id>')
def video_feed_vid_anno(video_id):
    """MJPEG stream for annotated video frames (with skeleton)."""
    return Response(
        video_mjpeg_generator(video_id, video_frames_anno),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/evidence_clip/<path:filename>')
def evidence_clip(filename):
    """Serve local per-alert evidence clips."""
    return send_from_directory(EVIDENCE_FOLDER, filename, mimetype=get_video_mimetype(filename), as_attachment=False, conditional=True)


@app.route('/evidence_download/<path:filename>')
def evidence_download(filename):
    """Download local per-alert evidence clips."""
    return send_from_directory(EVIDENCE_FOLDER, filename, mimetype=get_video_mimetype(filename), as_attachment=True, download_name=filename)


@app.route('/firebase_messaging_config')
def firebase_messaging_config():
    """Expose only public Firebase Web config needed by browser messaging."""
    return jsonify({
        'enabled': FCM_ENABLED,
        'config': FCM_WEB_CONFIG if FCM_ENABLED else {},
        'vapidKey': FCM_VAPID_KEY if FCM_ENABLED else '',
        'reason': '' if FCM_ENABLED else (
            'Complete KDRT_FIREBASE_API_KEY, KDRT_FIREBASE_MESSAGING_SENDER_ID, '
            'KDRT_FIREBASE_APP_ID, and KDRT_FIREBASE_VAPID_KEY'
        ),
    })


@app.route('/firebase_messaging_status')
def firebase_messaging_status():
    """Debug-safe Firebase Messaging status without exposing secret values."""
    return jsonify({
        'enabled': FCM_ENABLED,
        'apiKey': bool(FCM_WEB_CONFIG.get('apiKey')),
        'messagingSenderId': bool(FCM_WEB_CONFIG.get('messagingSenderId')),
        'appId': bool(FCM_WEB_CONFIG.get('appId')),
        'vapidKey': bool(FCM_VAPID_KEY),
        'vapidKeyValid': FCM_VAPID_STATUS['valid'],
        'vapidDecodedBytes': FCM_VAPID_STATUS['decoded_bytes'],
        'vapidError': FCM_VAPID_STATUS['error'],
        'projectId': FCM_WEB_CONFIG.get('projectId', ''),
    })


@app.route('/firebase-messaging-sw.js')
def firebase_messaging_service_worker():
    """Serve Firebase Messaging service worker with .env config embedded."""
    config_json = json.dumps(FCM_WEB_CONFIG if FCM_ENABLED else {})
    enabled_json = json.dumps(bool(FCM_ENABLED))
    script = f"""
var FCM_ENABLED = {enabled_json};
var firebaseConfig = {config_json};
var messagingReady = false;

self.addEventListener('install', function(event) {{
    self.skipWaiting();
}});

self.addEventListener('activate', function(event) {{
    event.waitUntil(self.clients.claim());
}});

function showKdrtNotification(payload) {{
    payload = payload || {{}};
    var notification = payload.notification || {{}};
    var data = payload.data || {{}};
    var title = notification.title || data.title || 'Alert KDRT Terdeteksi';
    var options = {{
        body: notification.body || data.body || 'Sistem mendeteksi indikasi kekerasan.',
        icon: '/assets/Frame 9.png',
        badge: '/assets/Frame 9.png',
        tag: data.history_id || 'kdrt-alert',
        requireInteraction: true,
        data: {{
            url: '/',
            historyId: data.history_id || ''
        }}
    }};
    return self.registration.showNotification(title, options);
}}

if (!FCM_ENABLED) {{
    console.warn('[firebase-messaging-sw] Firebase Messaging config is incomplete');
}} else {{
    try {{
        importScripts('https://www.gstatic.com/firebasejs/10.12.5/firebase-app-compat.js');
        importScripts('https://www.gstatic.com/firebasejs/10.12.5/firebase-messaging-compat.js');
        firebase.initializeApp(firebaseConfig);
        var messaging = firebase.messaging();
        messaging.onBackgroundMessage(showKdrtNotification);
        messagingReady = true;
    }} catch (error) {{
        console.error('[firebase-messaging-sw] Firebase Messaging init failed', error);
    }}
}}

self.addEventListener('push', function(event) {{
    if (messagingReady) return;
    var payload = {{}};
    try {{
        payload = event.data ? event.data.json() : {{}};
    }} catch (error) {{
        payload = {{ notification: {{ title: 'Alert KDRT Terdeteksi', body: event.data ? event.data.text() : '' }} }};
    }}
    event.waitUntil(showKdrtNotification(payload));
}});

self.addEventListener('notificationclick', function(event) {{
    event.notification.close();
    var targetUrl = (event.notification.data && event.notification.data.url) || '/';
    event.waitUntil(
        self.clients.matchAll({{ type: 'window', includeUncontrolled: true }}).then(function(clientList) {{
            for (var i = 0; i < clientList.length; i++) {{
                var client = clientList[i];
                if ('focus' in client) {{
                    return client.focus();
                }}
            }}
            if (self.clients.openWindow) return self.clients.openWindow(targetUrl);
            return null;
        }})
    );
}});
""".strip()
    return Response(script, mimetype='application/javascript')


@app.route('/register_fcm_token', methods=['POST'])
def register_fcm_token():
    """Store a browser FCM token so the backend can send alert notifications."""
    data = request.get_json(silent=True) or {}
    token = (data.get('token') or '').strip()
    if not token:
        return jsonify({'ok': False, 'error': 'Token kosong'}), 400

    try:
        db.collection('fcm_tokens').document(token).set({
            'token': token,
            'user_agent': request.headers.get('User-Agent', ''),
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'enabled': True,
        }, merge=True)
        return jsonify({'ok': True})
    except Exception as error:
        print(f"[FCM] Register token failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/register_web_push_subscription', methods=['POST'])
def register_web_push_subscription():
    """Store the browser Push API subscription for diagnostics/future Web Push sending."""
    data = request.get_json(silent=True) or {}
    subscription = data.get('subscription') or {}
    endpoint = subscription.get('endpoint', '')
    if not endpoint:
        return jsonify({'ok': False, 'error': 'Endpoint subscription kosong'}), 400

    doc_id = base64.urlsafe_b64encode(endpoint.encode('utf-8')).decode('ascii').rstrip('=')
    try:
        db.collection('web_push_subscriptions').document(doc_id).set({
            'endpoint': endpoint,
            'subscription': subscription,
            'user_agent': request.headers.get('User-Agent', ''),
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'enabled': True,
        }, merge=True)
        return jsonify({'ok': True})
    except Exception as error:
        print(f"[WebPush] Register subscription failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/auth/status')
def auth_status():
    return jsonify({
        'authenticated': current_role() in {'admin', 'user'},
        'username': session.get('username', ''),
        'role': current_role(),
    })


def make_auth_resume_token(username, role):
    raw = f"{username}|{role}|{app.secret_key}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def parse_auth_resume_token(token):
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        username, role, secret = raw.split("|", 2)
        if secret != app.secret_key or role not in {"admin", "user"}:
            return None
        return {"username": username, "role": role}
    except Exception:
        return None


@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    user_id, user = get_firestore_user(username)
    if user:
        if not user.get("active", True):
            return jsonify({'ok': False, 'error': 'User is not active'}), 403
        if not check_password_hash(user.get("password_hash", ""), password):
            return jsonify({'ok': False, 'error': 'Incorrect username or password'}), 401
        session['username'] = username
        session['role'] = user.get('role', 'user')
        session['user_id'] = user_id
        return jsonify({
            'ok': True,
            'username': username,
            'role': session['role'],
            'resume_token': make_auth_resume_token(username, session['role']),
        })

    bootstrap_user = AUTH_USERS.get(username)
    if not bootstrap_user or password != bootstrap_user.get('password'):
        return jsonify({'ok': False, 'error': 'Incorrect username or password'}), 401
    session['username'] = username
    session['role'] = bootstrap_user['role']
    session['user_id'] = 'bootstrap'
    return jsonify({
        'ok': True,
        'username': username,
        'role': bootstrap_user['role'],
        'resume_token': make_auth_resume_token(username, bootstrap_user['role']),
    })


@app.route('/auth/resume', methods=['POST'])
def auth_resume():
    data = request.get_json(silent=True) or {}
    parsed = parse_auth_resume_token(data.get('token') or '')
    if not parsed:
        return jsonify({'ok': False, 'error': 'Stored session is invalid'}), 401
    username = parsed['username']
    role = parsed['role']
    if username not in AUTH_USERS:
        _, user = get_firestore_user(username)
        if not user or not user.get('active', True) or user.get('role') != role:
            return jsonify({'ok': False, 'error': 'User is not active or role has changed'}), 401
    session['username'] = username
    session['role'] = role
    session['user_id'] = 'resumed'
    return jsonify({'ok': True, 'username': username, 'role': role})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({'ok': True})


@app.route('/users', methods=['GET'])
def list_users():
    if not is_admin_session():
        return jsonify({'ok': False, 'error': 'Only an admin can view users'}), 403
    try:
        users = []
        for doc in db.collection("app_users").order_by("username").stream():
            users.append(public_user_doc(doc.id, doc.to_dict() or {}))
        return jsonify({'ok': True, 'users': users})
    except Exception as error:
        print(f"[Users] List failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/users', methods=['POST'])
def create_user():
    if not is_admin_session():
        return jsonify({'ok': False, 'error': 'Only an admin can add users'}), 403
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    role = (data.get('role') or 'user').strip()
    active = bool(data.get('active', True))
    if not username or not password:
        return jsonify({'ok': False, 'error': 'Username and password are required'}), 400
    if role not in {'admin', 'user'}:
        return jsonify({'ok': False, 'error': 'Invalid role'}), 400
    _, existing = get_firestore_user(username)
    if existing or username in AUTH_USERS:
        return jsonify({'ok': False, 'error': 'Username is already in use'}), 409
    doc_data = {
        'username': username,
        'password_hash': generate_password_hash(password),
        'role': role,
        'active': active,
        'created_by': session.get('username', ''),
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
    }
    try:
        doc_ref = db.collection('app_users').document()
        doc_ref.set(doc_data)
        return jsonify({'ok': True, 'user': public_user_doc(doc_ref.id, doc_data)})
    except Exception as error:
        print(f"[Users] Create failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/users/<user_id>', methods=['PATCH'])
def update_user(user_id):
    if not is_admin_session():
        return jsonify({'ok': False, 'error': 'Only an admin can modify users'}), 403
    data = request.get_json(silent=True) or {}
    updates = {'updated_at': firestore.SERVER_TIMESTAMP}
    if 'role' in data:
        role = (data.get('role') or '').strip()
        if role not in {'admin', 'user'}:
            return jsonify({'ok': False, 'error': 'Invalid role'}), 400
        updates['role'] = role
    if 'active' in data:
        updates['active'] = bool(data.get('active'))
    if data.get('password'):
        updates['password_hash'] = generate_password_hash(data.get('password'))
    try:
        ref = db.collection('app_users').document(user_id)
        if not ref.get().exists:
            return jsonify({'ok': False, 'error': 'User not found'}), 404
        ref.update(updates)
        doc = ref.get()
        return jsonify({'ok': True, 'user': public_user_doc(doc.id, doc.to_dict() or {})})
    except Exception as error:
        print(f"[Users] Update failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if not is_admin_session():
        return jsonify({'ok': False, 'error': 'Only an admin can delete users'}), 403
    if user_id == session.get('user_id'):
        return jsonify({'ok': False, 'error': 'Cannot delete the account currently logged in'}), 400
    try:
        ref = db.collection('app_users').document(user_id)
        if not ref.get().exists:
            return jsonify({'ok': False, 'error': 'User not found'}), 404
        ref.delete()
        return jsonify({'ok': True, 'id': user_id})
    except Exception as error:
        print(f"[Users] Delete failed: {error}")
        return jsonify({'ok': False, 'error': str(error)}), 500


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ── WebSocket Events ──────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print("[WS] Client connected")
    emit('server_status', {
        'status': 'ready',
        'models_loaded': len(ensemble_models),
        'role': current_role(),
    })

@socketio.on('disconnect')
def on_disconnect():
    print("[WS] Client disconnected")

@socketio.on('connect_cctv')
def on_connect_cctv(data):
    """Start detection stream for a CCTV camera.
    Detection runs in background regardless of monitoring state."""
    ip = data.get('ip', '')
    name = data.get('name', 'CCTV')
    username = data.get('username', '')
    password = data.get('password', '')
    port = data.get('port', '554')
    path = data.get('path', '/Streaming/Channels/101')

    # Skip if already running
    if ip in active_streams and active_streams[ip].is_alive():
        print(f"[WS] Stream already running: {name} ({ip})")
        socketio.emit('cctv_connected', {'ip': ip, 'name': name})
        return

    # Stop existing dead stream for this IP
    if ip in stop_flags:
        stop_flags[ip].set()
        if ip in active_streams:
            active_streams[ip].join(timeout=3)

    # Start new stream
    stop_event = threading.Event()
    stop_flags[ip] = stop_event
    thread = threading.Thread(
        target=detection_loop,
        args=(ip, name, stop_event),
        kwargs={'username': username, 'password': password, 'port': port, 'path': path},
        daemon=True
    )
    active_streams[ip] = thread
    thread.start()

    # Note: cctv_connected is emitted from detection_loop after stream opens
    print(f"[WS] Starting stream: {name} ({ip})")

@socketio.on('start_all_streams')
def on_start_all_streams(data):
    """Start detection for ALL saved CCTV cameras at once.
    Called by frontend on page load to ensure background detection."""
    cameras = data.get('cameras', [])
    print(f"[WS] Starting all streams: {len(cameras)} cameras")
    for cam in cameras:
        on_connect_cctv(cam)

@socketio.on('disconnect_cctv')
def on_disconnect_cctv(data=None):
    """Stop detection stream for a specific camera (e.g. device removed)."""
    if data:
        ip = data.get('ip', '')
    else:
        ip = None

    if ip and ip in stop_flags:
        stop_flags[ip].set()
        if ip in detectors:
            del detectors[ip]
        print(f"[WS] Stopped stream: {ip}")
    else:
        # Stop all streams
        for ip, flag in stop_flags.items():
            flag.set()
        detectors.clear()
        print("[WS] Stopped all streams")

    emit('cctv_disconnected', {'ip': ip})

@socketio.on('stop_video')
def on_stop_video(data):
    """Stop video processing for a given video_id."""
    video_id = data.get('video_id', '') if data else ''
    if video_id and video_id in video_stop_flags:
        video_stop_flags[video_id].set()
        print(f"[WS] Stopped video: {video_id}")
    emit('video_stopped', {'video_id': video_id})

@socketio.on('get_history')
def on_get_history():
    """Fetch history from Firestore."""
    try:
        docs = db.collection('kdrt_detections')\
                 .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                 .limit(100)\
                 .stream()

        history = []
        for doc in docs:
            d = doc.to_dict()
            history.append({
                'id': doc.id,
                'name': d.get('event_name') or d.get('evidence_video_filename') or d.get('filename', ''),
                'time': d.get('time_str', ''),
                'date': d.get('date_str', ''),
                'url': d.get('screenshot_url', ''),
                'evidence_video_url': d.get('evidence_video_url', ''),
                'evidence_video_local_url': d.get('evidence_video_local_url', ''),
                'evidence_video_download_url': (
                    f"/evidence_download/{d.get('evidence_video_filename')}"
                    if d.get('evidence_video_filename') else ''
                ),
                'evidence_video_filename': d.get('evidence_video_filename', ''),
                'evidence_video_duration_seconds': d.get('evidence_video_duration_seconds', 0),
                'evidence_status': d.get('evidence_status', ''),
                'probability': d.get('probability', 0),
                'camera_name': d.get('camera_name', ''),
            })
        emit('history_list', history)
    except Exception as e:
        print(f"[Firebase] History fetch error: {e}")
        emit('history_list', [])

@socketio.on('delete_history')
def on_delete_history(data):
    """Delete a history entry from Firestore and Storage."""
    if not is_admin_session():
        emit('history_delete_denied', {'error': 'Only an admin can delete history'})
        print(f"[Auth] Delete history denied for role={current_role()}")
        return
    doc_id = data.get('id', '')
    try:
        doc_ref = db.collection('kdrt_detections').document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            d = doc.to_dict()
            # Delete from Storage
            filename = d.get('filename', '')
            if filename:
                blob = bucket.blob(f"screenshots/{filename}")
                if blob.exists():
                    blob.delete()
            evidence_filename = d.get('evidence_video_filename', '')
            if evidence_filename:
                evidence_blob = bucket.blob(f"evidence_clips/{evidence_filename}")
                if evidence_blob.exists():
                    evidence_blob.delete()
                local_evidence = os.path.abspath(os.path.join(EVIDENCE_FOLDER, evidence_filename))
                evidence_root = os.path.abspath(EVIDENCE_FOLDER)
                if local_evidence.startswith(evidence_root) and os.path.exists(local_evidence):
                    os.remove(local_evidence)
            # Delete from Firestore
            doc_ref.delete()
            emit('history_deleted', {'id': doc_id})
            print(f"[Firebase] Deleted: {doc_id}")
    except Exception as e:
        print(f"[Firebase] Delete error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KDRT Detection Server")
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    if load_models():
        print(f"\n{'='*60}")
        print(f"  KDRT Detection Server")
        print(f"  http://localhost:{args.port}")
        print(f"{'='*60}\n")
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    else:
        print("Failed to load models. Server not started.")
