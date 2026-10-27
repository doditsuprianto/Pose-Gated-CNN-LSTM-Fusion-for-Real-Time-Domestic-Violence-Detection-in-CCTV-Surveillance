"""
MobileNetV2 Feature Extraction - Local Version (`.py`)
=======================================================
Converted from `local_extraction.ipynb` so it can be run directly locally
as a Python script (without Google Colab / Jupyter).

Extracts visual features from the "Real Life Violence Dataset" using
MobileNetV2, producing `.npy` feature windows + `feature_manifest.csv`
for use in training the CNN-LSTM model.

Usage:
    python local_extraction.py
    python local_extraction.py --input "D:\\Sistem Skripsi\\Dataset"
    python local_extraction.py --output "D:\\Sistem Skripsi\\Local\\features_motion" \
        --batch-start 0 --batch-end 400

Prerequisites:
  - Dataset structure: <input>/Violence (Fight) and <input>/NonViolence (NonFight)
  - MobileNetV2 `imagenet` weights are downloaded automatically on first run (needs internet).
"""

import argparse
import csv
import hashlib
import random
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV2 feature extraction (local)")
    parser.add_argument(
        "--input",
        default=r"D:\Sistem Skripsi\Dataset",
        help="Dataset folder (containing Violence/NonViolence subfolders).",
    )
    parser.add_argument(
        "--output",
        default=r"D:\Sistem Skripsi\Local\features_motion",
        help="Folder where the extracted features are saved.",
    )
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-end", type=int, default=None)
    return parser.parse_args()

# ==== LOCAL PATH (Windows) ====
_args = parse_args()
INPUT_ROOT   = Path(_args.input).resolve()
FEATURE_ROOT = Path(_args.output).resolve()
FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

# ==== Batch mode (optional) ====
# Locally there is no time limit, so leave BATCH_END = None to process ALL videos.
# To split the process: e.g. set --batch-start 0 --batch-end 400, then 400,800, and so on.
BATCH_START = _args.batch_start
BATCH_END   = _args.batch_end

print("Dataset :", INPUT_ROOT, "| exists:", INPUT_ROOT.exists())
print("Output  :", FEATURE_ROOT)
print("GPU     :", tf.config.list_physical_devices("GPU"))
assert INPUT_ROOT.exists(), "Dataset folder not found. Check --input."



# ----------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

video_paths = sorted(
    path for path in INPUT_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
)
print("Number of videos found:", len(video_paths))
for video_path in video_paths[:10]:
    print(video_path)
if not video_paths:
    raise FileNotFoundError("No videos found under INPUT_ROOT. Check the path/folder.")

POSITIVE_FOLDERS = {"violence", "fight"}
NEGATIVE_FOLDERS = {
    "nonviolence", "non-violence", "non_violence",
    "nonfight", "non-fight", "non_fight", "normal",
}

def determine_label(video_path):
    parent_names = {parent.name.lower() for parent in video_path.parents}
    if parent_names & NEGATIVE_FOLDERS:
        return "NonFight"
    if parent_names & POSITIVE_FOLDERS:
        return "Fight"
    return None

classified_videos = []
unclassified_videos = []
for video_path in video_paths:
    label = determine_label(video_path)
    if label is None:
        unclassified_videos.append(video_path)
    else:
        classified_videos.append({"video_path": str(video_path), "label": label})

print("Successfully classified:", len(classified_videos))
print("Unrecognized            :", len(unclassified_videos))
if unclassified_videos:
    print("Sample videos without a label:")
    for video_path in unclassified_videos[:20]:
        print(video_path)
    raise RuntimeError("Some videos have an unrecognized class folder (please place them in the Violence/NonViolence folder).")

classification_check = pd.DataFrame(classified_videos)
print(classification_check["label"].value_counts())
if set(classification_check["label"]) != {"Fight", "NonFight"}:
    raise RuntimeError("Both the Fight and NonFight classes must be present.")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

records = []
for item in classified_videos:
    video_path = Path(item["video_path"])
    records.append({
        "video_id": sha256_file(video_path),
        "source": "RealLife",
        "label": item["label"],
        "video_path": str(video_path),
    })

manifest = pd.DataFrame(records).drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)
print("Identical duplicates excluded:", len(records) - len(manifest))

rng = random.Random(42)
split_rows = []
for label in ["Fight", "NonFight"]:
    indices = manifest.index[manifest["label"] == label].tolist()
    rng.shuffle(indices)
    train_end = int(len(indices) * 0.70)
    validation_end = train_end + int(len(indices) * 0.15)
    for position, index in enumerate(indices):
        if position < train_end:
            split = "train"
        elif position < validation_end:
            split = "validation"
        else:
            split = "test"
        row = manifest.loc[index].to_dict()
        row["split"] = split
        split_rows.append(row)

manifest = pd.DataFrame(split_rows)
MANIFEST_PATH = FEATURE_ROOT / "video_split_manifest.csv"
manifest.to_csv(MANIFEST_PATH, index=False)
print(manifest.groupby(["split", "label"]).size())
print("Video manifest:", MANIFEST_PATH)



# ----------------------------------------------------------------------

SEQUENCE_LENGTH = 20
FEATURE_SIZE = 1280
IMAGE_SIZE = 224


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_windows(total_frames, fps, target_fps, stride):
    step = max(1.0, fps / target_fps) if fps > 0 else 1.0
    sampled = np.unique(np.rint(np.arange(0, total_frames, step)).astype(np.int32))
    if len(sampled) < SEQUENCE_LENGTH:
        sampled = np.pad(sampled, (0, SEQUENCE_LENGTH - len(sampled)), mode="edge")
    starts = list(range(0, max(1, len(sampled) - SEQUENCE_LENGTH + 1), stride))
    last = max(0, len(sampled) - SEQUENCE_LENGTH)
    if starts[-1] != last:
        starts.append(last)
    return [sampled[start:start + SEQUENCE_LENGTH] for start in starts]


def select_windows(windows, maximum):
    if len(windows) <= maximum:
        return windows
    selected = np.rint(np.linspace(0, len(windows) - 1, maximum)).astype(int)
    return [windows[index] for index in np.unique(selected)]


def motion_score(capture, indices):
    previous = None
    values = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            continue
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        if previous is not None:
            values.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    return float(np.mean(values)) if values else 0.0


def select_motion_windows(capture, windows, maximum, candidate_cap):
    candidates = select_windows(windows, candidate_cap)
    scored = [(motion_score(capture, indices), indices) for indices in candidates]
    selected = sorted(
        sorted(scored, key=lambda item: item[0], reverse=True)[:maximum],
        key=lambda item: int(item[1][0]),
    )
    return [item[1] for item in selected], {int(item[1][0]): item[0] for item in selected}


def read_frames(capture, indices):
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            frame = frames[-1].copy() if frames else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return np.asarray(frames, dtype=np.uint8)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



# ----------------------------------------------------------------------

def run_feature_extraction(
    rows,
    output_dir,
    mobilenet_weights="imagenet",
    target_fps=6.0,
    stride=5,
    max_windows=5,
    candidate_cap=15,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir / "extraction_monitor.csv"
    extraction_started = time.perf_counter()
    status_counts = Counter()
    total_videos = len(rows)

    def report_progress(video_number, latest_status, force=False):
        if not force and video_number % 10 != 0:
            return
        elapsed = time.perf_counter() - extraction_started
        rate = video_number / elapsed if elapsed > 0 else 0.0
        remaining = max(total_videos - video_number, 0)
        eta_seconds = remaining / rate if rate > 0 else 0.0
        gpu_text = "GPU n/a"
        gpu_current_mb = None
        gpu_peak_mb = None
        try:
            gpu = tf.config.experimental.get_memory_info("GPU:0")
            gpu_current_mb = round(gpu["current"] / 1024**2, 1)
            gpu_peak_mb = round(gpu["peak"] / 1024**2, 1)
            gpu_text = f"GPU {gpu_current_mb:.1f}/{gpu_peak_mb:.1f} MB (current/peak)"
        except Exception:
            pass
        output_mb = round(sum(
            path.stat().st_size for path in output_dir.rglob("*.npy") if path.is_file()
        ) / 1024**2, 2)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "video_done": video_number,
            "video_total": total_videos,
            "progress_percent": round(100 * video_number / max(total_videos, 1), 2),
            "latest_status": latest_status,
            "processed": status_counts["processed"],
            "skipped_existing": status_counts["skipped_existing"],
            "video_unreadable": status_counts["video_unreadable"],
            "windows_recorded": len(feature_rows),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_seconds, 1),
            "videos_per_minute": round(rate * 60, 2),
            "output_mb": output_mb,
            "gpu_current_mb": gpu_current_mb,
            "gpu_peak_mb": gpu_peak_mb,
        }
        write_header = not monitor_path.exists()
        with monitor_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[MONITOR] {video_number}/{total_videos} "
            f"({row['progress_percent']:.2f}%) | status={latest_status} | "
            f"window={len(feature_rows)} | {rate*60:.2f} video/minute | "
            f"ETA={eta_seconds/60:.1f} minutes | output={output_mb:.2f} MB | {gpu_text}",
            flush=True,
        )

    print(f"[START] {total_videos} videos | {datetime.now().isoformat(timespec='seconds')}")
    extractor = MobileNetV2(
        weights=mobilenet_weights,
        include_top=False,
        pooling="avg",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    feature_rows = []
    for video_number, video in enumerate(
        tqdm(rows, desc="RealLife MobileNetV2", unit="video"), start=1,
    ):
        started = time.perf_counter()
        video_path = Path(video["video_path"])
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        if not capture.isOpened() or total_frames <= 0:
            capture.release()
            feature_rows.append({
                **video, "resolved_video_path": str(video_path), "feature_path": "",
                "window_index": -1, "frame_start": -1, "frame_end": -1, "variant": "original",
                "source_fps": round(fps, 4), "target_fps": target_fps, "motion_score": 0.0,
                "status": "video_unreadable",
                "processing_seconds": round(time.perf_counter() - started, 4),
            })
            status_counts["video_unreadable"] += 1
            report_progress(video_number, "video_unreadable", force=(video_number == total_videos))
            continue

        candidates = candidate_windows(total_frames, fps, target_fps, stride)
        windows, motion_scores = select_motion_windows(capture, candidates, max_windows, candidate_cap)

        planned = [
            output_dir / video["split"] / video["label"] / video["source"]
            / f"{video['video_id']}_w{index:02d}.npy"
            for index in range(len(windows))
        ]

        existing_valid = True
        for feature_path in planned:
            try:
                value = np.load(feature_path, mmap_mode="r")
                if value.shape != (SEQUENCE_LENGTH, FEATURE_SIZE) or not np.isfinite(value).all():
                    existing_valid = False
                    break
            except Exception:
                existing_valid = False
                break

        if existing_valid:
            capture.release()
            for index, (indices, feature_path) in enumerate(zip(windows, planned)):
                feature_rows.append({
                    **video, "resolved_video_path": str(video_path), "feature_path": str(feature_path),
                    "window_index": index, "frame_start": int(indices[0]), "frame_end": int(indices[-1]),
                    "source_fps": round(fps, 4), "target_fps": target_fps,
                    "motion_score": round(motion_scores[int(indices[0])], 6),
                    "variant": "original", "status": "skipped_existing",
                    "processing_seconds": round(time.perf_counter() - started, 4),
                })
            status_counts["skipped_existing"] += 1
            report_progress(video_number, "skipped_existing", force=(video_number == total_videos))
            continue

        frame_batch = np.concatenate([read_frames(capture, indices) for indices in windows], axis=0)
        capture.release()

        features = extractor.predict(
            preprocess_input(frame_batch.astype(np.float32)), batch_size=64, verbose=0,
        )

        for index, indices in enumerate(windows):
            start = index * SEQUENCE_LENGTH
            value = features[start:start + SEQUENCE_LENGTH].astype(np.float16)
            feature_path = planned[index]
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, value)
            feature_rows.append({
                **video, "resolved_video_path": str(video_path), "feature_path": str(feature_path),
                "window_index": index, "frame_start": int(indices[0]), "frame_end": int(indices[-1]),
                "source_fps": round(fps, 4), "target_fps": target_fps,
                "motion_score": round(motion_scores[int(indices[0])], 6),
                "variant": "original", "status": "processed",
                "processing_seconds": round(time.perf_counter() - started, 4),
            })

        status_counts["processed"] += 1
        report_progress(video_number, "processed", force=(video_number == total_videos))

        if video_number % 50 == 0:
            write_csv(output_dir / "feature_manifest_partial.csv", feature_rows)

    write_csv(output_dir / "feature_manifest.csv", feature_rows)
    total_seconds = time.perf_counter() - extraction_started
    print(
        f"[DONE] videos={total_videos} | windows={len(feature_rows)} | "
        f"duration={total_seconds/60:.2f} minutes | monitor={monitor_path}",
        flush=True,
    )
    return pd.DataFrame(feature_rows)



# ----------------------------------------------------------------------

rows = manifest.to_dict("records")
if BATCH_END is not None:
    rows = rows[BATCH_START:BATCH_END]
else:
    rows = rows[BATCH_START:]
print("Videos processed in this run:", len(rows))

feature_manifest = run_feature_extraction(
    rows=rows,
    output_dir=FEATURE_ROOT,
    mobilenet_weights="imagenet",
    target_fps=6.0,
    stride=5,
    max_windows=5,
    candidate_cap=15,
)



# ----------------------------------------------------------------------

print("=== EXTRACTION SUMMARY ===")
print("Number of windows:", len(feature_manifest))
print(feature_manifest["source"].value_counts())
print(feature_manifest["status"].value_counts())
print(feature_manifest.groupby(["split", "label"]).size())

assert (FEATURE_ROOT / "feature_manifest.csv").exists()
(FEATURE_ROOT / "EXTRACTION_SUCCESS.txt").write_text("success", encoding="utf-8")

monitor_path = FEATURE_ROOT / "extraction_monitor.csv"
if monitor_path.exists():
    monitor = pd.read_csv(monitor_path)
    print(monitor.tail(10))
    print("Monitor snapshot:", monitor_path)

print("Output feature size (MB):", round(sum(
    p.stat().st_size for p in FEATURE_ROOT.rglob("*.npy")
) / 1024**2, 2))
print("Finished at:", datetime.now().isoformat(timespec="seconds"))
print("feature_manifest.csv for training is located at:", FEATURE_ROOT / "feature_manifest.csv")
