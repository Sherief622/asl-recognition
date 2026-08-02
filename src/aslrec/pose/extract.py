"""MediaPipe Hands keypoint extraction with multiprocessing.

Ported from the coursework notebook (parallel extraction with a shared
progress counter) and hardened: results are cached to .npz so a long
extraction is resumable and never has to run twice, and detection failures
are counted and reported instead of silently dropped.

Each image yields 42 features: (x, y) for 21 hand landmarks, or is recorded
as a detection failure when MediaPipe finds no hand.
"""

import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# initialized per worker process by _init_worker
_hands = None

N_FEATURES = 42


def _init_worker():
    global _hands
    import mediapipe

    _hands = mediapipe.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1
    )


def _extract_one(image_path: str) -> list[float] | None:
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    features = []
    for lm in results.multi_hand_landmarks[0].landmark:
        features.extend([lm.x, lm.y])
    return features


def _extract_chunk(args):
    rows, counter, lock = args
    features_list, labels, paths, failures = [], [], [], []
    for path, label in rows:
        features = _extract_one(path)
        if features:
            features_list.append(features)
            labels.append(label)
            paths.append(path)
        else:
            failures.append(path)
        with lock:
            counter.value += 1
    return features_list, labels, paths, failures


def _progress_tracker(counter, total):
    pbar = tqdm(total=total, desc="Keypoint extraction")
    while True:
        value = counter.value
        pbar.n = value
        pbar.refresh()
        if value >= total:
            break
        time.sleep(0.5)
    pbar.close()


def extract_features(
    frame: pd.DataFrame,
    cache_file: Path,
    num_processes: int | None = None,
) -> dict:
    """Extract 42-dim keypoint features for every (path,label) row.

    Returns {"X": ndarray, "y": ndarray, "paths": ndarray,
             "n_failed": int, "failure_rate": float}; cached to cache_file.
    """
    if cache_file.exists():
        # cache is written by this module with plain float/unicode arrays,
        # so no pickle is involved in either direction
        data = np.load(cache_file)
        if set(data["paths"]) | set(data["failed_paths"]) == set(frame["path"]):
            return {
                "X": data["X"], "y": data["y"], "paths": data["paths"],
                "n_failed": int(len(data["failed_paths"])),
                "failure_rate": float(len(data["failed_paths"]) / max(1, len(frame))),
            }

    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    rows = list(zip(frame["path"], frame["label"]))
    chunks = np.array_split(np.arange(len(rows)), num_processes)

    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    tracker = mp.Process(target=_progress_tracker, args=(counter, len(rows)))
    tracker.start()

    args = [([rows[i] for i in chunk], counter, lock) for chunk in chunks if len(chunk)]
    with mp.Pool(processes=num_processes, initializer=_init_worker) as pool:
        results = pool.map(_extract_chunk, args)
    tracker.join()

    X, y, paths, failed = [], [], [], []
    for feats, labels, pths, fails in results:
        X.extend(feats)
        y.extend(labels)
        paths.extend(pths)
        failed.extend(fails)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    paths = np.array(paths)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file, X=X, y=y, paths=paths,
        failed_paths=np.array(failed, dtype=str),
    )
    return {
        "X": X, "y": y, "paths": paths,
        "n_failed": len(failed),
        "failure_rate": len(failed) / max(1, len(rows)),
    }
