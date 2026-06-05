from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

EMOTION_MAP: Dict[str, Dict[str, float]] = {
    "😊": {"valence": 1.0, "arousal": 0.2},
    "😰": {"valence": -1.0, "arousal": 0.9},
    "😡": {"valence": -1.0, "arousal": 0.8},
    "😢": {"valence": -0.8, "arousal": 0.4},
    "neutral": {"valence": 0.0, "arousal": 0.0},
    "none": {"valence": 0.0, "arousal": 0.0},
    "": {"valence": 0.0, "arousal": 0.0},
}


def normalize_stem(filename: str) -> str:
    """Extract pure alphanumeric lowercase stem for high-tolerance matching."""
    if not filename:
        return ""
    stem = Path(filename).stem
    return re.sub(r"[^a-zA-Z0-9]", "", stem).lower()


def try_parse_float(x: Any) -> Optional[float]:
    """Parse float; returns None for missing/NaN/unparseable."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


def parse_text_valence(x: Any) -> Tuple[float, str]:
    """
    Parse transcript-side valence:
    - numeric: use directly
    - label: map to {-1,0,1} with simple rules
    Returns (valence, source)
    """
    v = try_parse_float(x)
    if v is not None:
        return float(v), "numeric"

    s = str(x or "").lower()
    if any(k in s for k in ("positive", "joy", "happy")):
        return 1.0, "label"
    if any(k in s for k in ("negative", "stress", "anger", "sad", "fear", "anx", "disgust")):
        return -1.0, "label"
    return 0.0, "label"


def parse_voice_valence_arousal(x: Any) -> Tuple[float, float, str]:
    """
    Parse voice-side emotion tag:
    - If numeric: treat as valence (arousal unknown -> 0); source 'numeric'
    - Else: map via EMOTION_MAP; default neutral
    Returns (valence, arousal, source)
    """
    v = try_parse_float(x)
    if v is not None:
        return float(v), 0.0, "numeric"

    key = str(x or "").strip()
    if key in EMOTION_MAP:
        m = EMOTION_MAP[key]
        return float(m["valence"]), float(m["arousal"]), "map"

    # Try normalized textual keys
    key_l = key.lower()
    if key_l in EMOTION_MAP:
        m = EMOTION_MAP[key_l]
        return float(m["valence"]), float(m["arousal"]), "map"

    return 0.0, 0.0, "map"


def build_metadata_and_timeline(input_dir: Path) -> Tuple[Dict[str, str], Dict[str, Dict[int, pd.Timestamp]]]:
    """
    Map uid -> sound_file, and build timeline uid -> {original_index: datetime} from *_Emotion.txt JSONL.
    """
    print("📅 Building timeline mappings from raw transcripts...")
    idx_files = list(input_dir.rglob("*_Index.json"))
    uid_to_audio: Dict[str, str] = {}
    file_to_uid: Dict[str, str] = {}

    for idx_f in idx_files:
        try:
            with open(idx_f, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                uid = item.get("uid")
                if uid:
                    uid_to_audio[str(uid)] = str(item.get("sound_file", ""))
                    if item.get("transcript_sound"):
                        file_to_uid[str(item.get("transcript_sound"))] = str(uid)
        except Exception:
            continue

    time_map: Dict[str, Dict[int, pd.Timestamp]] = {}
    txt_files = list(input_dir.rglob("*_Emotion.txt"))
    if not txt_files:
        txt_files = [f for f in input_dir.rglob("*.txt") if "Index" not in f.name and "Stockprice" not in f.name]

    for txt_f in txt_files:
        uid = file_to_uid.get(txt_f.name, txt_f.stem)
        if uid not in time_map:
            time_map[uid] = {}

        try:
            with open(txt_f, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    ts_str = record.get("timestamp")
                    if ts_str:
                        try:
                            time_map[uid][int(line_idx)] = pd.to_datetime(ts_str)
                        except Exception:
                            pass
        except Exception:
            pass

    return uid_to_audio, time_map


def extract_acoustic_features(y_voice: np.ndarray, sr: int, min_voice_sec: float) -> Dict[str, float]:
    """
    Extract prosodic and voice-quality proxies from a voiced slice.
    - f0_mean / f0_std via pYIN
    - energy_mean via RMS
    - hnr_proxy via harmonic-vs-residual energy ratio (proxy)
    """
    out = {
        "f0_mean": np.nan,
        "f0_std": np.nan,
        "energy_mean": np.nan,
        "hnr_proxy": np.nan
    }
    if y_voice is None or len(y_voice) < int(min_voice_sec * sr):
        return out

    # Pitch (pYIN)
    try:
        f0, _, _ = librosa.pyin(y_voice, fmin=65, fmax=2093, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]
        if f0_voiced.size > 0:
            out["f0_mean"] = float(np.mean(f0_voiced))
            out["f0_std"] = float(np.std(f0_voiced))
    except Exception:
        pass

    
    try:
        y_harm = librosa.effects.harmonic(y_voice)
        y_res = y_voice - y_harm
        out["hnr_proxy"] = float(np.sum(y_harm ** 2) / (np.sum(y_res ** 2) + 1e-9))
    except Exception:
        pass

    # RMS energy
    try:
        out["energy_mean"] = float(np.mean(librosa.feature.rms(y=y_voice)[0]))
    except Exception:
        pass

    return out


def resolve_slice_times(
    row: pd.Series,
    uid_time_map: Dict[int, pd.Timestamp],
    t0_dt: pd.Timestamp,
    default_char_sec: float = 0.06,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve slice start/end in seconds relative to t0_dt.
    Uses turn spans if available: [original_index, turn_end_index].
    """
    start_idx = row.get("original_index", None)
    end_idx = row.get("turn_end_index", None)

    # start timestamp
    start_dt = None
    if start_idx is not None and not pd.isna(start_idx):
        start_dt = uid_time_map.get(int(start_idx))

    if start_dt is None:
        ts = row.get("timestamp", None)
        try:
            start_dt = pd.to_datetime(ts) if ts is not None and ts != "" else None
        except Exception:
            start_dt = None

    if start_dt is None:
        return None, None

    # end timestamp: prefer (end_idx + 1) if available
    end_dt = None
    if end_idx is not None and not pd.isna(end_idx):
        end_dt = uid_time_map.get(int(end_idx) + 1)

    if end_dt is None:
        # fallback to next line after start index (legacy)
        if start_idx is not None and not pd.isna(start_idx):
            end_dt = uid_time_map.get(int(start_idx) + 1)

    if end_dt is None:
        # fallback: explicit timestamp_end if provided
        ts_end = row.get("timestamp_end", None)
        try:
            end_dt = pd.to_datetime(ts_end) if ts_end is not None and ts_end != "" else None
        except Exception:
            end_dt = None

    # last resort: estimate by text length
    if end_dt is None:
        text_len = len(str(row.get("text", "")))
        est_dur = max(0.8, text_len * default_char_sec)
        start_sec = max(0.0, (start_dt - t0_dt).total_seconds())
        return start_sec, start_sec + est_dur

    start_sec = max(0.0, (start_dt - t0_dt).total_seconds())
    end_sec = max(0.0, (end_dt - t0_dt).total_seconds())
    if end_sec <= start_sec:
        end_sec = start_sec + 0.8
    return start_sec, end_sec


def process_uid_audio(args: Tuple[str, pd.DataFrame, Path, pd.Timestamp, Dict[int, pd.Timestamp], Dict[str, Any]]) -> pd.DataFrame:
    """Worker function: process all rows for one uid/audio file."""
    uid, group_df, audio_path, t0_dt, uid_time_map, cfg = args
    sr = int(cfg["sr"])
    top_db = float(cfg["top_db"])
    min_voice_sec = float(cfg["min_voice_sec"])

    # Load audio once per uid
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        audio_duration = librosa.get_duration(y=y, sr=sr_loaded)
    except Exception:
        # Fill NaN on failure
        for col in ["f0_mean", "f0_std", "energy_mean", "hnr_proxy", "harmonic_residual_proxy", "slice_start_sec", "slice_end_sec", "trimmed_sec"]:
            group_df[col] = np.nan
        return group_df

    results: List[pd.Series] = []
    for _, row in group_df.iterrows():
        start_end = resolve_slice_times(row, uid_time_map, t0_dt)
        start_sec, end_sec = start_end
        if start_sec is None or end_sec is None:
            row["f0_mean"] = np.nan
            row["f0_std"] = np.nan
            row["energy_mean"] = np.nan
            row["hnr_proxy"] = np.nan
            row["slice_start_sec"] = np.nan
            row["slice_end_sec"] = np.nan
            row["trimmed_sec"] = np.nan
            results.append(row)
            continue

        # Boundary safety
        start_sec = float(min(max(0.0, start_sec), max(0.0, audio_duration - 0.1)))
        end_sec = float(min(max(start_sec + 0.1, end_sec), audio_duration))

        start_sample = int(start_sec * sr_loaded)
        end_sample = int(end_sec * sr_loaded)
        y_slice = y[start_sample:end_sample]

        # Trim silence inside slice
        try:
            y_trim, _ = librosa.effects.trim(y_slice, top_db=top_db)
        except Exception:
            y_trim = y_slice

        feats = extract_acoustic_features(y_trim, sr_loaded, min_voice_sec=min_voice_sec)

        row["f0_mean"] = feats["f0_mean"]
        row["f0_std"] = feats["f0_std"]
        row["energy_mean"] = feats["energy_mean"]
        row["hnr_proxy"] = feats["hnr_proxy"]
        row["harmonic_residual_proxy"] = feats["hnr_proxy"]

        row["slice_start_sec"] = round(start_sec, 3)
        row["slice_end_sec"] = round(end_sec, 3)
        row["trimmed_sec"] = round(len(y_trim) / sr_loaded, 3) if y_trim is not None else np.nan

        results.append(row)

    return pd.DataFrame(results)


def add_multimodal_emotion_features(df: pd.DataFrame, scale_text_to_unit: bool = True) -> pd.DataFrame:
    """Compute voice/text valence and the (scaled) incongruence score."""
    print("🧠 Quantifying multimodal emotion (valence/arousal) and incongruence...")

    # Parse valence/arousal
    voice_vals: List[float] = []
    voice_arousals: List[float] = []
    voice_src: List[str] = []
    text_vals_raw: List[float] = []
    text_src: List[str] = []

    for _, row in df.iterrows():
        v_val, v_ar, v_s = parse_voice_valence_arousal(row.get("emotion", ""))
        t_val, t_s = parse_text_valence(row.get("text_emotion", ""))
        voice_vals.append(v_val)
        voice_arousals.append(v_ar)
        voice_src.append(v_s)
        text_vals_raw.append(t_val)
        text_src.append(t_s)

    df = df.copy()
    df["voice_valence_raw"] = voice_vals
    df["voice_arousal"] = voice_arousals
    df["voice_valence_source"] = voice_src
    df["text_valence_raw"] = text_vals_raw
    df["text_valence_source"] = text_src

    # Scale text valence into [-1,1] if needed (when numeric scores are on a narrow range)
    df["text_valence"] = df["text_valence_raw"].astype(float)
    df["text_valence_scale"] = 1.0

    if scale_text_to_unit and df["text_valence_raw"].notna().any():
        # Robust scale factor: 99th percentile of |valence|
        scale = float(df["text_valence_raw"].abs().quantile(0.99))
        if 1e-6 < scale < 0.5:
            df["text_valence"] = (df["text_valence_raw"] / scale).clip(-1.0, 1.0)
            df["text_valence_scale"] = scale
        else:
            df["text_valence"] = df["text_valence_raw"].clip(-1.0, 1.0)

    # For voice valence: keep as-is (already in a comparable range for emoji map)
    df["voice_valence"] = df["voice_valence_raw"].astype(float)

    # Incongruence as absolute distance in (scaled) valence
    df["emotion_incongruence"] = (df["voice_valence"] - df["text_valence"]).abs()

    valid_count = int((df["emotion_incongruence"] > 0).sum())
    print(f"✅ Incongruence > 0: {valid_count} / {len(df)}")

    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 02: extract acoustic features, cached text/voice scores, and acoustic-friction measures.")
    p.add_argument("--data_dir", type=str, default="data", help="Input data directory.")
    p.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    p.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    p.add_argument("--min_voice_sec", type=float, default=0.20, help="Minimum voiced duration after trimming.")
    p.add_argument("--top_db", type=float, default=30.0, help="Trim threshold for silence removal (librosa.effects.trim).")
    p.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = os.cpu_count()).")
    p.add_argument("--scale_text_to_unit", action="store_true", help="Scale numeric text valence to [-1,1] using robust percentile scaling.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    base_dir = Path(".")
    input_dir = base_dir / args.data_dir
    results_dir = base_dir / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Support both the legacy typo name and the corrected name
    input_csv_typo = results_dir / "stress_pairs_with_transcipts.csv"
    input_csv_fixed = results_dir / "stress_pairs_with_transcripts.csv"
    if input_csv_typo.exists():
        input_csv = input_csv_typo
    elif input_csv_fixed.exists():
        input_csv = input_csv_fixed
    else:
        raise FileNotFoundError(f"Missing Step 01 output: {input_csv_typo} (or {input_csv_fixed})")

    output_csv = results_dir / "stress_pairs_with_acoustics.csv"

    cfg = {
        "sr": int(args.sr),
        "min_voice_sec": float(args.min_voice_sec),
        "top_db": float(args.top_db),
    }

    print(f"🎧 Step 02 started\nCWD: {os.getcwd()}\nInput: {input_csv}\nOut: {output_csv}")

    full_df = pd.read_csv(input_csv)
    df = full_df[full_df["pair_id"].notna()].copy()
    if df.empty:
        print("⚠️ No valid pairs found in CSV. Exiting.")
        return

    uid_to_audio, time_map = build_metadata_and_timeline(input_dir)

    # Scan audio files
    valid_audio_exts = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg"}
    audio_file_paths: Dict[str, Path] = {}
    if input_dir.exists():
        for f in input_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_audio_exts:
                audio_file_paths[f.name] = f
                audio_file_paths[normalize_stem(f.name)] = f

    tasks: List[Tuple[str, pd.DataFrame, Path, pd.Timestamp, Dict[int, pd.Timestamp], Dict[str, Any]]] = []
    for uid, g in df.groupby("uid", sort=False):
        sound_file = uid_to_audio.get(str(uid), "")
        if not sound_file:
            continue

        audio_path = audio_file_paths.get(sound_file) or audio_file_paths.get(normalize_stem(sound_file))
        if not audio_path:
            continue

        uid_time_map = time_map.get(str(uid), {})
        if not uid_time_map:
            continue

        t0_dt = min(uid_time_map.values())
        tasks.append((str(uid), g.copy(), audio_path, t0_dt, uid_time_map, cfg))

    if not tasks:
        print("❌ No uid tasks prepared (missing audio or timestamps).")
        return

    workers = int(args.workers) if int(args.workers) > 0 else (os.cpu_count() or 1)
    print(f"🚀 Dispatching parallel audio extraction for {len(tasks)} files (workers={workers})...")

    processed: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_uid_audio, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting Audio"):
            try:
                processed.append(fut.result())
            except Exception as e:
                print(f"⚠️ Worker error for uid={futures.get(fut)}: {e}")

    if not processed:
        print("❌ No data processed successfully.")
        return

    final_df = pd.concat(processed, ignore_index=True)
    final_df = add_multimodal_emotion_features(final_df, scale_text_to_unit=bool(args.scale_text_to_unit))

    final_df.to_csv(output_csv, index=False)
    print("\n✅ Step 02 complete.")
    print(f"   -> {output_csv}")
    print(f"   Rows: {len(final_df)} | Pairs: {final_df['pair_id'].nunique()}")


if __name__ == "__main__":
    main()
