from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


FINANCE_STRESS_LEXICON = {
    "downgrade", "miss", "shortfall", "headwind", "margin pressure",
    "supply chain", "inflation", "recession", "cash burn", "debt",
    "default", "litigation", "lawsuit", "regulatory", "sec", "subpoena",
    "layoff", "restructuring", "severance", "bankrupt", "chapter 11",
    "liquidity", "covenant", "guidance cut", "write-down", "impairment"
}

RE_ANALYST = re.compile(r"(analyst|moderator|question)", re.IGNORECASE)
RE_OPERATOR = re.compile(r"(operator)", re.IGNORECASE)


def normalize_stem(filename: str) -> str:
    """Extract alphanumeric lowercase stem for tolerant file matching."""
    if not filename:
        return ""
    stem = Path(filename).stem
    return re.sub(r"[^a-zA-Z0-9]", "", stem).lower()


def try_parse_float(x: Any) -> Optional[float]:
    """Parse float if possible; returns None for missing/NaN/unparseable."""
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


def infer_turn_role(speaker: Any) -> str:
    """
    Infer role from speaker string.
    Returns one of: {"analyst","exec","other"}.
    """
    s = str(speaker or "")
    if RE_OPERATOR.search(s):
        return "other"
    if RE_ANALYST.search(s):
        return "analyst"
    # Default: management/executive side
    return "exec"


def check_lexicon_trigger(text: str, lexicon: Iterable[str]) -> int:
    t = (text or "").lower()
    return int(any(term in t for term in lexicon))


def trigger_from_text_emotion(text_emotion: Any, numeric_threshold: float = 0.0) -> int:
    """
    Decide whether a segment is 'stress/negative' from transcript-side emotion annotation.

    - If numeric: treat as valence (negativity if < numeric_threshold).
    - Else: keyword match on common negative/stress labels.
    """
    v = try_parse_float(text_emotion)
    if v is not None:
        return int(v < numeric_threshold)

    s = str(text_emotion or "").lower()
    neg_keys = ("stress", "negative", "anger", "sad", "fear", "anx", "disgust")
    return int(any(k in s for k in neg_keys))


def aggregate_text_emotion(values: List[Any]) -> Any:
    """
    Aggregate per-utterance text_emotion into a turn-level value.

    - If most are numeric: return mean numeric.
    - Else: return the mode (most frequent) string (lowercased).
    """
    nums: List[float] = []
    strs: List[str] = []
    for x in values:
        v = try_parse_float(x)
        if v is not None:
            nums.append(v)
        else:
            s = str(x or "").strip().lower()
            if s != "":
                strs.append(s)

    if len(nums) >= max(1, int(0.6 * len(values))):
        return float(np.mean(nums)) if nums else ""

    if not strs:
        return ""
    # mode (ties broken by first occurrence)
    counts: Dict[str, int] = {}
    best = strs[0]
    best_c = 0
    for s in strs:
        counts[s] = counts.get(s, 0) + 1
        if counts[s] > best_c:
            best, best_c = s, counts[s]
    return best


def aggregate_voice_emotion(values: List[Any]) -> str:
    """Mode aggregation for emoji/label voice emotion field."""
    strs = [str(x or "").strip() for x in values if str(x or "").strip() != ""]
    if not strs:
        return ""
    counts: Dict[str, int] = {}
    best = strs[0]
    best_c = 0
    for s in strs:
        counts[s] = counts.get(s, 0) + 1
        if counts[s] > best_c:
            best, best_c = s, counts[s]
    return best


@dataclass
class Turn:
    uid: str
    ticker: str
    event_start_et: str
    transcript_file: str

    turn_index: int
    turn_role: str  # analyst / exec
    speakers: List[str]

    start_index: int
    end_index: int
    start_timestamp: Any
    end_timestamp: Any

    text: str
    text_emotion: Any
    emotion: str

    trigger_lexicon: int
    trigger_emotion: int
    analyst_triggered: int


def parse_all_indices(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_Index.json files and attach source directory for relative paths."""
    idx_files = list(data_dir.rglob("*_Index.json"))
    all_indices: List[Dict[str, Any]] = []
    for idx_f in idx_files:
        try:
            with open(idx_f, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    item = dict(item)
                    item["_index_dir"] = str(idx_f.parent)
                    all_indices.append(item)
        except Exception as e:
            print(f"⚠️ Error reading {idx_f.name}: {e}")
    return all_indices


def calculate_dataset_stats(all_indices: List[Dict[str, Any]], search_dir: Path, min_audio_size_bytes: int) -> Dict[str, Any]:
    """Report dataset integrity: how many index entries have usable audio and stock files."""
    print("🎵 Analyzing audio and stock file mapping...")
    stats: Dict[str, Any] = {
        "expected_events_from_index": int(len(all_indices)),
        "valid_audio_files": 0,
        "missing_audio_files": 0,
        "abnormal_audio_ignored": 0,
        "events_with_stock_price": 0,
    }

    valid_audio_exts = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg"}
    audio_file_paths: Dict[str, Path] = {}

    if search_dir.exists():
        for f in search_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_audio_exts:
                audio_file_paths[f.name] = f
                audio_file_paths[normalize_stem(f.name)] = f

    for item in all_indices:
        stock_file = item.get("stock_price", "")
        if stock_file:
            stock_path = Path(item.get("_index_dir", "")) / stock_file
            if stock_path.exists():
                stats["events_with_stock_price"] += 1

        target_audio = item.get("sound_file", "")
        if not target_audio:
            continue

        matched = audio_file_paths.get(target_audio) or audio_file_paths.get(normalize_stem(target_audio))
        if not matched:
            stats["missing_audio_files"] += 1
            continue
        if matched.stat().st_size < min_audio_size_bytes:
            stats["abnormal_audio_ignored"] += 1
            continue
        stats["valid_audio_files"] += 1

    print(
        f"✅ File analysis complete: {stats['valid_audio_files']} valid audios, "
        f"{stats['missing_audio_files']} missing audios."
    )
    return stats


def load_native_nasdaq_data(all_indices: List[Dict[str, Any]], data_dir: Path) -> pd.DataFrame:
    """
    Load transcript JSONL/TXT records into a flat dataframe.
    Expected per-line JSON keys: speaker, timestamp, text, text_emotion, emotion, events.
    """
    print("📄 Loading transcript text data...")
    file_to_meta: Dict[str, Dict[str, Any]] = {}
    for item in all_indices:
        if item.get("transcript"):
            file_to_meta[item["transcript"]] = item
        if item.get("transcript_sound"):
            file_to_meta[item["transcript_sound"]] = item
        if item.get("uid"):
            file_to_meta[item["uid"]] = item

    txt_files = list(data_dir.rglob("*_Emotion.txt"))
    if not txt_files:
        txt_files = list(data_dir.rglob("*.jsonl")) + list(data_dir.rglob("*.txt"))
        txt_files = [f for f in txt_files if "Index" not in f.name and "Stockprice" not in f.name]

    rows: List[Dict[str, Any]] = []
    for txt_f in txt_files:
        meta = file_to_meta.get(txt_f.name) or file_to_meta.get(txt_f.stem) or {}
        ticker = meta.get("ticker", "UNKNOWN")
        real_uid = meta.get("uid", txt_f.stem)
        event_start = meta.get("event_start_et", "")

        try:
            with open(txt_f, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rows.append(
                        {
                            "uid": real_uid,
                            "ticker": ticker,
                            "event_start_et": event_start,
                            "original_index": int(line_idx),
                            "speaker": str(record.get("speaker", "")),
                            "timestamp": record.get("timestamp", None),
                            "text": str(record.get("text", "")),
                            # keep raw; downstream will parse numeric vs label
                            "text_emotion": record.get("text_emotion", ""),
                            "emotion": record.get("emotion", ""),
                            "events": record.get("events", ""),
                            "transcript_file": txt_f.name,
                        }
                    )
        except Exception as e:
            print(f"⚠️ Error reading {txt_f.name}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["uid", "original_index"], inplace=True, kind="stable")
        df.reset_index(drop=True, inplace=True)
    print(f"✅ Loaded {len(df)} transcript records.")
    return df


def build_turns_for_uid(
    uid_df: pd.DataFrame,
    lexicon: Iterable[str],
    merge_by_role_only: bool = True,
    numeric_valence_threshold: float = 0.0,
) -> List[Turn]:
    """Turn construction inside a single call (uid)."""
    if uid_df.empty:
        return []

    uid_df = uid_df.copy().reset_index(drop=True)
    uid_df["role"] = uid_df["speaker"].apply(infer_turn_role)

    # Identify Q&A start: first analyst-like speaker (fallback: whole call)
    analyst_mask = uid_df["role"].eq("analyst")
    if analyst_mask.any():
        qa_start = int(analyst_mask.idxmax())
        qa_df = uid_df.loc[qa_start:].copy().reset_index(drop=True)
    else:
        qa_df = uid_df.copy()

    # Drop 'other' (operator etc.) for turn formation
    qa_df = qa_df[qa_df["role"].ne("other")].copy().reset_index(drop=True)
    if qa_df.empty:
        return []

    turns: List[Turn] = []
    cur_role: Optional[str] = None
    cur_speakers: List[str] = []
    cur_text: List[str] = []
    cur_text_emotions: List[Any] = []
    cur_emotions: List[Any] = []
    cur_start_idx: Optional[int] = None
    cur_end_idx: Optional[int] = None
    cur_start_ts: Any = None
    cur_end_ts: Any = None
    cur_transcript_file: str = str(qa_df.loc[0, "transcript_file"])
    ticker = str(qa_df.loc[0, "ticker"])
    event_start = str(qa_df.loc[0, "event_start_et"])

    def flush(turn_index: int) -> None:
        nonlocal cur_role, cur_speakers, cur_text, cur_text_emotions, cur_emotions
        nonlocal cur_start_idx, cur_end_idx, cur_start_ts, cur_end_ts

        if cur_role is None or cur_start_idx is None or cur_end_idx is None:
            return

        text_joined = " ".join([t.strip() for t in cur_text if str(t).strip() != ""]).strip()
        agg_text_emotion = aggregate_text_emotion(cur_text_emotions)
        agg_voice_emotion = aggregate_voice_emotion(cur_emotions)

        trig_lex = check_lexicon_trigger(text_joined, lexicon)
        # emotion trigger: if any utterance in the turn indicates stress/negative
        trig_emo = int(any(trigger_from_text_emotion(x, numeric_threshold=numeric_valence_threshold) for x in cur_text_emotions))

        analyst_trig = int(cur_role == "analyst" and (trig_lex == 1 or trig_emo == 1))

        turns.append(
            Turn(
                uid=str(uid_df.loc[0, "uid"]),
                ticker=ticker,
                event_start_et=event_start,
                transcript_file=cur_transcript_file,
                turn_index=turn_index,
                turn_role=cur_role,
                speakers=list(dict.fromkeys(cur_speakers)),  # preserve order, unique
                start_index=int(cur_start_idx),
                end_index=int(cur_end_idx),
                start_timestamp=cur_start_ts,
                end_timestamp=cur_end_ts,
                text=text_joined,
                text_emotion=agg_text_emotion,
                emotion=agg_voice_emotion,
                trigger_lexicon=trig_lex if cur_role == "analyst" else 0,
                trigger_emotion=trig_emo if cur_role == "analyst" else 0,
                analyst_triggered=analyst_trig,
            )
        )

        # reset
        cur_role = None
        cur_speakers = []
        cur_text = []
        cur_text_emotions = []
        cur_emotions = []
        cur_start_idx = None
        cur_end_idx = None
        cur_start_ts = None
        cur_end_ts = None

    turn_index = 0
    for _, row in qa_df.iterrows():
        role = str(row["role"])
        speaker = str(row.get("speaker", ""))

        start_new = False
        if cur_role is None:
            start_new = True
        else:
            if merge_by_role_only:
                start_new = role != cur_role
            else:
                start_new = (role != cur_role) or (speaker != (cur_speakers[-1] if cur_speakers else ""))

        if start_new:
            flush(turn_index)
            turn_index += 1
            cur_role = role
            cur_speakers = [speaker]
            cur_text = [str(row.get("text", ""))]
            cur_text_emotions = [row.get("text_emotion", "")]
            cur_emotions = [row.get("emotion", "")]
            cur_start_idx = int(row["original_index"])
            cur_end_idx = int(row["original_index"])
            cur_start_ts = row.get("timestamp", None)
            cur_end_ts = row.get("timestamp", None)
            cur_transcript_file = str(row.get("transcript_file", cur_transcript_file))
        else:
            cur_speakers.append(speaker)
            cur_text.append(str(row.get("text", "")))
            cur_text_emotions.append(row.get("text_emotion", ""))
            cur_emotions.append(row.get("emotion", ""))
            cur_end_idx = int(row["original_index"])
            cur_end_ts = row.get("timestamp", cur_end_ts)

    flush(turn_index)

    # Re-number turn_index to be consecutive starting from 0 (flush increments first)
    for i, t in enumerate(turns):
        t.turn_index = i

    return turns


def extract_pairs(
    turns: List[Turn],
    keep_all_pairs: bool = True,
    pair_id_start: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    """
    Build Analyst→Executive adjacency pairs from a turn sequence.

    Parameters
    ----------
    turns:
        Turn sequence within a single call (uid).
    keep_all_pairs:
        If True, emit all Analyst→Executive pairs (triggered or not).
        If False, emit only stress-triggered pairs (recommended for compute efficiency).
    pair_id_start:
        Global pair-id counter start value. This ensures unique pair_id across calls.

    Returns
    -------
    rows_for_csv, stats_dict, pair_id_next
    """
    rows: List[Dict[str, Any]] = []
    n_pairs_total = 0
    n_pairs_triggered = 0
    n_pairs_emitted = 0

    pair_id_counter = int(pair_id_start)

    for i, t in enumerate(turns):
        if t.turn_role != "analyst":
            continue

        # find the next executive turn
        j = i + 1
        while j < len(turns) and turns[j].turn_role != "exec":
            j += 1
        if j >= len(turns):
            break

        n_pairs_total += 1
        pair_is_triggered = int(t.analyst_triggered == 1)

        if (not keep_all_pairs) and (pair_is_triggered == 0):
            continue

        pair_id = f"PAIR_{pair_id_counter:06d}"
        pair_id_counter += 1
        n_pairs_emitted += 1
        if pair_is_triggered == 1:
            n_pairs_triggered += 1

        for turn in (t, turns[j]):
            rows.append(
                {
                    "uid": turn.uid,
                    "ticker": turn.ticker,
                    "event_start_et": turn.event_start_et,
                    "pair_id": pair_id,
                    "pair_is_stress_triggered": pair_is_triggered,
                    # backward compat: keep original naming
                    "is_stress_triggered": pair_is_triggered,
                    "turn_role": turn.turn_role,
                    "speaker": " / ".join(turn.speakers) if turn.speakers else "",
                    # audio slicing spans:
                    "original_index": turn.start_index,       # START index (compat)
                    "turn_end_index": turn.end_index,         # END index (new)
                    "timestamp": turn.start_timestamp,
                    "timestamp_end": turn.end_timestamp,
                    "text": turn.text,
                    "text_emotion": turn.text_emotion,
                    "emotion": turn.emotion,
                    "trigger_lexicon": turn.trigger_lexicon,
                    "trigger_emotion": turn.trigger_emotion,
                    "transcript_file": turn.transcript_file,
                    "turn_index": turn.turn_index,
                }
            )

    stats = {
        "turns_total": int(len(turns)),
        "pairs_total": int(n_pairs_total),
        "pairs_emitted": int(n_pairs_emitted),
        "pairs_triggered": int(n_pairs_triggered),
        "pair_id_start": int(pair_id_start),
        "pair_id_next": int(pair_id_counter),
    }
    return rows, stats, int(pair_id_counter)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 01: build stress-triggered Analyst→Executive adjacency pairs for EmoAlignBench.")
    p.add_argument("--data_dir", type=str, default="data", help="Input data directory.")
    p.add_argument("--results_dir", type=str, default="results", help="Output directory.")
    p.add_argument("--min_audio_size_kb", type=int, default=500, help="Ignore audio smaller than this (KB).")
    p.add_argument("--merge_by_speaker", action="store_true", help="Merge consecutive utterances by BOTH role and speaker (stricter). Default merges by role only.")
    p.add_argument("--keep_all_pairs", action="store_true", help="Emit all Analyst→Exec pairs (not only triggered).")
    p.add_argument("--numeric_valence_threshold", type=float, default=0.0, help="Threshold for numeric text_emotion valence to be considered negative.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    base_dir = Path(".")
    input_dir = base_dir / args.data_dir
    results_dir = base_dir / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / "stress_pairs_with_transcripts.csv"
    out_meta = results_dir / "stress_pairs_with_metastats.json"

    print(f"🚀 Step 01 started\nCWD: {os.getcwd()}\nData: {input_dir}\nOut: {results_dir}")

    all_indices = parse_all_indices(input_dir)
    if not all_indices:
        print("❌ No *_Index.json files found. Exiting.")
        return

    dataset_stats = calculate_dataset_stats(
        all_indices,
        input_dir,
        min_audio_size_bytes=int(args.min_audio_size_kb * 1024),
    )

    raw_df = load_native_nasdaq_data(all_indices, input_dir)
    if raw_df.empty:
        print("❌ No transcript records loaded. Exiting.")
        return

    # Per-call turn building and pair extraction
    emitted_rows: List[Dict[str, Any]] = []
    global_pair_id_counter: int = 0  # ensures unique PAIR_XXXXXX across all uids
    uid_stats: Dict[str, Any] = {}
    uids_with_pairs = 0

    for uid, g in raw_df.groupby("uid", sort=False):
        turns = build_turns_for_uid(
            g,
            lexicon=FINANCE_STRESS_LEXICON,
            merge_by_role_only=not bool(args.merge_by_speaker),
            numeric_valence_threshold=float(args.numeric_valence_threshold),
        )
        rows, stats, global_pair_id_counter = extract_pairs(
            turns,
            keep_all_pairs=bool(args.keep_all_pairs),
            pair_id_start=global_pair_id_counter,
        )
        uid_stats[str(uid)] = stats
        if stats["pairs_emitted"] > 0:
            uids_with_pairs += 1
            emitted_rows.extend(rows)

    out_df = pd.DataFrame(emitted_rows)
    if not out_df.empty:
        # Ensure deterministic ordering
        out_df.sort_values(["uid", "pair_id", "turn_role"], inplace=True, kind="stable")
        out_df.reset_index(drop=True, inplace=True)

    meta = {
        "dataset_integrity_statistics": dataset_stats,
        "turn_pair_statistics": {
            "uids_total": int(raw_df["uid"].nunique()),
            "uids_with_emitted_pairs": int(uids_with_pairs),
            "pairs_emitted_total": int(out_df["pair_id"].nunique()) if not out_df.empty else 0,
            "rows_emitted_total": int(len(out_df)),
        },
        "per_uid_stats": uid_stats,
        "config": {
            "merge_by_role_only": not bool(args.merge_by_speaker),
            "keep_all_pairs": bool(args.keep_all_pairs),
            "numeric_valence_threshold": float(args.numeric_valence_threshold),
            "min_audio_size_kb": int(args.min_audio_size_kb),
        },
        "pipeline_version": "emobench-v3-icdm2026",
    }

    out_df.to_csv(out_csv, index=False)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Step 01 complete.")
    print(f"   -> {out_csv}")
    print(f"   -> {out_meta}")
    if not out_df.empty:
        print(f"   Pairs emitted: {out_df['pair_id'].nunique()} | Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
