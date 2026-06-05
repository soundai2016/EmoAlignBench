from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

ROLE_MAP = {"analyst": "Analyst", "exec": "Executive", "Analyst": "Analyst", "Executive": "Executive"}
ROLE_ORDER = ["Analyst", "Executive"]
EXCLUDE_SUFFIXES = {".zip", ".pyc"}
EXCLUDE_NAMES = {"checksums_manifest.json"}
OBSOLETE_RESULT_FILES = [
    "FigS1_Validation_Gates.png",
    "FigS1_Validation_Gates.pdf",
    "FigS2_Call_Grouped_Splits.png",
    "FigS2_Call_Grouped_Splits.pdf",
    "paper_metrics.tex",
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def remove_obsolete_outputs(results_dir: Path) -> None:
    for name in OBSOLETE_RESULT_FILES:
        path = results_dir / name
        if path.exists():
            path.unlink()


def normalize_roles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turn_role"] = df["turn_role"].map(ROLE_MAP).fillna(df["turn_role"])
    return df


def add_ratios(df: pd.DataFrame, clip: float) -> pd.DataFrame:
    df = df.copy()
    for feat in ["f0_mean", "f0_std", "hnr_proxy", "voice_arousal"]:
        if feat not in df.columns:
            continue
        base = df[df["turn_role"].eq("Analyst")].set_index("pair_id")[feat].replace(0, np.nan)
        values = []
        for _, row in df.iterrows():
            values.append(row[feat] / base.get(row["pair_id"], np.nan) if row["turn_role"] == "Executive" else 1.0)
        df[f"{feat}_ratio"] = pd.Series(values, index=df.index, dtype="float64").clip(upper=clip)
    return df


def complete_pair_ids(df: pd.DataFrame, required_signal_cols: List[str]) -> Set[str]:
    keep: Set[str] = set()
    for pid, g in df.groupby("pair_id", sort=False):
        if set(g["turn_role"]) != set(ROLE_ORDER):
            continue
        if any((g["turn_role"] == role).sum() != 1 for role in ROLE_ORDER):
            continue
        if g[required_signal_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=None):
            continue
        keep.add(str(pid))
    return keep


def structural_role_report(df: pd.DataFrame) -> Dict[str, Any]:
    role_counts = df.groupby("pair_id")["turn_role"].value_counts().unstack(fill_value=0)
    exact_one_each = int(((role_counts.get("Analyst", 0) == 1) & (role_counts.get("Executive", 0) == 1)).sum())
    duplicate_or_missing = int(len(role_counts) - exact_one_each)
    time_ok = True
    if {"slice_start_sec", "slice_end_sec"}.issubset(df.columns):
        time_ok = bool((df["slice_end_sec"] > df["slice_start_sec"]).fillna(False).all())
    return {
        "pairs_with_exactly_one_analyst_and_one_executive": exact_one_each,
        "pairs_with_duplicate_or_missing_roles": duplicate_or_missing,
        "slice_start_before_end_for_all_rows": time_ok,
        "observed_roles": sorted([str(x) for x in df["turn_role"].dropna().unique()]),
    }


def signal_report(df: pd.DataFrame, complete: Set[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    required = cfg["validation"]["required_signal_columns"]
    missing = [c for c in required if c not in df.columns]
    finite_counts = {}
    for col in required:
        if col in df.columns:
            finite_counts[col] = int(np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum())
    trimmed = None
    if "trimmed_sec" in df.columns:
        trimmed = {
            "rows_ge_min_trimmed_sec": int((df["trimmed_sec"] >= float(cfg["validation"]["min_trimmed_sec"])).fillna(False).sum()),
            "min_trimmed_sec": float(pd.to_numeric(df["trimmed_sec"], errors="coerce").min()),
            "median_trimmed_sec": float(pd.to_numeric(df["trimmed_sec"], errors="coerce").median()),
        }
    return {
        "missing_required_signal_columns": missing,
        "finite_counts_by_signal_column": finite_counts,
        "complete_signal_pairs": int(len(complete)),
        "dropped_pairs_after_signal_qc": int(df["pair_id"].nunique() - len(complete)),
        "trimmed_duration_summary": trimmed,
    }


def pairwise_disjoint(sets: Iterable[Set[str]]) -> bool:
    seen: Set[str] = set()
    for s in sets:
        if seen & s:
            return False
        seen |= s
    return True


def make_splits(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg["statistics"].get("seed", 7))
    fractions = cfg["statistics"].get("split_fractions", {"train": 0.70, "validation": 0.15, "test": 0.15})
    pair_df = df.drop_duplicates(["uid", "pair_id"])[["uid", "pair_id"]].astype(str)
    uid_counts = pair_df.groupby("uid")["pair_id"].nunique().reset_index(name="pairs")
    uids = uid_counts["uid"].to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)
    n = len(uids)
    n_train = int(round(float(fractions["train"]) * n))
    n_val = int(round(float(fractions["validation"]) * n))
    split_uid_sets = {
        "train": set(uids[:n_train]),
        "validation": set(uids[n_train : n_train + n_val]),
        "test": set(uids[n_train + n_val :]),
    }
    out: Dict[str, Any] = {
        "seed": seed,
        "split_key": cfg["statistics"].get("split_key", "uid"),
        "source_of_truth": "call_grouped_splits.json",
        "splits": {},
    }
    all_pairs: Set[str] = set()
    uid_sets: List[Set[str]] = []
    pair_sets: List[Set[str]] = []
    for split in ["train", "validation", "test"]:
        uid_set = split_uid_sets[split]
        sub = pair_df[pair_df["uid"].isin(uid_set)]
        pairs = sorted(sub["pair_id"].unique().tolist())
        uid_list = sorted([str(u) for u in uid_set])
        all_pairs.update(pairs)
        uid_sets.append(set(uid_list))
        pair_sets.append(set(pairs))
        out["splits"][split] = {
            "n_uids": int(len(uid_list)),
            "n_pairs": int(len(pairs)),
            "uids": uid_list,
            "pair_ids": pairs,
        }
    out["no_uid_overlap"] = pairwise_disjoint(uid_sets)
    out["no_pair_overlap"] = pairwise_disjoint(pair_sets)
    out["n_unique_uids_covered"] = int(len(set().union(*uid_sets))) if uid_sets else 0
    out["n_unique_pairs_covered"] = int(len(all_pairs))
    out["split_summary"] = [
        {
            "split": split,
            "n_uids": out["splits"][split]["n_uids"],
            "n_pairs": out["splits"][split]["n_pairs"],
        }
        for split in ["train", "validation", "test"]
    ]
    return out


def metric_from_stats(stats: Dict[str, Any], name: str, legacy: str | None = None) -> Any:
    metrics = stats.get("metrics", {})
    if name in metrics:
        return metrics[name]
    if legacy and legacy in metrics:
        return metrics[legacy]
    if name == "num_pairs" and "robustness_summary" in stats:
        return stats["robustness_summary"].get("n_pairs")
    return None


def value_status(value: Any, expected: Any | None = None, pass_if: bool | None = None) -> str:
    if pass_if is not None:
        return "PASS" if pass_if else "FAIL"
    if expected is None:
        return "INFO"
    return "PASS" if value == expected else "FAIL"


def validation_table_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    ds = report["dataset"]
    obs = report["observed_cached_artifact"]
    structural = report["validation_gates"]["structural_and_role"]
    signal = report["validation_gates"]["signal"]
    statistical = report["validation_gates"]["statistical"]
    rows = [
        ("dataset", "raw_utterance_rows", ds["raw_utterance_rows"], ds["raw_utterance_rows"], True),
        ("dataset", "source_audio_hours", ds["source_audio_hours"], ds["source_audio_hours"], True),
        ("cached_artifact", "turn_rows_before_signal_pair_qc", obs["turn_rows_before_signal_pair_qc"], ds["raw_cached_turn_rows"], obs["turn_rows_before_signal_pair_qc"] == ds["raw_cached_turn_rows"]),
        ("cached_artifact", "pairs_before_signal_pair_qc", obs["pairs_before_signal_pair_qc"], ds["raw_pairs_before_signal_qc"], obs["pairs_before_signal_pair_qc"] == ds["raw_pairs_before_signal_qc"]),
        ("structural_role", "pairs_with_exactly_one_analyst_and_one_executive", structural["pairs_with_exactly_one_analyst_and_one_executive"], ds["raw_pairs_before_signal_qc"], structural["pairs_with_exactly_one_analyst_and_one_executive"] == ds["raw_pairs_before_signal_qc"]),
        ("structural_role", "pairs_with_duplicate_or_missing_roles", structural["pairs_with_duplicate_or_missing_roles"], 0, structural["pairs_with_duplicate_or_missing_roles"] == 0),
        ("structural_role", "slice_start_before_end_for_all_rows", structural["slice_start_before_end_for_all_rows"], True, bool(structural["slice_start_before_end_for_all_rows"])),
        ("signal", "missing_required_signal_columns", len(signal["missing_required_signal_columns"]), 0, len(signal["missing_required_signal_columns"]) == 0),
        ("signal", "complete_signal_pairs", signal["complete_signal_pairs"], ds["curated_valid_pairs"], signal["complete_signal_pairs"] == ds["curated_valid_pairs"]),
        ("signal", "complete_signal_turn_rows", obs["complete_signal_turn_rows"], ds["curated_valid_turn_rows"], obs["complete_signal_turn_rows"] == ds["curated_valid_turn_rows"]),
        ("signal", "dropped_pairs_after_signal_qc", signal["dropped_pairs_after_signal_qc"], obs["pairs_before_signal_pair_qc"] - ds["curated_valid_pairs"], signal["dropped_pairs_after_signal_qc"] == obs["pairs_before_signal_pair_qc"] - ds["curated_valid_pairs"]),
        ("statistical", "call_groups_available", statistical["call_groups_available"], obs["uids_in_complete_signal_pairs"], statistical["call_groups_available"] == obs["uids_in_complete_signal_pairs"]),
        ("overall", "overall_pass", report["validation_gates"]["overall_pass"], True, bool(report["validation_gates"]["overall_pass"])),
    ]
    return [
        {
            "table": "S1_validation_gates",
            "section": section,
            "metric": metric,
            "value": value,
            "expected": expected,
            "status": value_status(value, expected, passed),
            "source_json": "validation_report.json",
        }
        for section, metric, value, expected, passed in rows
    ]


def split_table_rows(splits: Dict[str, Any]) -> List[Dict[str, Any]]:
    split_names = [s for s in ["train", "validation", "test"] if s in splits.get("splits", {})]
    total_uids = sum(int(splits["splits"][name]["n_uids"]) for name in split_names)
    total_pairs = sum(int(splits["splits"][name]["n_pairs"]) for name in split_names)
    rows: List[Dict[str, Any]] = []
    for name in split_names:
        item = splits["splits"][name]
        rows.append(
            {
                "table": "S2_call_grouped_splits",
                "split": name,
                "n_uids": int(item["n_uids"]),
                "n_pairs": int(item["n_pairs"]),
                "uid_fraction": round(int(item["n_uids"]) / total_uids, 6) if total_uids else None,
                "pair_fraction": round(int(item["n_pairs"]) / total_pairs, 6) if total_pairs else None,
                "seed": splits.get("seed"),
                "split_key": splits.get("split_key"),
                "source_json": "call_grouped_splits.json",
            }
        )
    rows.append(
        {
            "table": "S2_call_grouped_splits",
            "split": "total",
            "n_uids": total_uids,
            "n_pairs": total_pairs,
            "uid_fraction": 1.0,
            "pair_fraction": 1.0,
            "seed": splits.get("seed"),
            "split_key": splits.get("split_key"),
            "source_json": "call_grouped_splits.json",
        }
    )
    return rows


def write_table_bundle(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path, source_json: str) -> None:
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "source_json": source_json,
        "storage_policy": "json_or_csv_only",
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_supplementary_tables(report: Dict[str, Any], splits: Dict[str, Any], results_dir: Path) -> None:
    write_table_bundle(
        validation_table_rows(report),
        results_dir / "table_s1_validation_gates.csv",
        results_dir / "table_s1_validation_gates.json",
        "validation_report.json",
    )
    write_table_bundle(
        split_table_rows(splits),
        results_dir / "table_s2_call_grouped_splits.csv",
        results_dir / "table_s2_call_grouped_splits.json",
        "call_grouped_splits.json",
    )


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    return pd.read_csv(path).to_dict(orient="records")


def expected_split_counts(splits: Dict[str, Any]) -> Dict[str, int]:
    return {split: int(item["n_pairs"]) for split, item in splits.get("splits", {}).items()}


def table_split_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {str(r["split"]): int(r["n_pairs"]) for r in rows if str(r.get("split")) != "total"}


def write_consistency_check(report: Dict[str, Any], splits: Dict[str, Any], results_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    s1_csv = results_dir / "table_s1_validation_gates.csv"
    s2_csv = results_dir / "table_s2_call_grouped_splits.csv"
    s1_rows = read_csv_records(s1_csv) if s1_csv.exists() else []
    s2_rows = read_csv_records(s2_csv) if s2_csv.exists() else []
    split_counts_json = expected_split_counts(splits)
    split_counts_table = table_split_counts(s2_rows)
    total_pairs_from_splits = sum(split_counts_json.values())
    total_uids_from_splits = sum(int(item["n_uids"]) for item in splits.get("splits", {}).values())
    valid_pairs = int(report["observed_cached_artifact"]["complete_signal_pairs"])
    valid_uids = int(report["observed_cached_artifact"]["uids_in_complete_signal_pairs"])
    stale = [name for name in OBSOLETE_RESULT_FILES if (results_dir / name).exists()]
    paper_metrics_ok = (results_dir / "paper_metrics.json").exists() and (results_dir / "paper_metrics.csv").exists()
    expected_pair_counts = cfg.get("statistics", {}).get("expected_split_pair_counts")
    expected_uid_counts = cfg.get("statistics", {}).get("expected_split_uid_counts")
    uid_counts_json = {split: int(item["n_uids"]) for split, item in splits.get("splits", {}).items()}
    check = {
        "source_of_truth_policy": {
            "s1_table_source": "validation_report.json",
            "s2_table_source": "call_grouped_splits.json",
            "supplementary_figures_generated": False,
            "table_storage": "json_or_csv_only",
        },
        "s1_rows_present": len(s1_rows) > 0,
        "s2_counts_from_call_grouped_splits_json": split_counts_json,
        "s2_uid_counts_from_call_grouped_splits_json": uid_counts_json,
        "s2_counts_from_table": split_counts_table,
        "s2_table_matches_call_grouped_splits_json": split_counts_table == split_counts_json,
        "split_pair_counts_match_config_expectation": (split_counts_json == expected_pair_counts) if expected_pair_counts else True,
        "split_uid_counts_match_config_expectation": (uid_counts_json == expected_uid_counts) if expected_uid_counts else True,
        "split_pair_total_matches_valid_pairs": total_pairs_from_splits == valid_pairs,
        "split_uid_total_matches_valid_call_groups": total_uids_from_splits == valid_uids,
        "no_uid_overlap": bool(splits.get("no_uid_overlap")),
        "no_pair_overlap": bool(splits.get("no_pair_overlap")),
        "no_stale_supplementary_figures": len([x for x in stale if x.startswith("FigS")]) == 0,
        "no_latex_macro_result_files": not (results_dir / "paper_metrics.tex").exists(),
        "paper_metrics_json_csv_present": paper_metrics_ok,
        "obsolete_result_files_present": stale,
    }
    check["all_ok"] = all(
        [
            check["s1_rows_present"],
            check["s2_table_matches_call_grouped_splits_json"],
            check["split_pair_counts_match_config_expectation"],
            check["split_uid_counts_match_config_expectation"],
            check["split_pair_total_matches_valid_pairs"],
            check["split_uid_total_matches_valid_call_groups"],
            check["no_uid_overlap"],
            check["no_pair_overlap"],
            check["no_stale_supplementary_figures"],
            check["no_latex_macro_result_files"],
            check["paper_metrics_json_csv_present"],
        ]
    )
    with open(results_dir / "artifact_consistency_check.json", "w", encoding="utf-8") as f:
        json.dump(check, f, ensure_ascii=False, indent=2)
    return check


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path, results_dir: Path) -> Dict[str, Any]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_SUFFIXES or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    manifest = {"algorithm": "SHA256", "file_count": len(files), "files": files}
    with open(results_dir / "checksums_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the EmoAlignBench code/data artifact and write release metadata.")
    parser.add_argument("--config", default="configs/emobench_config.json")
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_json(resolve(root, args.config))
    results_dir = resolve(root, args.results or cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs(results_dir)

    csv_path = resolve(root, cfg["paths"]["cached_acoustics_csv"])
    df = add_ratios(normalize_roles(pd.read_csv(csv_path)), float(cfg["acoustics"].get("ratio_clip", 5.0)))
    complete = complete_pair_ids(df, cfg["validation"]["required_signal_columns"])
    complete_df = df[df["pair_id"].astype(str).isin(complete)].copy()

    metastats_path = resolve(root, cfg["paths"]["metastats_json"])
    metastats = load_json(metastats_path) if metastats_path.exists() else {}
    stats_path = results_dir / "statistics_tables.json"
    paper_stats = load_json(stats_path) if stats_path.exists() else {}
    paper_reported_pairs = metric_from_stats(paper_stats, "num_pairs", "\\NumPairs")

    report = {
        "dataset": cfg["dataset"],
        "observed_cached_artifact": {
            "turn_rows_before_signal_pair_qc": int(len(df)),
            "pairs_before_signal_pair_qc": int(df["pair_id"].nunique()),
            "complete_signal_turn_rows": int(len(complete_df)),
            "complete_signal_pairs": int(len(complete)),
            "uids_in_complete_signal_pairs": int(complete_df["uid"].nunique()),
        },
        "validation_gates": {
            "structural_and_role": structural_role_report(df),
            "signal": signal_report(df, complete, cfg),
            "statistical": {
                "cluster_key": cfg["statistics"].get("split_key", "uid"),
                "bootstrap_resamples": int(cfg["statistics"].get("bootstrap_resamples", 2000)),
                "permutation_resamples": int(cfg["statistics"].get("permutation_resamples", 4000)),
                "paper_reported_pairs": paper_reported_pairs,
                "call_groups_available": int(complete_df["uid"].nunique()),
            },
        },
        "source_metastats": metastats.get("turn_pair_statistics", {}),
        "best_paper_artifact_checks": {
            "fig0_framework_present": (results_dir / "Fig0_Framework.png").exists(),
            "main_figures_present": all((results_dir / f"Fig{i}_{name}.png").exists() for i, name in [(1, "Landscape"), (2, "Raincloud"), (3, "Dynamics")]),
            "main_paper_tables_present": all((results_dir / name).exists() for name in ["table_baseline_stats.csv", "table_ratio_stats.csv", "table_robustness.csv"]),
            "supplementary_tables_present": False,
            "paper_metrics_json_csv_present": (results_dir / "paper_metrics.json").exists() and (results_dir / "paper_metrics.csv").exists(),
            "latex_macro_outputs_present": (results_dir / "paper_metrics.tex").exists(),
        },
        "release_policy": {
            "supplementary_diagnostics_format": "tables_only_csv_and_json",
            "validation_table_source": "validation_report.json",
            "split_table_source": "call_grouped_splits.json",
            "stale_supplementary_figures_removed": True,
        },
    }
    expected = int(cfg["dataset"]["curated_valid_pairs"])
    report["validation_gates"]["overall_pass"] = int(len(complete)) == expected

    splits = make_splits(complete_df, cfg)
    with open(results_dir / "call_grouped_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    export_supplementary_tables(report, splits, results_dir)
    report["best_paper_artifact_checks"]["supplementary_tables_present"] = all(
        (results_dir / name).exists()
        for name in ["table_s1_validation_gates.csv", "table_s1_validation_gates.json", "table_s2_call_grouped_splits.csv", "table_s2_call_grouped_splits.json"]
    )
    with open(results_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    export_supplementary_tables(report, splits, results_dir)
    consistency = write_consistency_check(report, splits, results_dir, cfg)
    report["artifact_consistency_check"] = {
        "all_ok": consistency["all_ok"],
        "s2_counts_from_call_grouped_splits_json": consistency["s2_counts_from_call_grouped_splits_json"],
        "s2_table_matches_call_grouped_splits_json": consistency["s2_table_matches_call_grouped_splits_json"],
        "no_stale_supplementary_figures": consistency["no_stale_supplementary_figures"],
        "no_latex_macro_result_files": consistency["no_latex_macro_result_files"],
    }
    with open(results_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    export_supplementary_tables(report, splits, results_dir)
    consistency = write_consistency_check(report, splits, results_dir, cfg)
    if not consistency["all_ok"]:
        raise SystemExit("Artifact consistency check failed; inspect results/artifact_consistency_check.json.")

    manifest = write_manifest(root, results_dir)
    split_counts = consistency["s2_counts_from_call_grouped_splits_json"]
    print(
        "Validated "
        f"{len(complete)} complete pairs; wrote validation_report.json, call_grouped_splits.json, "
        f"S1/S2 CSV+JSON tables, and {manifest['file_count']} checksums. "
        f"Split pairs: train={split_counts.get('train')}, validation={split_counts.get('validation')}, test={split_counts.get('test')}."
    )


if __name__ == "__main__":
    main()
