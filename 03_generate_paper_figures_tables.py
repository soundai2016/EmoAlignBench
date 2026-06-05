from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROLE_ORDER = ["Analyst", "Executive"]
ROLE_MAP = {"analyst": "Analyst", "exec": "Executive", "Analyst": "Analyst", "Executive": "Executive"}
ROLE_PALETTE = {"Analyst": "#4DBBD5", "Executive": "#E64B35"}
ACCENT = "#3C5488"
SEED = 7

FEATURE_LABELS = {
    "text_valence": "Text score",
    "voice_valence": "Voice score",
    "f0_std": r"$F_0$ std. (Hz)",
    "emotion_incongruence": "Incongruence",
    "f0_std_ratio": "Pitch variability ratio",
    "hnr_proxy_ratio": "Harmonic-residual proxy ratio",
    "voice_arousal_ratio": "Voice arousal ratio",
}

TABLE_LABELS = {
    "text_valence": "Text score",
    "voice_valence": "Voice score",
    "f0_std": "F0 std. (Hz)",
    "emotion_incongruence": "Incongruence",
    "f0_std_ratio": "Pitch variability ratio",
    "hnr_proxy_ratio": "Harmonic-residual proxy ratio",
    "voice_arousal_ratio": "Voice arousal ratio",
}

OBSOLETE_RESULT_FILES = [
    "FigS1_Validation_Gates.png",
    "FigS1_Validation_Gates.pdf",
    "FigS2_Call_Grouped_Splits.png",
    "FigS2_Call_Grouped_Splits.pdf",
    "paper_metrics.tex",
]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def bh_fdr(pvals: Iterable[float]) -> List[float]:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if not valid.any():
        return q.tolist()
    pv = p[valid]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[valid] = out
    return q.tolist()


def sig_label(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def q_text(q: float | None) -> str:
    if q is None or pd.isna(q):
        return "--"
    if q < 0.001:
        return "<0.001"
    return f"{q:.3f}"


def direction(mean_an: float, mean_ex: float) -> str:
    if pd.isna(mean_an) or pd.isna(mean_ex):
        return "missing"
    if math.isclose(mean_an, mean_ex, rel_tol=0.0, abs_tol=1e-12):
        return "approx"
    return "up" if mean_ex > mean_an else "down"


def clean_direction(value: Any) -> str:
    mapping = {
        r"$\uparrow$": "up",
        "\\uparrow": "up",
        "↑": "up",
        r"$\downarrow$": "down",
        "\\downarrow": "down",
        "↓": "down",
        r"$\approx$": "approx",
        "\\approx": "approx",
        "≈": "approx",
        "--": "missing",
        None: "missing",
    }
    return mapping.get(value, str(value))


def remove_obsolete_outputs(out_dir: Path) -> None:
    for name in OBSOLETE_RESULT_FILES:
        path = out_dir / name
        if path.exists():
            path.unlink()


def clean_metric_name(name: str) -> str:
    return {
        "\\NumPairs": "num_pairs",
        "\\ContagionR": "contagion_r",
        "\\ContagionP": "contagion_p",
    }.get(name, name.strip("\\").lower())


def normalize_stats_schema(stats_json: Dict[str, Any]) -> Dict[str, Any]:
    stats_json = dict(stats_json)
    raw_metrics = stats_json.get("metrics", {})
    metrics: Dict[str, Any] = {}
    p_values: Dict[str, Any] = {}
    q_values: Dict[str, Any] = {}
    significance: Dict[str, Any] = {}

    for key, value in raw_metrics.items():
        if key.startswith("\\Pval_"):
            p_values[key.replace("\\Pval_", "")] = value
        elif key.startswith("\\Qval_"):
            q_values[key.replace("\\Qval_", "")] = value
        elif key.startswith("\\Sig_"):
            significance[key.replace("\\Sig_", "")] = value
        else:
            metrics[clean_metric_name(key)] = value

    c = stats_json.get("robustness_summary", {}).get("contagion", {})
    robust = stats_json.get("robustness_summary", {})
    if "n_pairs" in robust:
        metrics["num_pairs"] = robust.get("n_pairs")
    if "r" in c:
        metrics["contagion_r"] = round(float(c["r"]), 3)
    if "p" in c:
        metrics["contagion_p"] = float(c["p"])
    if c.get("ci_fisher"):
        metrics["contagion_fisher_ci_low"] = float(c["ci_fisher"][0])
        metrics["contagion_fisher_ci_high"] = float(c["ci_fisher"][1])
    if c.get("ci_cluster_bootstrap"):
        metrics["contagion_cluster_ci_low"] = float(c["ci_cluster_bootstrap"][0])
        metrics["contagion_cluster_ci_high"] = float(c["ci_cluster_bootstrap"][1])
    if c.get("p_within_call_permutation") is not None:
        metrics["within_call_permutation_p"] = float(c["p_within_call_permutation"])

    role_stats = stats_json.get("table_role_stats", {})
    ex = role_stats.get("Executive", {})
    for feat, metric_name in [
        ("f0_std_ratio", "pitch_variability_ratio_mean"),
        ("hnr_proxy_ratio", "harmonic_residual_ratio_mean"),
        ("voice_arousal_ratio", "voice_arousal_ratio_mean"),
    ]:
        if feat in ex and ex[feat].get("mean") is not None:
            metrics[metric_name] = float(ex[feat]["mean"])

    for feat, values in stats_json.get("robustness_summary", {}).get("role_tests", {}).items():
        if values.get("p_raw") is not None:
            p_values[feat] = values.get("p_raw")
        if values.get("q_fdr") is not None:
            q_values[feat] = values.get("q_fdr")
        if values.get("sig_fdr") is not None:
            significance[feat] = values.get("sig_fdr")
        values["direction"] = clean_direction(values.pop("dir", values.get("direction", "missing")))

    stats_json["metrics"] = {
        **metrics,
        "p_values": p_values,
        "q_values": q_values,
        "significance": significance,
    }
    stats_json["schema"] = {
        "version": "2.0",
        "table_storage": "json_or_csv_only",
        "latex_macro_outputs": False,
    }
    return stats_json


def normalize_roles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turn_role"] = df["turn_role"].map(ROLE_MAP).fillna(df["turn_role"])
    return df


def add_interactional_ratios(df: pd.DataFrame, clip: float) -> pd.DataFrame:
    df = df.copy()
    for feat in ["f0_mean", "f0_std", "hnr_proxy", "voice_arousal"]:
        if feat not in df.columns:
            continue
        baseline = df[df["turn_role"].eq("Analyst")].set_index("pair_id")[feat].replace(0, np.nan)
        ratio = []
        for _, row in df.iterrows():
            if row["turn_role"] == "Executive":
                ratio.append(row.get(feat, np.nan) / baseline.get(row["pair_id"], np.nan))
            else:
                ratio.append(1.0)
        df[f"{feat}_ratio"] = pd.Series(ratio, index=df.index, dtype="float64").clip(upper=clip)
    return df


def valid_pair_ids(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    keep = []
    for pair_id, g in df.groupby("pair_id", sort=False):
        if set(g["turn_role"]) != set(ROLE_ORDER):
            continue
        if any((g["turn_role"] == role).sum() != 1 for role in ROLE_ORDER):
            continue
        if g[required_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=None):
            continue
        keep.append(pair_id)
    return keep


def prepare_analysis_df(csv_path: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_roles(df)
    df = add_interactional_ratios(df, clip=float(cfg["acoustics"].get("ratio_clip", 5.0)))
    required = cfg["validation"]["required_signal_columns"]
    keep = valid_pair_ids(df, required)
    out = df[df["pair_id"].isin(keep)].copy()
    out.sort_values(["uid", "pair_id", "turn_role"], inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out


def paired_vectors(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    an = df[df["turn_role"].eq("Analyst")][["uid", "pair_id", feature]].rename(columns={feature: "Analyst"})
    ex = df[df["turn_role"].eq("Executive")][["uid", "pair_id", feature]].rename(columns={feature: "Executive"})
    return pd.merge(an, ex, on=["uid", "pair_id"], how="inner").dropna()


def paired_or_ratio_test(df: pd.DataFrame, feature: str) -> float:
    if feature.endswith("_ratio"):
        x = df[df["turn_role"].eq("Executive")][feature].dropna() - 1.0
        return float(stats.wilcoxon(x).pvalue) if len(x) else np.nan
    pair = paired_vectors(df, feature)
    return float(stats.wilcoxon(pair["Analyst"], pair["Executive"]).pvalue) if len(pair) else np.nan


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 3 or pd.isna(r):
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def pearson_from_sums(n: float, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    num = n * sxy - sx * sy
    den = math.sqrt(max(n * sxx - sx * sx, 0.0) * max(n * syy - sy * sy, 0.0))
    return float(num / den) if den > 0 else np.nan


def cluster_bootstrap_ci(pair_df: pd.DataFrame, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = []
    for _, sub in pair_df.groupby("uid", sort=False):
        x = sub["Analyst"].to_numpy(dtype=float)
        y = sub["Executive"].to_numpy(dtype=float)
        groups.append((len(x), x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum(), np.multiply(x, y).sum()))
    if len(groups) < 2:
        return np.nan, np.nan
    arr = np.asarray(groups, dtype=float)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        sums = arr[idx].sum(axis=0)
        r = pearson_from_sums(*sums)
        if pd.notna(r):
            rs.append(r)
    if len(rs) < 50:
        return np.nan, np.nan
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return float(lo), float(hi)


def within_call_permutation_p(pair_df: pd.DataFrame, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = stats.pearsonr(pair_df["Analyst"], pair_df["Executive"]).statistic
    groups = []
    for _, sub in pair_df.groupby("uid", sort=False):
        if len(sub) >= 2:
            groups.append((sub["Analyst"].to_numpy(dtype=float), sub["Executive"].to_numpy(dtype=float)))
    if not groups:
        return np.nan
    more_extreme = 0
    valid = 0
    for _ in range(n_perm):
        n = sx = sy = sxx = syy = sxy = 0.0
        for x, y0 in groups:
            y = rng.permutation(y0)
            n += len(x)
            sx += x.sum()
            sy += y.sum()
            sxx += np.square(x).sum()
            syy += np.square(y).sum()
            sxy += np.multiply(x, y).sum()
        r = pearson_from_sums(n, sx, sy, sxx, syy, sxy)
        if pd.isna(r):
            continue
        valid += 1
        if abs(r) >= abs(observed):
            more_extreme += 1
    return float((more_extreme + 1) / (valid + 1)) if valid else np.nan


def compute_statistics(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    baseline_features = cfg["paper_features"]["baseline_table"]
    ratio_features = cfg["paper_features"]["ratio_table"]
    all_test_features = ["voice_valence", "text_valence", "f0_std_ratio", "hnr_proxy_ratio", "voice_arousal_ratio", "f0_std", "emotion_incongruence"]

    p_raw = {feat: paired_or_ratio_test(df, feat) for feat in all_test_features}
    q_vals = dict(zip(all_test_features, bh_fdr([p_raw[f] for f in all_test_features])))

    role_stats: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for role in ROLE_ORDER:
        sub = df[df["turn_role"].eq(role)]
        role_stats[role] = {}
        for feat in baseline_features + ratio_features:
            if feat in sub.columns:
                role_stats[role][feat] = {
                    "mean": round(float(sub[feat].mean()), 4) if sub[feat].notna().any() else None,
                    "std": round(float(sub[feat].std()), 4) if sub[feat].notna().sum() > 1 else None,
                }

    pair = paired_vectors(df, "emotion_incongruence")
    r = float(stats.pearsonr(pair["Analyst"], pair["Executive"]).statistic)
    p = float(stats.pearsonr(pair["Analyst"], pair["Executive"]).pvalue)
    ci_f = fisher_ci(r, len(pair))
    ci_b = cluster_bootstrap_ci(pair, int(cfg["statistics"]["bootstrap_resamples"]), int(cfg["statistics"]["seed"]))
    perm = within_call_permutation_p(pair, int(cfg["statistics"]["permutation_resamples"]), int(cfg["statistics"]["seed"]))

    metrics = {
        "num_pairs": int(len(pair)),
        "contagion_r": round(r, 3),
        "contagion_p": float(p),
        "p_values": {feat: float(p_raw[feat]) if pd.notna(p_raw[feat]) else None for feat in all_test_features},
        "q_values": {feat: float(q_vals[feat]) if pd.notna(q_vals[feat]) else None for feat in all_test_features},
        "significance": {feat: sig_label(q_vals[feat]) for feat in all_test_features},
    }

    role_tests = {}
    for feat in all_test_features:
        mean_an = role_stats.get("Analyst", {}).get(feat, {}).get("mean", np.nan)
        mean_ex = role_stats.get("Executive", {}).get(feat, {}).get("mean", np.nan)
        role_tests[feat] = {
            "p_raw": float(p_raw[feat]) if pd.notna(p_raw[feat]) else None,
            "q_fdr": float(q_vals[feat]) if pd.notna(q_vals[feat]) else None,
            "sig_fdr": sig_label(q_vals[feat]),
            "dir": direction(mean_an, mean_ex),
        }

    return {
        "metrics": metrics,
        "table_role_stats": role_stats,
        "robustness_summary": {
            "n_pairs": int(len(pair)),
            "contagion": {
                "r": r,
                "p": p,
                "ci_fisher": [float(ci_f[0]), float(ci_f[1])],
                "ci_cluster_bootstrap": [float(ci_b[0]), float(ci_b[1])],
                "p_within_call_permutation": float(perm) if pd.notna(perm) else None,
            },
            "role_tests": role_tests,
        },
    }


def export_tables(df: pd.DataFrame, stats_json: Dict[str, Any], out_dir: Path, cfg: Dict[str, Any]) -> None:
    stats_json = normalize_stats_schema(stats_json)
    role_stats = stats_json["table_role_stats"]
    role_tests = stats_json["robustness_summary"]["role_tests"]

    baseline_rows = []
    for feat in cfg["paper_features"]["baseline_table"]:
        baseline_rows.append(
            {
                "feature": TABLE_LABELS[feat],
                "analyst_mean": role_stats["Analyst"][feat]["mean"],
                "analyst_std": role_stats["Analyst"][feat]["std"],
                "executive_mean": role_stats["Executive"][feat]["mean"],
                "executive_std": role_stats["Executive"][feat]["std"],
                "q_fdr": role_tests[feat]["q_fdr"],
                "q_display": q_text(role_tests[feat]["q_fdr"]),
            }
        )
    pd.DataFrame(baseline_rows).to_csv(out_dir / "table_baseline_stats.csv", index=False)

    ratio_rows = []
    for feat in cfg["paper_features"]["ratio_table"]:
        ex = role_stats["Executive"][feat]
        ratio_rows.append(
            {
                "interaction_ratio": TABLE_LABELS[feat],
                "base": 1.0,
                "mean": ex["mean"],
                "std": ex["std"],
                "q_fdr": role_tests[feat]["q_fdr"],
                "q_display": q_text(role_tests[feat]["q_fdr"]),
            }
        )
    pd.DataFrame(ratio_rows).to_csv(out_dir / "table_ratio_stats.csv", index=False)

    c = stats_json["robustness_summary"]["contagion"]
    pd.DataFrame(
        [
            {"check": "Pearson coupling", "result": f"r={c['r']:.3f}, p={c['p']:.3g}"},
            {"check": "Fisher-z interval", "result": f"95% CI [{c['ci_fisher'][0]:.3f}, {c['ci_fisher'][1]:.3f}]"},
            {"check": "Cluster bootstrap", "result": f"2000 call-clustered resamples, CI [{c['ci_cluster_bootstrap'][0]:.3f}, {c['ci_cluster_bootstrap'][1]:.3f}]"},
            {"check": "Within-call permutation", "result": f"p={c['p_within_call_permutation']:.4f}"},
        ]
    ).to_csv(out_dir / "table_robustness.csv", index=False)

    metrics = {
        "num_pairs": int(stats_json["metrics"]["num_pairs"]),
        "contagion_r": round(float(c["r"]), 3),
        "contagion_p": float(c["p"]),
        "contagion_fisher_ci": [round(float(c["ci_fisher"][0]), 3), round(float(c["ci_fisher"][1]), 3)],
        "contagion_cluster_ci": [round(float(c["ci_cluster_bootstrap"][0]), 3), round(float(c["ci_cluster_bootstrap"][1]), 3)],
        "within_call_permutation_p": round(float(c["p_within_call_permutation"]), 4),
        "pitch_variability_ratio_mean": round(float(role_stats["Executive"]["f0_std_ratio"]["mean"]), 3),
        "harmonic_residual_ratio_mean": round(float(role_stats["Executive"]["hnr_proxy_ratio"]["mean"]), 3),
        "voice_arousal_ratio_mean": round(float(role_stats["Executive"]["voice_arousal_ratio"]["mean"]), 3),
        "storage_policy": "json_or_csv_only",
    }
    with open(out_dir / "paper_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, list) else v} for k, v in metrics.items()]).to_csv(out_dir / "paper_metrics.csv", index=False)
    remove_obsolete_outputs(out_dir)


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def generate_fig0(out_dir: Path, cfg: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        ("Public earnings-call\naudio + transcripts", "142,751 timestamped\nutterance rows"),
        ("Q&A filtering", "remove scripted remarks\nlocalize high-stakes turns"),
        ("Role-aware pairing", "Analyst → Executive\nadjacency pairs"),
        ("Text + voice scoring", "$v_{text}$ and $v_{voice}$\ncommon [-1,1] scale"),
        ("Acoustic friction", "$\\Delta_{incong}=|v_{text}-v_{voice}|$\nratios + coupling"),
        ("Benchmark release", "812 validated pairs\nfigures, tables, checks"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.62
    w, h = 0.135, 0.22
    for i, (title, subtitle) in enumerate(steps):
        face = "#F7FBFF" if i % 2 == 0 else "#FFF8F5"
        edge = ROLE_PALETTE["Analyst"] if i % 2 == 0 else ROLE_PALETTE["Executive"]
        box = patches.FancyBboxPatch(
            (xs[i] - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(xs[i], y + 0.035, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(xs[i], y - 0.055, subtitle, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - w / 2 - 0.01, y), xytext=(xs[i] + w / 2 + 0.01, y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#555555"))

    tasks = [
        "Risk review",
        "Investor-relations coaching",
        "Audit/compliance triage",
        "Spoken-dialogue evaluation",
        "Computational social science",
    ]
    ax.text(0.5, 0.28, "Reusable benchmark tasks and application targets", ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT)
    task_xs = np.linspace(0.14, 0.86, len(tasks))
    for tx, task in zip(task_xs, tasks):
        pill = patches.FancyBboxPatch((tx - 0.075, 0.12), 0.15, 0.09, boxstyle="round,pad=0.01,rounding_size=0.045", linewidth=1.0, edgecolor="#C7C7C7", facecolor="#F4F4F4")
        ax.add_patch(pill)
        ax.text(tx, 0.165, task, ha="center", va="center", fontsize=8.5)

    ax.text(0.5, 0.93, cfg["dataset"]["name"] + ": Open multimodal mining of acoustic friction", ha="center", va="center", fontsize=15, fontweight="bold")
    save_figure(fig, out_dir, "Fig0_Framework")


def generate_fig1(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    plot_df = df.copy()
    plot_df["text_jitter"] = plot_df["text_valence"] + rng.normal(0, 0.025, len(plot_df))
    plot_df["voice_jitter"] = plot_df["voice_valence"] + rng.normal(0, 0.025, len(plot_df))

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    sns.kdeplot(data=plot_df, x="text_jitter", y="voice_jitter", hue="turn_role", fill=True, palette=ROLE_PALETTE, alpha=0.15, levels=8, thresh=0.05, legend=False, ax=ax)
    sns.scatterplot(data=plot_df, x="text_jitter", y="voice_jitter", hue="turn_role", size="emotion_incongruence", sizes=(18, 220), palette=ROLE_PALETTE, alpha=0.72, edgecolor="white", linewidth=0.35, legend=False, ax=ax)
    ax.axhline(0, color="#777777", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="#777777", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel(r"Text score $v_{text}$")
    ax.set_ylabel(r"Voice score $v_{voice}$")
    ax.set_title("Multimodal incongruence landscape", fontweight="bold")
    handles = [mlines.Line2D([], [], color="white", marker="o", markerfacecolor=ROLE_PALETTE[r], markeredgecolor="white", markersize=9) for r in ROLE_ORDER]
    ax.legend(handles, ROLE_ORDER, loc="upper right", frameon=True)
    save_figure(fig, out_dir, "Fig1_Landscape")


def add_sig_bracket(ax: plt.Axes, label: str, y: float, h: float) -> None:
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="#333333", lw=1.0)
    ax.text(0.5, y + h, label, ha="center", va="bottom", fontsize=11, fontweight="bold" if label != "ns" else "normal")


def generate_fig2(df: pd.DataFrame, stats_json: Dict[str, Any], out_dir: Path, cfg: Dict[str, Any]) -> None:
    features = cfg["paper_features"]["figure2_order"]
    q_lookup = {k: v["q_fdr"] for k, v in stats_json["robustness_summary"]["role_tests"].items()}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6))
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, feat, letter in zip(axes.flatten(), features, letters):
        if feat.endswith("_ratio"):
            plot_df = df[df["turn_role"].eq("Executive")].copy()
            sns.violinplot(data=plot_df, x="turn_role", y=feat, color=ROLE_PALETTE["Executive"], inner=None, cut=0, linewidth=0, ax=ax)
            sns.stripplot(data=plot_df, x="turn_role", y=feat, color="black", alpha=0.28, size=2.6, jitter=0.23, ax=ax)
            sns.boxplot(data=plot_df, x="turn_role", y=feat, color="white", width=0.12, showfliers=False, medianprops={"color": "black", "lw": 1.6}, ax=ax)
            ax.axhline(1.0, color="#666666", lw=1.2, ls="--", alpha=0.8)
            ax.set_xlabel("")
            ax.set_xticks([0])
            ax.set_xticklabels(["Executive / Analyst"])
        else:
            sns.violinplot(data=df, x="turn_role", y=feat, palette=ROLE_PALETTE, order=ROLE_ORDER, inner=None, cut=0, linewidth=0, ax=ax)
            sns.stripplot(data=df, x="turn_role", y=feat, order=ROLE_ORDER, color="black", alpha=0.24, size=2.4, jitter=0.20, ax=ax)
            sns.boxplot(data=df, x="turn_role", y=feat, order=ROLE_ORDER, color="white", width=0.13, showfliers=False, medianprops={"color": "black", "lw": 1.6}, ax=ax)
            ax.set_xlabel("")
        finite = df[feat].replace([np.inf, -np.inf], np.nan).dropna() if feat in df else pd.Series(dtype=float)
        y_min, y_max = float(finite.quantile(0.01)), float(finite.quantile(0.98)) if len(finite) else (0.0, 1.0)
        if feat.endswith("_ratio"):
            y_min = min(0.0, y_min)
            y_max = min(max(y_max, 1.25), cfg["acoustics"].get("ratio_clip", 5.0))
        pad = max((y_max - y_min) * 0.18, 0.05)
        ax.set_ylim(y_min - 0.2 * pad, y_max + 1.8 * pad)
        add_sig_bracket(ax, sig_label(q_lookup.get(feat)), y_max + 0.45 * pad, 0.18 * pad)
        ax.set_title(f"{letter} {FEATURE_LABELS[feat]}", loc="left", fontweight="bold")
        ax.set_ylabel("Value")

    fig.tight_layout(pad=2.8)
    save_figure(fig, out_dir, "Fig2_Raincloud")


def generate_fig3(df: pd.DataFrame, stats_json: Dict[str, Any], out_dir: Path) -> None:
    pair = paired_vectors(df, "emotion_incongruence")
    c = stats_json["robustness_summary"]["contagion"]
    q_incong = stats_json["robustness_summary"]["role_tests"]["emotion_incongruence"]["q_fdr"]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    sns.violinplot(data=df, x="turn_role", y="emotion_incongruence", palette=ROLE_PALETTE, order=ROLE_ORDER, inner=None, cut=0, linewidth=0, ax=axes[0])
    sns.stripplot(data=df, x="turn_role", y="emotion_incongruence", order=ROLE_ORDER, color="black", alpha=0.25, size=2.4, jitter=0.22, ax=axes[0])
    sns.boxplot(data=df, x="turn_role", y="emotion_incongruence", order=ROLE_ORDER, color="white", width=0.13, showfliers=False, medianprops={"color": "black", "lw": 1.6}, ax=axes[0])
    ymax = float(df["emotion_incongruence"].quantile(0.98))
    add_sig_bracket(axes[0], sig_label(q_incong), ymax + 0.08, 0.04)
    axes[0].set_ylim(0, ymax + 0.28)
    axes[0].set_title("(a) Incongruence distribution", loc="left", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(r"$\Delta_{incong}$")

    sns.regplot(data=pair, x="Analyst", y="Executive", ax=axes[1], scatter_kws={"alpha": 0.22, "s": 16, "color": "#777777"}, line_kws={"linewidth": 2.2, "color": ROLE_PALETTE["Executive"]})
    max_xy = max(float(pair["Analyst"].max()), float(pair["Executive"].max()), 1.0)
    axes[1].plot([0, max_xy], [0, max_xy], color="#666666", ls=":", lw=1.2)
    axes[1].annotate(f"Pearson r = {c['r']:.3f}\n95% CI [{c['ci_fisher'][0]:.3f}, {c['ci_fisher'][1]:.3f}]", xy=(0.05, 0.86), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.92))
    axes[1].set_title("(b) Adjacent-turn coupling", loc="left", fontweight="bold")
    axes[1].set_xlabel("Analyst incongruence")
    axes[1].set_ylabel("Executive incongruence")

    hb = axes[2].hexbin(pair["Analyst"], pair["Executive"], gridsize=24, mincnt=1, cmap="GnBu", alpha=0.82)
    sns.kdeplot(data=pair, x="Analyst", y="Executive", levels=5, color=ACCENT, linewidths=1.0, alpha=0.55, ax=axes[2])
    axes[2].plot([0, max_xy], [0, max_xy], color="#666666", ls=":", lw=1.2)
    axes[2].set_title("(c) Pair-density map", loc="left", fontweight="bold")
    axes[2].set_xlabel("Analyst incongruence")
    axes[2].set_ylabel("Executive incongruence")
    fig.colorbar(hb, ax=axes[2], shrink=0.78, label="Pair count")

    fig.tight_layout(pad=2.2)
    save_figure(fig, out_dir, "Fig3_Dynamics")



def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate EmoAlignBench paper figures and statistics from cached scores.")
    parser.add_argument("--config", default="configs/emobench_config.json")
    parser.add_argument("--input", default=None, help="Cached stress_pairs_with_acoustics.csv path.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--recompute-stats", action="store_true", help="Recompute Wilcoxon/FDR/bootstrap/permutation instead of using cached statistics_tables.json when present.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(resolve_path(root, args.config))
    input_csv = resolve_path(root, args.input or cfg["paths"]["cached_acoustics_csv"])
    out_dir = resolve_path(root, args.out or cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    remove_obsolete_outputs(out_dir)
    df = prepare_analysis_df(input_csv, cfg)
    stats_path = out_dir / "statistics_tables.json"
    if stats_path.exists() and not args.recompute_stats:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_json = normalize_stats_schema(json.load(f))
    else:
        stats_json = normalize_stats_schema(compute_statistics(df, cfg))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)
    df.to_csv(out_dir / "analysis_valid_pairs.csv", index=False)
    export_tables(df, stats_json, out_dir, cfg)

    generate_fig0(out_dir, cfg)
    generate_fig1(df, out_dir, int(cfg["statistics"].get("seed", SEED)))
    generate_fig2(df, stats_json, out_dir, cfg)
    generate_fig3(df, stats_json, out_dir)
    observed = int(df["pair_id"].nunique())
    expected = int(cfg["dataset"]["curated_valid_pairs"])
    if observed != expected:
        raise SystemExit(f"Expected {expected} valid pairs, observed {observed}.")
    print(f"Generated paper figures/tables for {observed} validated Analyst->Executive pairs.")


if __name__ == "__main__":
    main()
