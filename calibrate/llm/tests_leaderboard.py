import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None


def generate_leaderboard(output_dir: str, save_dir: str) -> None:
    """
    Generate leaderboard from model results in output_dir.

    Expected structure:
        output_dir/
            model1/
                metrics.json  (contains {"total": N, "passed": M, "criteria": {...}})
            model2/
                metrics.json
            ...

    The leaderboard shows:
    - Overall `pass_rate` (all test cases)
    - Per-criterion pass rate columns when the suite has response-type tests
      (e.g., `accuracy`, `tone`)

    Args:
        output_dir: Directory containing model subdirectories with metrics.json files
        save_dir: Directory where leaderboard artifacts will be saved
    """
    base_path = Path(output_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    # Find model directories (skip 'leaderboard' folder if present)
    model_dirs = sorted(
        p for p in base_path.iterdir()
        if p.is_dir() and p.name != "leaderboard"
    )

    if not model_dirs:
        print(f"No model folders found under {base_path}")
        return

    model_data: Dict[str, dict] = {}
    for model_dir in model_dirs:
        data = _read_metrics(model_dir / "metrics.json")
        if data is None:
            continue
        model_data[model_dir.name] = data

    if not model_data:
        print("No results found to compile.")
        return

    # Collect union of criterion names across all models (sorted for stable column order)
    criterion_names: List[str] = sorted(
        {
            name
            for data in model_data.values()
            for name in (data.get("criteria") or {}).keys()
        }
    )

    # Resolve type + scale per criterion (first model that has it wins)
    criterion_info: Dict[str, dict] = {}
    for data in model_data.values():
        for name, crit in (data.get("criteria") or {}).items():
            if name not in criterion_info and isinstance(crit, dict):
                info = {"type": crit.get("type", "binary")}
                if crit.get("type") == "rating":
                    info["scale_min"] = crit.get("scale_min")
                    info["scale_max"] = crit.get("scale_max")
                criterion_info[name] = info

    leaderboard_df = _build_leaderboard(model_data, criterion_names)
    csv_path = save_path / "llm_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")

    chart_path = save_path / "llm_leaderboard.png"
    _create_comparison_chart(
        leaderboard_df, criterion_names, criterion_info, chart_path
    )


def _read_metrics(metrics_path: Path) -> Optional[dict]:
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent}")
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse {metrics_path}")
        return None

    return data


def _to_percent(passed: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return (passed / total) * 100


def _build_leaderboard(
    model_data: Dict[str, dict],
    criterion_names: List[str],
) -> pd.DataFrame:
    """Build leaderboard DataFrame.

    Columns: model, passed, total, pass_rate, [criterion_1, criterion_2, ...]

    Per-criterion column values:
    - binary criterion → pass_rate (%)
    - rating criterion → mean score (raw, on the criterion's scale)
    """
    rows = []
    for model_name in sorted(model_data):
        data = model_data[model_name]
        passed = int(data.get("passed", 0))
        total = int(data.get("total", 0))
        row: Dict[str, object] = {
            "model": model_name,
            "passed": passed,
            "total": total,
            "pass_rate": _to_percent(passed, total),
        }

        criteria = data.get("criteria") or {}
        for name in criterion_names:
            crit = criteria.get(name)
            if not crit:
                row[name] = None
            elif crit.get("type") == "rating":
                row[name] = crit.get("mean")
            else:
                row[name] = crit.get("pass_rate")

        rows.append(row)

    return pd.DataFrame(rows)


def _create_comparison_chart(
    df: pd.DataFrame,
    criterion_names: List[str],
    criterion_info: Dict[str, dict],
    chart_path: Path,
) -> None:
    """Render the grouped bar chart.

    Rating columns are normalized to 0-100 for the chart only (the CSV
    keeps the raw mean) so binary % and rating means appear on the same
    y-axis without misleading scale or clipping. The label and title
    reflect that the values shown are normalized scores. The accompanying
    label column suffix ``(rating N-M)`` makes the original scale visible.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    if df.empty:
        print("Leaderboard dataframe is empty, skipping chart creation.")
        return

    metric_columns = ["pass_rate"] + [c for c in criterion_names if c in df.columns]

    if len(metric_columns) == 1:
        _simple_pass_rate_chart(df, chart_path)
        return

    # Build a chart-only copy of the DataFrame: rating columns scaled to %.
    # Also rename rating columns to expose their raw scale on the x-axis.
    chart_df = df.set_index("model")[metric_columns].copy()
    has_rating = False
    rename_map: Dict[str, str] = {}
    for name in criterion_names:
        if name not in chart_df.columns:
            continue
        info = criterion_info.get(name) or {}
        if info.get("type") == "rating":
            has_rating = True
            scale_min = info.get("scale_min")
            scale_max = info.get("scale_max")
            if (
                scale_min is not None
                and scale_max is not None
                and scale_max > scale_min
            ):
                rng = float(scale_max) - float(scale_min)
                chart_df[name] = (
                    (chart_df[name].astype(float) - float(scale_min)) / rng * 100
                )
                rename_map[name] = f"{name} (rating {scale_min}-{scale_max})"

    chart_df = chart_df.rename(index=rename_map)
    plot_df = chart_df.T

    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 1.5), 5))
    plot_df.plot(kind="bar", ax=ax)
    if has_rating:
        ax.set_ylabel("Score (%) — rating criteria normalized to scale")
        ax.set_title("LLM Test Score by Metric (normalized)")
    else:
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title("LLM Test Pass Rate by Metric")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison chart to {chart_path}")


def _simple_pass_rate_chart(df: pd.DataFrame, chart_path: Path) -> None:
    """Original flat bar chart — used when no per-criterion data exists."""
    if "pass_rate" not in df.columns:
        print("No pass_rate column available for charting.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))

    models = df["model"].tolist()
    pass_rates = df["pass_rate"].tolist()

    bars = ax.bar(models, pass_rates, color="steelblue")

    for bar, rate in zip(bars, pass_rates):
        if rate is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Pass Rate (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 105)
    ax.set_title("LLM Test Pass Rate by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison chart to {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory with scenario subdirectories",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="Directory where leaderboard artifacts will be stored",
    )
    args = parser.parse_args()
    generate_leaderboard(args.output_dir, args.save_dir)


if __name__ == "__main__":
    main()
