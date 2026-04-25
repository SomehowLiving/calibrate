import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None


def generate_leaderboard(output_dir: str, save_dir: str) -> None:
    base_path = Path(output_dir).expanduser().resolve()
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    model_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"No model folders found under {base_path}")
        return

    model_results: Dict[str, Dict[str, float]] = {}
    model_normalized: Dict[str, Dict[str, float]] = {}
    criterion_info: Dict[str, dict] = {}
    metric_names: List[str] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        result = _read_metrics(model_dir / "metrics.json")
        if result is None:
            continue

        display, normalized, info = result
        model_results[model_name] = display
        model_normalized[model_name] = normalized
        for name, ci in info.items():
            criterion_info.setdefault(name, ci)

        # Collect all metric names
        for metric_name in display.keys():
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    if not model_results:
        print("No results found to compile.")
        return

    leaderboard_df = _build_leaderboard(
        model_results, model_normalized, metric_names
    )
    csv_path = save_path / "simulation_leaderboard.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"Saved leaderboard CSV to {csv_path}")

    chart_path = save_path / "simulation_leaderboard.png"
    _create_comparison_chart(
        leaderboard_df, model_normalized, criterion_info, chart_path
    )


def _read_metrics(
    metrics_path: Path,
) -> Optional[Tuple[Dict[str, float], Dict[str, float], Dict[str, dict]]]:
    """Read metrics.json and return (display, normalized, info) per-metric dicts.

    - ``display``: raw value to show in the CSV.
      Binary metrics are 0-1 fractions → converted to % for chart readability.
      Rating metrics are raw means on the criterion's own scale → kept as-is.
    - ``normalized``: 0-100 value used both for computing the unit-consistent
      ``overall`` column and for plotting rating bars without misleading
      scale. Binary already in 0-100 after conversion; rating is rescaled
      via ``(mean - scale_min) / (scale_max - scale_min) * 100``. Legacy
      flat values are treated as binary.
    - ``info``: per-metric ``{type, scale_min?, scale_max?}`` so the chart
      can label rating columns with their original scale.
    """
    if not metrics_path.exists():
        print(f"[WARN] metrics.json missing for {metrics_path.parent}")
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse {metrics_path}")
        return None

    display: Dict[str, float] = {}
    normalized: Dict[str, float] = {}
    info: Dict[str, dict] = {}

    for metric_name, metric_data in data.items():
        if isinstance(metric_data, dict) and "mean" in metric_data:
            is_rating_metric = metric_data.get("type") == "rating"
            mean = float(metric_data["mean"])
            if is_rating_metric:
                display[metric_name] = mean
                scale_min = float(metric_data.get("scale_min", 0))
                scale_max = float(metric_data.get("scale_max", 1))
                scale_range = scale_max - scale_min
                if scale_range > 0:
                    normalized[metric_name] = (
                        (mean - scale_min) / scale_range * 100
                    )
                else:
                    normalized[metric_name] = 0.0
                info[metric_name] = {
                    "type": "rating",
                    "scale_min": metric_data.get("scale_min"),
                    "scale_max": metric_data.get("scale_max"),
                }
            else:
                display[metric_name] = mean * 100
                normalized[metric_name] = mean * 100
                info[metric_name] = {"type": "binary"}
        elif isinstance(metric_data, (int, float)):
            # Legacy flat value — assume binary pass-rate in [0,1]
            display[metric_name] = float(metric_data) * 100
            normalized[metric_name] = float(metric_data) * 100
            info[metric_name] = {"type": "binary"}

    return display, normalized, info


def _build_leaderboard(
    model_results: Dict[str, Dict[str, float]],
    model_normalized: Dict[str, Dict[str, float]],
    metric_names: List[str],
) -> pd.DataFrame:
    """Build the leaderboard DataFrame.

    Per-metric columns show the raw ``display`` value (binary → %, rating →
    raw mean on the criterion's scale). The ``overall`` column is computed
    from the ``normalized`` (0-100) values so it stays unit-consistent even
    when the config mixes binary and rating criteria on different scales.
    """
    rows = []
    for model_name in sorted(model_results):
        row: Dict[str, Optional[float]] = {"model": model_name}
        for metric in metric_names:
            row[metric] = model_results[model_name].get(metric)

        # Overall = mean of normalized (0-100) values — unit-consistent
        normalized_values = [
            v
            for v in model_normalized.get(model_name, {}).values()
            if isinstance(v, (int, float))
        ]
        row["overall"] = (
            sum(normalized_values) / len(normalized_values)
            if normalized_values
            else None
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _create_comparison_chart(
    df: pd.DataFrame,
    model_normalized: Dict[str, Dict[str, float]],
    criterion_info: Dict[str, dict],
    chart_path: Path,
) -> None:
    """Render the grouped bar chart.

    Rating bars use the normalized 0-100 values (the CSV keeps raw means)
    so binary % and rating means appear on a single comparable axis. Rating
    columns are renamed with a ``(rating N-M)`` suffix so the original
    scale stays visible. The y-axis label and title flip when any rating
    metric is present.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate charts. Please install it."
        )

    if df.empty:
        print("Leaderboard dataframe is empty, skipping chart creation.")
        return

    metric_columns = [col for col in df.columns if col not in ["model", "overall"]]
    if not metric_columns:
        print("No metrics available for charting.")
        return

    # Build a chart-only copy: rating columns use normalized values.
    chart_df = df.set_index("model")[metric_columns].copy()
    has_rating = False
    rename_map: Dict[str, str] = {}
    for name in metric_columns:
        info = criterion_info.get(name) or {}
        if info.get("type") == "rating":
            has_rating = True
            scale_min = info.get("scale_min")
            scale_max = info.get("scale_max")
            for model_name in chart_df.index:
                norm = model_normalized.get(model_name, {}).get(name)
                if norm is not None:
                    chart_df.loc[model_name, name] = float(norm)
            if scale_min is not None and scale_max is not None:
                rename_map[name] = f"{name} (rating {scale_min}-{scale_max})"

    chart_df = chart_df.rename(columns=rename_map)
    plot_df = chart_df.T

    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 1.5), 5))
    plot_df.plot(kind="bar", ax=ax)
    if has_rating:
        ax.set_ylabel("Score (%) — rating criteria normalized to scale")
        ax.set_title("Model Score by Metric (normalized)")
    else:
        ax.set_ylabel("Score %")
        ax.set_title("Model Score by Metric")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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
        help="Path to the output directory with model subdirectories",
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
