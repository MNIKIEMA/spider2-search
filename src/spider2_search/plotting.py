from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(df: pd.DataFrame, metrics=None, save_dir="plots", figsize=(10, 6)):
    """
    Plot one figure per metric, combining hybrid and vector search types.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation results with columns: ['metric', 'k', 'reranker', 'embedding_model', 'query_type', 'score']
    metrics : list, optional
        Metrics to plot. Defaults to all available.
    save_dir : str, optional
        Directory to save figures. If None, plots are shown instead.
    figsize : tuple, optional
        Size of each figure.
    """
    save_dir = Path(save_dir)
    df = df[df["query_type"].isin(["vector", "hybrid"])]

    available_metrics = sorted(df["metric"].unique())
    available_models = sorted(df["embedding_model"].unique())

    if metrics is None:
        metrics = available_metrics
    elif isinstance(metrics, str):
        metrics = [metrics]

    colors = plt.get_cmap("tab10")
    markers = ["o", "^", "s", "D", "X", "*", "p", "h", "+", "x"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=figsize)

        filtered_df = df[df["metric"] == metric]
        color_idx = 0

        for model in available_models:
            for query_type in ["vector", "hybrid"]:
                group_df = filtered_df[
                    (filtered_df["embedding_model"] == model)
                    & (filtered_df["query_type"] == query_type)
                ]

                if not group_df.empty:
                    group_df = group_df.sort_values("k")
                    label = f"{model.split('-')[-1]}"
                    label += f" [{query_type}]"

                    ax.plot(
                        group_df["k"],
                        group_df["avg_score"],
                        label=label,
                        marker=markers[color_idx % len(markers)],
                        color=colors(color_idx % 10),
                        linewidth=2,
                        markersize=7,
                    )
                    color_idx += 1

        ax.set_title(f"{metric.upper()}@k for Hybrid & Vector Search", fontsize=14)
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

        if metric.lower() in ["mrr", "precision", "recall", "f1"]:
            ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{metric}_hybrid_vector.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"✅ Saved: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    return
