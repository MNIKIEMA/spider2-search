import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(
    df: pd.DataFrame, metrics=None, query_types=None, save_path=None, figsize=(20, 12)
):
    """
    Plot metrics by k for different models, rerankers, and query types.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing evaluation results with columns:
        ['metric', 'k', 'reranker', 'embedding_model', 'query_type', 'score']
    metrics : list, optional
        List of metrics to plot. If None, all available metrics are plotted.
    query_types : list, optional
        List of query types to plot. If None, all available query types are plotted.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """

    # Get unique values for filtering
    available_metrics = sorted(df["metric"].unique())
    available_query_types = sorted(df["query_type"].unique())
    available_models = sorted(df["embedding_model"].unique())
    available_rerankers = sorted(df["reranker"].unique())

    # Set default parameters if not provided
    if metrics is None:
        metrics = available_metrics
    elif isinstance(metrics, str):
        metrics = [metrics]

    if query_types is None:
        query_types = available_query_types
    elif isinstance(query_types, str):
        query_types = [query_types]

    # Calculate grid layout
    n_metrics = len(metrics)
    n_query_types = len(query_types)
    n_plots = n_metrics * n_query_types

    n_cols = min(n_query_types, 3)  # Maximum 3 columns
    n_rows = int(np.ceil(n_plots / n_cols))

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes array for easy iteration
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Set up color and marker cycles
    colors = plt.get_cmap("tab10")
    markers = ["o", "^", "s", "D", "X", "*", "p", "h", "+", "x"]

    # Plot each metric and query type combination
    plot_idx = 0
    for metric in metrics:
        for query_type in query_types:
            if plot_idx >= len(axes):
                print(
                    f"Warning: Not enough axes for all combinations. Skipping {metric}, {query_type}."
                )
                continue

            ax = axes[plot_idx]

            # Filter data for current metric and query type
            filtered_data = df[(df["metric"] == metric) & (df["query_type"] == query_type)]

            if filtered_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for\n{metric} + {query_type}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                plot_idx += 1
                continue

            # Plot each model and reranker combination
            color_idx = 0
            for model in available_models:
                marker_idx = 0
                for reranker in available_rerankers:
                    data = filtered_data[
                        (filtered_data["reranker"] == reranker)
                        & (filtered_data["embedding_model"] == model)
                    ]

                    if len(data) > 0:
                        # Sort by k to ensure proper line plotting
                        data = data.sort_values("k")

                        # Create label with model and reranker
                        label = f"{model.split('-')[-1]}"
                        if reranker != "none":
                            label += f" + {reranker.split('-')[-1]}"

                        # Plot the data
                        ax.plot(
                            data["k"],
                            data["score"],
                            marker=markers[marker_idx % len(markers)],
                            color=colors[color_idx % len(colors)],
                            label=label,
                            linewidth=2,
                            markersize=8,
                        )

                    marker_idx += 1
                color_idx += 1

            # Set title and labels
            ax.set_title(f"{metric.upper()}@k for {query_type.capitalize()} Search", fontsize=14)
            ax.set_xlabel("k", fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add legend with smaller font
            ax.legend(fontsize=10, loc="lower right")

            # Adjust y-axis limits
            if metric.lower() in ["mrr", "precision", "recall", "f1"]:
                ax.set_ylim(0, 1.05)

            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig

