"""Static plots of a polling average."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_average(margins: pd.DataFrame, avg: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    shown = margins[margins["midpoint"].between(avg["date"].min(), avg["date"].max())]
    ax.scatter(shown["midpoint"], shown["margin"], s=10, alpha=0.35, color="#7a7a7a",
               label="Individual polls", linewidths=0)
    ax.fill_between(avg["date"], avg["margin"] - 1.96 * avg["se"], avg["margin"] + 1.96 * avg["se"],
                    color="#3b6fb6", alpha=0.18, linewidth=0, label="±1.96 SE")
    ax.plot(avg["date"], avg["margin"], color="#3b6fb6", linewidth=2, label="Time-weighted average")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Democratic margin (pts)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
