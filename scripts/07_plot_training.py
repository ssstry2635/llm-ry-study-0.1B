"""Rebuild the README loss figure from the historical CSV (no smoothing).

Run: python -m pip install -r requirements-viz.txt
     python scripts/07_plot_training.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = ROOT / "reports" / "metrics.original.csv"
    with src.open(encoding="utf-8", newline="") as f:
        original = list(csv.DictReader(f))
    # The 21/41-step runs used synthetic data; only real-corpus evaluations
    # at 250-step intervals belong to this figure. Last-write wins on resume.
    by_step = {}
    for row in original:
        step = int(row["iter"])
        if row["config"] == "012b" and 250 <= step <= 15000 and step % 250 == 0:
            by_step[step] = row
    rows = [by_step[s] for s in sorted(by_step)]
    if not rows:
        raise ValueError("No formal-training rows found")
    fields = ["iter", "train_loss", "val_loss", "lr"]
    with (ROOT / "reports" / "metrics.training.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    steps = [int(r["iter"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    best = min(range(len(rows)), key=lambda i: val[i])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.plot(steps, train, color="#0072B2", label="Train", lw=1.5,
            marker="o", markersize=3, markevery=3)
    ax.plot(steps, val, color="#D55E00", label="Validation", lw=1.5,
            linestyle="--", marker="s", markersize=3, markevery=3)
    ax.scatter([steps[best]], [val[best]], color="#D55E00", marker="*", s=130, zorder=5)
    ax.annotate(f"Best validation: {val[best]:.4f}\nStep {steps[best]:,}",
                xy=(steps[best], val[best]), xytext=(9800, 2.9),
                arrowprops={"arrowstyle": "->", "color": "#475569"},
                fontsize=10, bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#cbd5e1"})
    ax.set(title="RY Study 0.1B | Historical pretraining run", xlabel="Optimizer step",
           ylabel="Cross-entropy loss (nats)", xlim=(0, 15500), ylim=(0, 4.2))
    ax.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000],
                  ["0", "2,500", "5,000", "7,500", "10,000", "12,500", "15,000"])
    ax.grid(axis="y", color="#dce3eb", linewidth=.7)
    ax.legend(loc="upper right", frameon=False)
    fig.get_layout_engine().set(rect=(0, .06, 1, .90))
    fig.text(.5, .018, "60 evaluation steps; raw observations; last record per step; synthetic runs excluded.",
             ha="center", fontsize=9, color="#475569")
    out = ROOT / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "training-loss.png", dpi=180)
    fig.savefig(out / "training-loss.svg")
    svg = out / "training-loss.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    # Pixel-exact bounds check for text and axis labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise RuntimeError(f"Clipped label: {label.get_text()}")
    plt.close(fig)
    print(f"Selected {len(rows)} unique steps; best val {val[best]:.4f} at {steps[best]}")


if __name__ == "__main__":
    main()
