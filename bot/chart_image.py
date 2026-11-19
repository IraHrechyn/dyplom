from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _fmt_pct(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}%"
    return f"{v:.1f}%"

def render_conclusion_chart(truth_pct: float, risk_pct: float) -> bytes:
    truth = max(0.0, float(truth_pct))
    risk = max(0.0, float(risk_pct))
    if truth + risk <= 0:
        truth, risk = 50.0, 50.0

    fig = plt.figure(figsize=(5.2, 6.6), facecolor="#000000")
    ax_title = fig.add_axes([0.05, 0.88, 0.9, 0.08])
    ax_pie = fig.add_axes([0.12, 0.28, 0.76, 0.58])
    ax_legend = fig.add_axes([0.08, 0.04, 0.84, 0.22])

    for ax in (ax_title, ax_pie, ax_legend):
        ax.set_facecolor("#000000")
        ax.axis("off")

    ax_title.text(
        0.5,
        0.5,
        "Загальний висновок",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=15,
        fontweight="semibold",
        transform=ax_title.transAxes,
    )

    ax_pie.pie(
        [truth, risk],
        colors=["#e8e8e8", "#dc2626"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="#000000", linewidth=0),
        radius=1.0,
    )
    ax_pie.set_aspect("equal")

    rows = [
        ("#e8e8e8", "Надійність", _fmt_pct(truth), "#ffffff"),
        ("#dc2626", "Ризик", _fmt_pct(risk), "#dc2626"),
    ]
    for i, (swatch, label, value, value_color) in enumerate(rows):
        y = 0.72 - i * 0.42
        ax_legend.plot([0.06], [y], marker="s", markersize=12, color=swatch, linestyle="None",
                       transform=ax_legend.transAxes, clip_on=False)
        ax_legend.text(
            0.14, y, label, color="#9ca3af", fontsize=12, va="center", ha="left",
            transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.94, y, value, color=value_color, fontsize=17, fontweight="bold",
            va="center", ha="right", transform=ax_legend.transAxes,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
