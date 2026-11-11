import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paired test results (results/test_paired_output.txt): model minus baseline MAE, 95% CI
rows = [
    ("Linear (global)",          -0.0066, -0.0133, +0.0003),
    ("Linear (positional)",      -0.0062, -0.0143, +0.0015),
    ("Random Forest (global)",   -0.0092, -0.0165, -0.0014),
    ("Random Forest (positional)",-0.0070, -0.0149, +0.0004),
    ("XGBoost (global)",         -0.0033, -0.0121, +0.0051),
    ("XGBoost (positional)",     -0.0068, -0.0151, +0.0019),
]
INK, MUTED, GRID, ACCENT = "#1f2328", "#6b7280", "#d9dce1", "#1d6fb8"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(3.4, 2.1), dpi=300)
n = len(rows)
for i, (name, d, lo, hi) in enumerate(rows):
    y = n - 1 - i
    sig = hi < 0
    c = ACCENT if sig else MUTED
    ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot([d], [y], marker="o", ms=5, mfc=c if sig else "white", mec=c, mew=1.2, zorder=3)
ax.axvline(0, color=INK, lw=0.9, zorder=1)
ax.set_yticks(range(n))
ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
ax.set_xlim(-0.019, 0.007)
ax.set_xlabel("MAE change vs. baseline (\u00d710\u207b\u00b3)", color=INK)
ax.set_xticks([-0.015, -0.010, -0.005, 0.0, 0.005])
ax.set_xticklabels(["\u221215", "\u221210", "\u22125", "0", "+5"])
ax.text(0.0005, n - 0.45, "worse", color=MUTED, fontsize=6.5, ha="left", va="bottom")
ax.text(-0.0005, n - 0.45, "better", color=MUTED, fontsize=6.5, ha="right", va="bottom")
ax.set_ylim(-0.6, n - 0.1 + 0.4)
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED, length=2)
fig.tight_layout(pad=0.4)
from pathlib import Path
out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig_paired_delta_mae.png", dpi=300)
print("ok")
