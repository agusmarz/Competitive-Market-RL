import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# palette, used by build_figure / build_summary_table
COMP_COLORS = {"seller_0": "#E63946", "seller_1": "#457B9D"}
COOP_COLORS = {"seller_0": "#F4A261", "seller_1": "#2A9D8F"}
GRID_ALPHA  = 0.18


def smooth(x, w=5):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def ts_for_smoothed(ts, w=5):
    if len(ts) < w:
        return np.array(ts)
    return np.array(ts[w - 1:])


def build_figure(comp: dict, coop: dict, unit_cost: float = 8.0, mu_budget: float = 25.0) -> plt.Figure:
    # 8-panel figure: price, reward, price gap, price correlation, sales and loyalty
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "figure.facecolor": "#F8F7F4",
        "axes.facecolor":   "#F8F7F4",
        "axes.grid":        True,
        "grid.color":       "#CCCCCC",
        "grid.linewidth":   0.5,
        "grid.alpha":       GRID_ALPHA,
    })

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        "SharedIPPO — Competitive vs Cooperative Reward  (250k timesteps)",
        fontsize=15, fontweight="bold", y=0.98, color="#1A1A2E",
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.3,
                           top=0.94, bottom=0.04, left=0.07, right=0.97)

    ax_pc = fig.add_subplot(gs[0, 0])   # price – competitive
    ax_po = fig.add_subplot(gs[0, 1])   # price – cooperative
    ax_rc = fig.add_subplot(gs[1, 0])   # reward – competitive
    ax_ro = fig.add_subplot(gs[1, 1])   # reward – cooperative
    ax_gap  = fig.add_subplot(gs[2, 0]) # price gap
    ax_corr = fig.add_subplot(gs[2, 1]) # price correlation
    ax_sales = fig.add_subplot(gs[3, 0])# sales (both runs)
    ax_loy   = fig.add_subplot(gs[3, 1])# loyalty (both runs)

    agent_ids = list(comp["agents"].keys())
    w = 5  # smoothing window

    def ts(m): return m["timesteps"]
    def price(m, a): return m["agents"][a]["avg_price"]
    def rew(m, a):   return m["agents"][a]["avg_reward"]
    def sales(m, a): return m["agents"][a]["avg_sales"]
    def loy(m, a):   return m["agents"][a]["avg_loyalty"]

    # price panels, with buyer budget and unit cost as reference lines
    ax_pc.set_title("Avg Price per Iteration — Competitive")
    ax_pc.axhline(mu_budget,  color="#A8DADC", lw=1.5, ls="--",
                  label=f"Avg buyer budget (μ={mu_budget:.0f})", zorder=3)
    ax_pc.axhline(unit_cost,  color="#E9C46A", lw=1.5, ls=":",
                  label=f"Unit cost (c={unit_cost:.0f})", zorder=3)
    for a, col in COMP_COLORS.items():
        raw = price(comp, a)
        ax_pc.plot(ts(comp), raw, color=col, alpha=0.20, lw=0.8)
        ax_pc.plot(ts_for_smoothed(ts(comp), w), smooth(raw, w),
                   color=col, lw=2.0, label=a)
    ax_pc.set_xlabel("Agent-steps"); ax_pc.set_ylabel("Price")
    ax_pc.legend(fontsize=8)

    ax_po.set_title("Avg Price per Iteration — Cooperative")
    ax_po.axhline(mu_budget,  color="#A8DADC", lw=1.5, ls="--",
                  label=f"Avg buyer budget (μ={mu_budget:.0f})", zorder=3)
    ax_po.axhline(unit_cost,  color="#E9C46A", lw=1.5, ls=":",
                  label=f"Unit cost (c={unit_cost:.0f})", zorder=3)
    for a, col in COOP_COLORS.items():
        raw = price(coop, a)
        ax_po.plot(ts(coop), raw, color=col, alpha=0.20, lw=0.8)
        ax_po.plot(ts_for_smoothed(ts(coop), w), smooth(raw, w),
                   color=col, lw=2.0, label=a)
    ax_po.set_xlabel("Agent-steps"); ax_po.set_ylabel("Price")
    ax_po.legend(fontsize=8)

    # reward panels
    for ax, m, colors, title in [
        (ax_rc, comp, COMP_COLORS, "Episode Reward — Competitive"),
        (ax_ro, coop, COOP_COLORS, "Episode Reward — Cooperative"),
    ]:
        ax.set_title(title)
        for a, col in colors.items():
            raw = rew(m, a)
            ax.plot(ts(m), raw, color=col, alpha=0.20, lw=0.8)
            ax.plot(ts_for_smoothed(ts(m), w), smooth(raw, w),
                    color=col, lw=2.0, label=a)
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("Agent-steps"); ax.set_ylabel("Total reward / episode")
        ax.legend(fontsize=8)

    # price gap, symmetry indicator
    ax_gap.set_title("|Price Gap|  p₀ − p₁  (symmetry indicator)")
    for m, col, lbl in [(comp, "#E63946", "Competitive"),
                         (coop, "#2A9D8F", "Cooperative")]:
        raw = m["price_gap"]
        ax_gap.plot(ts(m), raw, color=col, alpha=0.20, lw=0.8)
        ax_gap.plot(ts_for_smoothed(ts(m), w), smooth(raw, w),
                    color=col, lw=2.0, label=lbl)
    ax_gap.axhline(0, color="#999999", lw=0.8, ls=":")
    ax_gap.set_xlabel("Agent-steps"); ax_gap.set_ylabel("|p₀ − p₁|")
    ax_gap.legend(fontsize=8)

    # price correlation, coordination signal
    ax_corr.set_title("Price Correlation ρ(p₀, p₁)  (coordination signal)")
    for m, col, lbl in [(comp, "#E63946", "Competitive"),
                         (coop, "#2A9D8F", "Cooperative")]:
        raw = m["price_correlation"]
        ax_corr.plot(ts(m), raw, color=col, alpha=0.20, lw=0.8)
        ax_corr.plot(ts_for_smoothed(ts(m), w), smooth(raw, w),
                     color=col, lw=2.0, label=lbl)
    ax_corr.axhline(0, color="#999999", lw=0.8, ls=":")
    ax_corr.axhline( 1, color="#AAAAAA", lw=0.6, ls=":", alpha=0.5)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.set_xlabel("Agent-steps"); ax_corr.set_ylabel("Pearson ρ")
    ax_corr.legend(fontsize=8)

    # sales, only seller_0 shown to keep the panel readable
    ax_sales.set_title("Avg Sales per Step  (both runs, seller_0 shown)")
    a0 = agent_ids[0]
    for m, col, lbl in [(comp, "#E63946", "Competitive"),
                         (coop, "#2A9D8F", "Cooperative")]:
        raw = sales(m, a0)
        ax_sales.plot(ts(m), raw, color=col, alpha=0.20, lw=0.8)
        ax_sales.plot(ts_for_smoothed(ts(m), w), smooth(raw, w),
                      color=col, lw=2.0, label=lbl)
    ax_sales.set_xlabel("Agent-steps"); ax_sales.set_ylabel("Sales / step")
    ax_sales.legend(fontsize=8)

    # loyalty, same simplification as sales
    ax_loy.set_title("Avg Loyalty  (both runs, seller_0 shown)")
    for m, col, lbl in [(comp, "#E63946", "Competitive"),
                         (coop, "#2A9D8F", "Cooperative")]:
        raw = loy(m, a0)
        ax_loy.plot(ts(m), raw, color=col, alpha=0.20, lw=0.8)
        ax_loy.plot(ts_for_smoothed(ts(m), w), smooth(raw, w),
                    color=col, lw=2.0, label=lbl)
    ax_loy.set_ylim(0, 1)
    ax_loy.set_xlabel("Agent-steps"); ax_loy.set_ylabel("Loyalty")
    ax_loy.legend(fontsize=8)

    return fig


def build_summary_table(comp: dict, coop: dict, unit_cost: float = 8.0, mu_budget: float = 25.0) -> plt.Figure:
    # compact table with the final-iteration stats, next to budget and cost references
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#F8F7F4")
    ax.set_facecolor("#F8F7F4")
    ax.axis("off")

    N = 5
    def tail_mean(lst): return float(np.mean(lst[-N:]))

    rows = []
    for a in list(comp["agents"].keys()):
        rows.append([
            a,
            f"{tail_mean(comp['agents'][a]['avg_price']):.3f}",
            f"{tail_mean(coop['agents'][a]['avg_price']):.3f}",
            f"{mu_budget:.1f}",
            f"{unit_cost:.1f}",
            f"{tail_mean(comp['agents'][a]['avg_reward']):.4f}",
            f"{tail_mean(coop['agents'][a]['avg_reward']):.4f}",
            f"{tail_mean(comp['agents'][a]['avg_sales']):.3f}",
            f"{tail_mean(coop['agents'][a]['avg_sales']):.3f}",
        ])

    cols = ["Agent",
            "Price\n(Competitive)", "Price\n(Cooperative)",
            "Avg Buyer\nBudget (μ)", "Unit\nCost (c)",
            "Reward\n(Comp)", "Reward\n(Coop)",
            "Sales\n(Comp)", "Sales\n(Coop)"]

    t = ax.table(
        cellText=rows, colLabels=cols,
        loc="center", cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.0)

    # header styling
    for j in range(len(cols)):
        t[(0, j)].set_facecolor("#1A1A2E")
        t[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, start=1):
        bg = "#EEF0F3" if i % 2 == 0 else "#F8F7F4"
        for j in range(len(cols)):
            t[(i, j)].set_facecolor(bg)

    ax.set_title(
        f"Summary — Final {N}-iteration averages  |  Avg buyer budget μ={mu_budget:.0f}  |  Unit cost c={unit_cost:.0f}",
        fontsize=11, fontweight="bold", pad=20, color="#1A1A2E",
    )

    return fig
