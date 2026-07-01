import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe


# theme constants
BG       = "#0D1117"   # deep github-dark background
BG_PANEL = "#161B22"   # slightly lighter panel
FG       = "#E6EDF3"   # main text
FG_DIM   = "#7D8590"   # secondary text / grid

# competitive palette (warm, tension)
C0 = "#FF6B6B"   # seller_0 red-coral
C1 = "#4ECDC4"   # seller_1 teal

# cooperative palette (cool, harmony)
O0 = "#FF9F43"   # seller_0 orange
O1 = "#54A0FF"   # seller_1 sky-blue

BAND_ALPHA_WIDE   = 0.10
BAND_ALPHA_NARROW = 0.22
LINE_ALPHA_RAW    = 0.18
LINE_W            = 2.2

FONT_TICK   = "Latin Modern Roman"


def load_or_synthesise():
    # loads real JSON metrics if they exist, otherwise fakes plausible data for a preview
    comp_path = Path("results/metrics_competitive.json")
    coop_path = Path("results/metrics_cooperative.json")

    if comp_path.exists() and coop_path.exists():
        print("  Loading real experiment data …")
        with open(comp_path) as f: comp = json.load(f)
        with open(coop_path) as f: coop = json.load(f)
        return comp, coop

    print("  No JSON found — generating synthetic data for preview …")
    rng = np.random.default_rng(0)

    def make_metrics(label, price_fn_0, price_fn_1, rew_scale, reward_noise):
        n = 25
        ts = list(np.arange(1, n + 1) * 10_000)
        t  = np.linspace(0, 1, n)

        p0 = price_fn_0(t) + rng.normal(0, 1.2, n)
        p1 = price_fn_1(t) + rng.normal(0, 1.2, n)
        s0 = np.clip(3.0 - 0.5 * p0 / 10 + rng.normal(0, 0.3, n), 0.1, 5)
        s1 = np.clip(3.0 - 0.5 * p1 / 10 + rng.normal(0, 0.3, n), 0.1, 5)
        l0 = 0.3 + 0.4 * t + rng.normal(0, 0.04, n)
        l1 = 0.3 + 0.35 * t + rng.normal(0, 0.04, n)
        r0 = rew_scale * (p0 - 8) * s0 / 420 + rng.normal(0, reward_noise, n)
        r1 = rew_scale * (p1 - 8) * s1 / 420 + rng.normal(0, reward_noise, n)
        gap  = list(np.abs(p0 - p1))
        corr = list(np.linspace(-0.7, -0.1, n) + rng.normal(0, 0.15, n))
        return {
            "label": label, "timesteps": ts,
            "agents": {
                "seller_0": {"avg_price": list(p0), "avg_sales": list(s0),
                             "avg_loyalty": list(l0), "avg_reward": list(r0)},
                "seller_1": {"avg_price": list(p1), "avg_sales": list(s1),
                             "avg_loyalty": list(l1), "avg_reward": list(r1)},
            },
            "price_gap": gap,
            "price_correlation": corr,
            "actor_loss": list(-0.02 * np.exp(-3 * t) + rng.normal(0, 0.002, n)),
            "critic_loss": list(0.06 * np.exp(-4 * t) + rng.normal(0, 0.002, n)),
        }

    # competitive: price wars, then a gradual descent
    comp = make_metrics(
        "Competitive",
        lambda t: 35 - 14 * t + 4 * np.sin(6 * t),
        lambda t: 32 - 11 * t + 4 * np.cos(6 * t),
        rew_scale=1.0, reward_noise=0.003,
    )
    # cooperative: chaotic at first, then settles at a higher price
    coop = make_metrics(
        "Cooperative",
        lambda t: 20 + 10 * np.sin(8 * t) * np.exp(-3 * t) + 5 * np.sin(3 * t),
        lambda t: 15 + 8  * np.cos(8 * t) * np.exp(-3 * t) + 4 * np.cos(3 * t),
        rew_scale=1.8, reward_noise=0.010,
    )
    return comp, coop


def rolling(x, w=5):
    # rolling mean and std, for the uncertainty bands
    x = np.array(x, dtype=float)
    n = len(x)
    if n < w:
        return x, np.zeros_like(x)
    mu  = uniform_filter1d(x, size=w, mode="nearest")
    sq  = uniform_filter1d(x ** 2, size=w, mode="nearest")
    std = np.sqrt(np.maximum(sq - mu ** 2, 0))
    return mu, std


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":     BG,
        "axes.facecolor":       BG_PANEL,
        "axes.edgecolor":       FG_DIM,
        "axes.labelcolor":      FG_DIM,
        "axes.titlecolor":      FG,
        "axes.titlesize":       11,
        "axes.titleweight":     "bold",
        "axes.titlepad":        10,
        "axes.labelsize":       9,
        "axes.grid":            True,
        "grid.color":           FG_DIM,
        "grid.linewidth":       0.35,
        "grid.alpha":           0.25,
        "grid.linestyle":       ":",
        "xtick.color":          FG_DIM,
        "ytick.color":          FG_DIM,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "text.color":           FG,
        "legend.facecolor":     "#1C2128",
        "legend.edgecolor":     FG_DIM,
        "legend.fontsize":      8,
        "legend.framealpha":    0.85,
        "font.family":          FONT_TICK,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle":"round",
    })


def styled_ax(ax, title, xlabel="Agent-steps", ylabel=""):
    ax.set_title(title, color=FG, pad=10)
    ax.set_xlabel(xlabel,  color=FG_DIM, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel,color=FG_DIM, labelpad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(FG_DIM)
    ax.spines["bottom"].set_color(FG_DIM)
    ax.tick_params(axis="both", which="both", color=FG_DIM)


def band_plot(ax, ts, raw, color, label, w=5):
    # layered uncertainty band: +/-2std, +/-1std, mean line, and the raw dots
    ts  = np.array(ts)
    mu, std = rolling(raw, w)

    ax.scatter(ts, raw, color=color, s=8, alpha=LINE_ALPHA_RAW,
               zorder=1, linewidths=0)
    ax.fill_between(ts, mu - 2 * std, mu + 2 * std,
                    color=color, alpha=BAND_ALPHA_WIDE, zorder=2, linewidth=0)
    ax.fill_between(ts, mu - std, mu + std,
                    color=color, alpha=BAND_ALPHA_NARROW, zorder=3, linewidth=0)
    ax.plot(ts, mu, color=color, lw=LINE_W, zorder=4, label=label,
            path_effects=[pe.Stroke(linewidth=LINE_W + 1.5,
                                    foreground=BG, alpha=0.4),
                          pe.Normal()])


def build_main_figure(comp, coop, unit_cost: float = 8.0, mu_budget: float = 25.0):
    # 8-panel figure, dark theme version of build_figure
    setup_style()

    fig = plt.figure(figsize=(17, 20))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975,
             "SharedIPPO — Competitive vs Cooperative Reward",
             ha="center", va="top",
             fontsize=18, color=FG, fontweight="bold")
    fig.text(0.5, 0.963,
             "Shaded bands: +/-1std  /  +/-2std rolling window  |  Dots: raw iteration values  |  250 k agent-steps",
             ha="center", va="top",
              fontsize=9.5, color=FG_DIM, style="italic")

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        top=0.945, bottom=0.045,
        left=0.065, right=0.975,
        hspace=0.52, wspace=0.28,
    )

    ax_pc  = fig.add_subplot(gs[0, 0])
    ax_po  = fig.add_subplot(gs[0, 1])
    ax_rc  = fig.add_subplot(gs[1, 0])
    ax_ro  = fig.add_subplot(gs[1, 1])
    ax_gap = fig.add_subplot(gs[2, 0])
    ax_cor = fig.add_subplot(gs[2, 1])
    ax_sal = fig.add_subplot(gs[3, 0])
    ax_loy = fig.add_subplot(gs[3, 1])

    ts_c = comp["timesteps"]
    ts_o = coop["timesteps"]

    # prices
    for ax, m, ts, ca, cb, run_label in [
        (ax_pc, comp, ts_c, C0, C1, "Competitive"),
        (ax_po, coop, ts_o, O0, O1, "Cooperative"),
    ]:
        band_plot(ax, ts, m["agents"]["seller_0"]["avg_price"], ca, "seller 0")
        band_plot(ax, ts, m["agents"]["seller_1"]["avg_price"], cb, "seller 1")
        ax.axhline(mu_budget, color="#A8DADC", lw=1.5, ls="--", zorder=5,
                   label=f"Avg buyer budget (μ={mu_budget:.0f})")
        ax.axhline(unit_cost, color="#E9C46A", lw=1.5, ls=":",  zorder=5,
                   label=f"Unit cost (c={unit_cost:.0f})")
        styled_ax(ax, f"Average Price  —  {run_label}", ylabel="Price")
        ax.legend(loc="upper right")

    # rewards
    for ax, m, ts, ca, cb, run_label in [
        (ax_rc, comp, ts_c, C0, C1, "Competitive"),
        (ax_ro, coop, ts_o, O0, O1, "Cooperative"),
    ]:
        band_plot(ax, ts, m["agents"]["seller_0"]["avg_reward"], ca, "seller 0")
        band_plot(ax, ts, m["agents"]["seller_1"]["avg_reward"], cb, "seller 1")
        ax.axhline(0, color=FG_DIM, lw=0.8, ls=":", zorder=0)
        styled_ax(ax, f"Episode Reward  —  {run_label}", ylabel="Sum reward / episode")
        ax.legend(loc="lower right")

    # gap and correlation
    band_plot(ax_gap, ts_c, comp["price_gap"], C0, "Competitive")
    band_plot(ax_gap, ts_o, coop["price_gap"], O0, "Cooperative")
    ax_gap.axhline(0, color=FG_DIM, lw=0.8, ls=":", zorder=0)
    styled_ax(ax_gap,
              "|Price Gap|  p0 − p1",
              ylabel="|p0 − p1|")
    ax_gap.legend()
    ax_gap.text(0.02, 0.93, "v more symmetric",
                transform=ax_gap.transAxes, fontsize=8, color=FG_DIM,
                 style="italic", va="top")

    band_plot(ax_cor, ts_c, comp["price_correlation"], C0, "Competitive")
    band_plot(ax_cor, ts_o, coop["price_correlation"], O0, "Cooperative")
    ax_cor.axhline(0,  color=FG_DIM, lw=0.8, ls=":",  zorder=0)
    ax_cor.axhline( 1, color=FG_DIM, lw=0.6, ls=":",  zorder=0, alpha=0.5)
    ax_cor.axhline(-1, color=FG_DIM, lw=0.6, ls=":",  zorder=0)
    ax_cor.set_ylim(-1.25, 1.25)
    for y, lbl, col in [(0.85, "coordination (+1)", FG_DIM),
                         (-0.85, "anti-coordination (−1)", FG_DIM)]:
        ax_cor.text(ts_c[0], y, lbl, color=col, fontsize=7,
                     style="italic", va="center", alpha=0.7)
    styled_ax(ax_cor,
              "Price Correlation  rho(p0, p1)",
              ylabel="Pearson rho")
    ax_cor.legend()

    # sales and loyalty, seller_0 only
    a0 = "seller_0"
    band_plot(ax_sal, ts_c, comp["agents"][a0]["avg_sales"], C0, "Competitive")
    band_plot(ax_sal, ts_o, coop["agents"][a0]["avg_sales"], O0, "Cooperative")
    styled_ax(ax_sal,
              "Avg Sales / Step  (seller 0)",
              ylabel="Units sold")
    ax_sal.legend()

    band_plot(ax_loy, ts_c, comp["agents"][a0]["avg_loyalty"], C0, "Competitive")
    band_plot(ax_loy, ts_o, coop["agents"][a0]["avg_loyalty"], O0, "Cooperative")
    ax_loy.set_ylim(-0.05, 1.05)
    ax_loy.axhline(0.5, color=FG_DIM, lw=0.7, ls=":", alpha=0.5, zorder=0)
    styled_ax(ax_loy,
              "Avg Loyalty  (seller 0)",
              ylabel="Loyalty [0, 1]")
    ax_loy.legend()

    return fig


def build_dist_figure(comp, coop, unit_cost: float = 8.0, mu_budget: float = 25.0):
    # ridgeline-style KDE of the last 30% of iterations, shows where each run settled
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 1.01,
             "Final-Phase Distributions  (last 30 % of training)",
             ha="center",  fontsize=14,
             color=FG, fontweight="bold")
    fig.text(0.5, 0.95,
             "Kernel density estimate of iteration-average values  |  wider spread = higher variability",
             ha="center",  fontsize=9, color=FG_DIM, style="italic")

    metrics_cfg = [
        ("avg_price",   "Average Price",          "Price",       True),
        ("avg_reward",  "Episode Reward",          "Sum reward",  False),
        ("avg_sales",   "Sales / Step",            "Units",       False),
        ("avg_loyalty", "Loyalty",                 "[0, 1]",      False),
    ]

    def tail(lst, frac=0.30):
        n = max(int(len(lst) * frac), 3)
        return np.array(lst[-n:])

    for ax, (key, title, xlabel, show_refs) in zip(axes, metrics_cfg):
        ax.set_facecolor(BG_PANEL)

        for run, c0, c1, run_lbl in [
            (comp, C0, C1, "Competitive"),
            (coop, O0, O1, "Cooperative"),
        ]:
            for agent, color in [("seller_0", c0), ("seller_1", c1)]:
                vals = tail(run["agents"][agent][key])
                if vals.std() < 1e-9:
                    continue
                kde  = gaussian_kde(vals, bw_method="silverman")
                xs   = np.linspace(vals.min() - vals.std(),
                                   vals.max() + vals.std(), 300)
                ys   = kde(xs)

                label = f"{run_lbl} | {agent.replace('_', ' ')}"
                ax.fill_between(xs, ys, alpha=0.25, color=color)
                ax.plot(xs, ys, color=color, lw=1.8, label=label)
                ax.axvline(vals.mean(), color=color, lw=1.0,
                           ls="--", alpha=0.7)

        if show_refs:
            ax.axvline(mu_budget, color="#A8DADC", lw=1.8,
                       ls="--", label=f"Avg buyer budget (μ={mu_budget:.0f})", zorder=6)
            ax.axvline(unit_cost, color="#E9C46A", lw=1.8,
                       ls=":",  label=f"Unit cost (c={unit_cost:.0f})", zorder=6)

        ax.set_xlabel(xlabel,  color=FG_DIM)
        ax.set_ylabel("Density",  color=FG_DIM)
        ax.set_title(title, color=FG, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(FG_DIM)
        ax.spines["bottom"].set_color(FG_DIM)
        ax.tick_params(colors=FG_DIM)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right",
                  facecolor="#1C2128", edgecolor=FG_DIM, framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
