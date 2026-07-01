import json
from pathlib import Path

from market_rl.train import RESULTS_DIR
from market_rl.viz_light import build_figure, build_summary_table
from market_rl.viz_dark import build_main_figure, build_dist_figure, load_or_synthesise, BG

if __name__ == "__main__":
    comp_path = RESULTS_DIR / "metrics_competitive.json"
    coop_path = RESULTS_DIR / "metrics_cooperative.json"

    with open(comp_path) as f: comp = json.load(f)
    with open(coop_path) as f: coop = json.load(f)

    print("Building light-theme figures …")
    build_figure(comp, coop).savefig(
        RESULTS_DIR / "experiment_comparison.png", dpi=150, bbox_inches="tight")
    build_summary_table(comp, coop).savefig(
        RESULTS_DIR / "summary_table.png", dpi=150, bbox_inches="tight")

    print("Building dark-theme figures …")
    comp_dark, coop_dark = load_or_synthesise()  # falls back to synthetic data if no JSON found
    build_main_figure(comp_dark, coop_dark).savefig(
        RESULTS_DIR / "viz_main.png", dpi=150, bbox_inches="tight", facecolor=BG)
    build_dist_figure(comp_dark, coop_dark).savefig(
        RESULTS_DIR / "viz_distributions.png", dpi=150, bbox_inches="tight", facecolor=BG)

    print("Done. Figures saved under", RESULTS_DIR)
