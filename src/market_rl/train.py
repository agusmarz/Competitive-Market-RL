import json
from pathlib import Path

import numpy as np

from .envs import CompetitiveMarket
from .ppo import SharedIPPO


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(env_cls, label: str, seed: int = 42, total_timesteps: int = 250_000) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    np.random.seed(seed)
    env = env_cls()

    agent = SharedIPPO(
        env,
        lr=3e-4,
        timesteps_per_batch=8_192,
        gamma=0.999,
        n_updates=10,
        clip=0.2,
        entropy_coef_init=0.01,
        entropy_coef_final=0.0001,
        lam=0.95,
    )

    metrics = agent.learn(env, total_timesteps=total_timesteps)
    metrics["label"] = label

    # dump raw metrics so the viz functions can load them later
    out_path = RESULTS_DIR / f"metrics_{label.lower().replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {out_path}")

    return metrics
