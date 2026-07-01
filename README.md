# Competitive market environment for Reinforcement Learning

Multi-agent PPO trained on a two-seller pricing environment,
comparing a competitive scenario against a cooperative one. 

## Structure

```
market-rl/
├── src/market_rl/
│   ├── envs.py        # CompetitiveMarket + CooperativeMarket (gymnasium.Env)
│   ├── networks.py     # ActorNN, CriticNN, RunningMeanStd
│   ├── ppo.py           # SharedIPPO: rollout, GAE, PPO update
│   ├── train.py         # run_experiment(): trains one run and saves metrics
│   ├── viz_light.py    # matplotlib figures, light theme
│   └── viz_dark.py      # matplotlib figures, dark theme + uncertainty bands
├── scripts/
│   ├── run_experiment.py   # trains both the competitive and cooperative runs
│   └── make_figures.py     # builds figures from the saved metrics
├── results/             # metrics_*.json and *.png land here (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

```bash
python scripts/run_experiment.py   # trains, writes results/metrics_*.json
python scripts/make_figures.py     # builds results/*.png from those metrics
```

## Notes

- `viz_light.py` and `viz_dark.py` are two independent visualization styles
  for the same metrics — pick one, or keep both like `make_figures.py` does.

