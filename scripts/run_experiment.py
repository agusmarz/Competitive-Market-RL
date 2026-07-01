from market_rl import CompetitiveMarket, CooperativeMarket, run_experiment

TIMESTEPS = 2_000_000

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)

    run_experiment(CompetitiveMarket, "Competitive", total_timesteps=TIMESTEPS)
    run_experiment(CooperativeMarket, "Cooperative", total_timesteps=TIMESTEPS)
