from typing import Optional

import numpy as np
from scipy.stats import norm
import gymnasium as gym
from gymnasium.envs.registration import register


truncation_window = 1000
WINDOW = 16


class CompetitiveMarket(gym.Env):

    def __init__(self, initial_price: int = 10, n_potential_buyers: int = 10, max_price : int = 50):
        
        self.initial_price = initial_price # we save this for the reset()
        self.window = WINDOW # number of past states that agents will see

        # agents in the environment. We will always use 2 agents
        n_agents = 2
        self.agent_ids = [f"seller_{i}" for i in range(n_agents)]
        self.n_agents = len(self.agent_ids)

        self.n_potential_buyers = n_potential_buyers # number of buyers inside the environment
        self.max_price = max_price # max price allowed by the environment, this could be +infinite

        #the dimensionality of the observation space:
        obs_dim = self.n_agents * 2 + self.n_agents * self.window * 3 + 1

        self.mu_budget = 25.0   # mean buyer budget
        self.sigma     = 7.0   # std budget
        self.beta      = 0.5    # sensibility to price differences
        self.gamma     = 0.5    # weight of loyalty in the demand model

        self.eta           = 0.05   # loyalty gain rate
        self.delta_loyalty = 0.10   # loyalty decay rate 

        self.unit_cost = 8.0  # marginal cost of the product sold by the agent

        # Now we define the spaces for observations and actions
        single_obs = gym.spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        single_act = gym.spaces.Box(low=-5.0, high=5.0,   shape=(1,),       dtype=np.float32)

        self.observation_space = gym.spaces.Dict({a: single_obs for a in self.agent_ids})
        self.action_space      = gym.spaces.Dict({a: single_act for a in self.agent_ids})


        # Initialise state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.n_steps   = 0
        self.prices    = {a: float(self.initial_price)          for a in self.agent_ids}
        self.loyalties = {a: self.np_random.uniform(0.0, 1.0)   for a in self.agent_ids}
        self.sales     = {a: 0                                   for a in self.agent_ids}

        self.price_history   = {a: [0.0] * self.window for a in self.agent_ids}
        self.sales_history   = {a: [0.0] * self.window for a in self.agent_ids}
        self.revenue_history = {a: [0.0] * self.window for a in self.agent_ids}

        observations = {a: self._get_obs(a)  for a in self.agent_ids}
        infos        = {a: self._get_info(a) for a in self.agent_ids}

        return observations, infos

    def compute_demand(self) -> tuple[dict, dict]:

        prices_list = [self.prices[a] for a in self.agent_ids]
        p_floor = min(prices_list)

        # Stage 1, fraction of buyers that will buy
        alpha = norm.sf(p_floor, loc=self.mu_budget, scale=self.sigma)

        # Stage 2, assign buyers to sellers
        weights = {
            a: (1.0 + self.gamma * self.loyalties[a])
               * np.exp(-self.beta * max(self.prices[a] - p_floor, 0.0))
            for a in self.agent_ids
        }
        W = sum(weights.values())

        demand_fractions = {a: alpha * weights[a] / W for a in self.agent_ids}

        sales = {
            a: int(self.np_random.binomial(
                n=self.n_potential_buyers,
                p=np.clip(demand_fractions[a], 0.0, 1.0)
            ))
            for a in self.agent_ids
        }

        return sales, demand_fractions


    def _get_obs(self, agent: str) -> np.ndarray:
        # Each agent sees itself first
        ordered = [agent] + [a for a in self.agent_ids if a != agent]

        # for each agent, we add the price and sales (both normalized)
        current = []
        for a in ordered:
            current.extend([
                self.prices[a] / self.max_price,
                self.sales[a]  / self.n_potential_buyers,
            ])

        # and we add the history of each variable
        history = []
        for a in ordered:
            for p, s, r in zip(
                self.price_history[a],
                self.sales_history[a],
                self.revenue_history[a],
            ):
                history.extend([
                    p / self.max_price,
                    s / self.n_potential_buyers,
                    r / (self.max_price * self.n_potential_buyers),
                ])

        return np.array([
            *current,                           # n_agents × 2
            *history,                           # n_agents × window × 3
            self.n_steps / truncation_window,   # 1
        ], dtype=np.float32)

    def _get_info(self, agent: str) -> dict:
        return {
            "price":   self.prices[agent],
            "sales":   self.sales[agent],
            "revenue": self.prices[agent] * self.sales[agent],
            "loyalty": self.loyalties[agent],
        }

    def step(self, actions: dict):

        self.n_steps += 1

        # Agents do actions
        for a in self.agent_ids:
            delta = float(actions[a][0])
            self.prices[a] = np.clip(self.prices[a] + delta, 0.0, self.max_price)

        # Environments gives reward
        self.sales, demand_fractions = self.compute_demand()
        # We normalize the reward
        max_val = self.get_max_possible_value()
        rewards = {
            a: (self.prices[a] - self.unit_cost) * self.sales[a] / max_val
            for a in self.agent_ids
        }

        # update latent variable (the loyalty)
        for a in self.agent_ids:
            df = demand_fractions[a]
            self.loyalties[a] = np.clip(
                self.loyalties[a]
                + self.eta          * df
                - self.delta_loyalty * (1.0 - df),
                0.0, 1.0,
            )

        # update the history
        for a in self.agent_ids:
            self.price_history[a]   = self.price_history[a][1:]   + [self.prices[a]]
            self.sales_history[a]   = self.sales_history[a][1:]   + [float(self.sales[a])]
            self.revenue_history[a] = self.revenue_history[a][1:] + [self.prices[a] * self.sales[a]]

        
        terminated = False # this environment is never terminated
        truncated  = self.n_steps >= truncation_window

        observations = {a: self._get_obs(a)  for a in self.agent_ids}
        infos        = {a: self._get_info(a) for a in self.agent_ids}

        return observations, rewards, terminated, truncated, infos
    def get_max_possible_value(self):
        return (self.max_price - self.unit_cost) * self.n_potential_buyers


class CooperativeMarket(CompetitiveMarket):
    # same env, but every agent gets the joint industry profit as reward
    # (the "social planner" scenario, agents maximize total profit instead of market share)

    def step(self, actions: dict):
        obs, rewards, terminated, truncated, infos = super().step(actions)

        joint_reward = sum(rewards.values())
        coop_rewards = {a: joint_reward for a in self.agent_ids}

        return obs, coop_rewards, terminated, truncated, infos


register(
    id="gymnasium_env/CompetitiveMarket-v0",
    entry_point="gymnasium_env.envs:CompetitiveMarket",
)
