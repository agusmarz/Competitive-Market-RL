import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

from .envs import CompetitiveMarket
from .networks import ActorNN, CriticNN, RunningMeanStd


class SharedIPPO:
    # a single actor-critic shared by both sellers, since the game is symmetric
    # This is the setup for self-play

    def __init__(
        self,
        env: CompetitiveMarket,
        lr: float = 3e-4,
        timesteps_per_batch: int = 8_192,
        gamma: float = 0.999,
        n_updates: int = 10,
        clip: float = 0.2,
        entropy_coef_init: float = 0.01,
        entropy_coef_final: float = 0.0001,
        lam: float = 0.95,
    ):
        self.agent_ids = env.agent_ids
        a0             = self.agent_ids[0]
        obs_dim        = env.observation_space[a0].shape[0]
        act_dim        = env.action_space[a0].shape[0]

        self.actor  = ActorNN(obs_dim, act_dim)
        self.critic = CriticNN(obs_dim)

        self.actor_optim  = Adam(self.actor.parameters(),  lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr * 0.5, eps=1e-5)

        self.timesteps_per_batch = timesteps_per_batch
        self.gamma               = gamma
        self.n_updates           = n_updates
        self.clip                = clip

        self.entropy_coef_init  = entropy_coef_init
        self.entropy_coef_final = entropy_coef_final

        self.max_grad_norm = 0.5 # standard value in the PPO literature
        self.lr            = lr
        self.lam            = lam
        self.target_kl = 0.01 # standard early-stop threshold

        # running stats used to normalize obs and rewards
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.ret_rms = RunningMeanStd(shape=(1,))
        self.returns = {a: 0.0 for a in self.agent_ids}

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        mu, std  = self.actor(obs)
        dist     = Normal(mu, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum() # scalar
        return action.numpy(), log_prob

    def _gae_episode(
        self,
        rews:     list[float],
        vals:     list[float],   # V(s_0) ... V(s_{T-1}), from the critic
        last_val: float,         # V(s_T): 0 if terminal, critic(s_T) if truncated
    ) -> tuple[np.ndarray, np.ndarray]:
        # generalized advantage estimation, walked backwards through the episode
        T    = len(rews)
        advs = np.zeros(T, dtype=np.float32)
        gae  = 0.0

        for t in reversed(range(T)):
            next_v = vals[t + 1] if t + 1 < T else last_val # bootstrap at the edge
            delta  = rews[t] + self.gamma * next_v - vals[t] # td residual
            gae    = delta + self.gamma * self.lam * gae
            advs[t] = gae

        vtargs = advs + np.array(vals, dtype=np.float32) # used for the critic's MSE loss
        return advs, vtargs

    def evaluate(
        self,
        obs_t: torch.Tensor,
        acts_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        V         = self.critic(obs_t).squeeze(-1)
        mu, std   = self.actor(obs_t)
        log_probs = Normal(mu, std).log_prob(acts_t).sum(dim=-1)
        return V, log_probs

    def rollout(self, env: CompetitiveMarket):
        all_obs, all_acts, all_log_probs = [], [], []
        all_advs, all_vtargs             = [], []
        ep_metrics: list[dict]           = []
        total_agent_steps = 0

        while total_agent_steps < self.timesteps_per_batch:
            raw_obs_dict, _ = env.reset()

            # normalize the initial observation
            obs_dict = {}
            for a in self.agent_ids:
                self.obs_rms.update(raw_obs_dict[a][np.newaxis, :])
                obs_dict[a] = np.clip((raw_obs_dict[a] - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)

            done = False
            buf = {
                a: dict(obs=[], acts=[], log_probs=[],
                        rews=[], raw_rews=[], vals=[],
                        prices=[], sales=[], loyalties=[])
                for a in self.agent_ids
            }

            while not done:
                actions_for_env: dict[str, np.ndarray] = {}

                for a in self.agent_ids:
                    act, lp = self.get_action(obs_dict[a])
                    with torch.no_grad():
                        v = self.critic(
                            torch.tensor(obs_dict[a], dtype=torch.float32)
                        ).item()

                    # clip the action for the env, but keep the unclipped one in the buffer
                    actions_for_env[a] = np.clip(act, env.action_space[a].low, env.action_space[a].high)

                    buf[a]["obs"].append(obs_dict[a])
                    buf[a]["acts"].append(act)
                    buf[a]["log_probs"].append(lp)
                    buf[a]["vals"].append(v)

                raw_next_obs_dict, raw_rew_dict, term, trunc, info_dict = env.step(actions_for_env)
                done = term or trunc

                next_obs_dict = {}
                for a in self.agent_ids:
                    # scale and clip the reward
                    self.returns[a] = self.returns[a] * self.gamma + raw_rew_dict[a]
                    self.ret_rms.update(np.array([[self.returns[a]]]))
                    norm_rew = np.clip(raw_rew_dict[a] / np.sqrt(self.ret_rms.var[0] + 1e-8), -10.0, 10.0)

                    buf[a]["rews"].append(norm_rew)
                    buf[a]["raw_rews"].append(raw_rew_dict[a]) # kept for human-readable metrics

                    # normalize and clip the next observation
                    self.obs_rms.update(raw_next_obs_dict[a][np.newaxis, :])
                    next_obs_dict[a] = np.clip((raw_next_obs_dict[a] - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)

                    buf[a]["prices"].append(info_dict[a]["price"])
                    buf[a]["sales"].append(info_dict[a]["sales"])
                    buf[a]["loyalties"].append(info_dict[a]["loyalty"])

                obs_dict = next_obs_dict
                total_agent_steps += len(self.agent_ids)

            # reset the return tracker at each episode boundary
            for a in self.agent_ids:
                self.returns[a] = 0.0
                if trunc:
                    # time-limit cutoff, so bootstrap the value from the final obs
                    with torch.no_grad():
                        last_val = self.critic(
                            torch.tensor(obs_dict[a], dtype=torch.float32)
                        ).item()
                else:
                    last_val = 0.0 # natural terminal state

                advs, vtargs = self._gae_episode(
                    buf[a]["rews"],
                    buf[a]["vals"],
                    last_val,
                )
                all_obs.extend(buf[a]["obs"])
                all_acts.extend(buf[a]["acts"])
                all_log_probs.extend(buf[a]["log_probs"])
                all_advs.extend(advs.tolist())
                all_vtargs.extend(vtargs.tolist())

            ep_metrics.append({
                a: {
                    "prices":    buf[a]["prices"],
                    "sales":     buf[a]["sales"],
                    "loyalties": buf[a]["loyalties"],
                    "total_rew": float(np.sum(buf[a]["raw_rews"])),
                }
                for a in self.agent_ids
            })

        obs_t    = torch.tensor(np.array(all_obs),   dtype=torch.float32)
        acts_t   = torch.tensor(np.array(all_acts),  dtype=torch.float32)
        lp_t     = torch.stack(all_log_probs)
        advs_t   = torch.tensor(all_advs,   dtype=torch.float32)
        vtargs_t = torch.tensor(all_vtargs, dtype=torch.float32)

        return obs_t, acts_t, lp_t, advs_t, vtargs_t, ep_metrics

    def update(self, obs_t, acts_t, old_lp_t, advs_t, vtargs_t,
              entropy_coef=0.01, minibatch_size=256):

        dataset_size = obs_t.shape[0]
        indices = np.arange(dataset_size)

        a_losses, c_losses = [], []
        kl_exceeded = False

        for _ in range(self.n_updates):
            np.random.shuffle(indices) # reshuffle every epoch

            if kl_exceeded:
                break
            for start in range(0, dataset_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                mb_obs    = obs_t[mb_idx]
                mb_acts   = acts_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]
                mb_vtargs = vtargs_t[mb_idx]
                mb_advs   = advs_t[mb_idx]

                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8) # normalize inside the minibatch

                V, curr_lp = self.evaluate(mb_obs, mb_acts)
                with torch.no_grad():
                    log_ratio = curr_lp - mb_old_lp
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item() # Schulman's k2 estimator

                if approx_kl > self.target_kl:
                    kl_exceeded = True
                    break # stop this epoch early

                r = torch.exp(curr_lp - mb_old_lp)

                ppo_loss = -torch.min(
                    r * mb_advs,
                    torch.clamp(r, 1 - self.clip, 1 + self.clip) * mb_advs,
                ).mean()

                mu, std = self.actor(mb_obs)
                entropy = Normal(mu, std).entropy().sum(dim=-1).mean()
                a_loss  = ppo_loss - entropy_coef * entropy

                self.actor_optim.zero_grad()
                a_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                c_loss = nn.MSELoss()(V, mb_vtargs) * 0.5
                self.critic_optim.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                a_losses.append(float(a_loss.detach()))
                c_losses.append(float(c_loss.detach()))

        return float(np.mean(a_losses)), float(np.mean(c_losses))

    def learn(self, env: CompetitiveMarket, total_timesteps: int) -> dict:
        # trains until total_timesteps agent-steps have been collected
        # price_correlation > 0 means coordination, < 0 means a price war
        t_so_far = 0
        a0, a1   = self.agent_ids[0], self.agent_ids[1]

        metrics: dict = {
            "timesteps":         [],
            "agents":            {a: dict(avg_price=[], avg_sales=[],
                                          avg_loyalty=[], avg_reward=[])
                                  for a in self.agent_ids},
            "actor_loss":        [],
            "critic_loss":       [],
            "price_gap":         [],
            "price_correlation": [],
        }

        while t_so_far < total_timesteps:
            obs_t, acts_t, lp_t, advs_t, vtargs_t, ep_metrics = self.rollout(env)
            t_so_far += obs_t.shape[0]

            # linear LR decay
            frac   = t_so_far / total_timesteps
            new_lr = max(self.lr * (1.0 - frac), 1e-6)
            for opt in (self.actor_optim, self.critic_optim):
                for pg in opt.param_groups:
                    pg["lr"] = new_lr

            # entropy annealing, more exploration early on
            entropy_coef = self.entropy_coef_init + (
                self.entropy_coef_final - self.entropy_coef_init
            ) * frac

            a_loss, c_loss = self.update(
                obs_t, acts_t, lp_t, advs_t, vtargs_t,
                entropy_coef=entropy_coef,
            )
            # per-agent metrics
            for a in self.agent_ids:
                metrics["agents"][a]["avg_price"].append(
                    float(np.mean([np.mean(ep[a]["prices"])    for ep in ep_metrics])))
                metrics["agents"][a]["avg_sales"].append(
                    float(np.mean([np.mean(ep[a]["sales"])     for ep in ep_metrics])))
                metrics["agents"][a]["avg_loyalty"].append(
                    float(np.mean([np.mean(ep[a]["loyalties"]) for ep in ep_metrics])))
                metrics["agents"][a]["avg_reward"].append(
                    float(np.mean([ep[a]["total_rew"]          for ep in ep_metrics])))

            p0 = metrics["agents"][a0]["avg_price"][-1]
            p1 = metrics["agents"][a1]["avg_price"][-1]

            # price correlation within the batch, a coordination signal
            p0s  = [np.mean(ep[a0]["prices"]) for ep in ep_metrics]
            p1s  = [np.mean(ep[a1]["prices"]) for ep in ep_metrics]
            corr = float(np.corrcoef(p0s, p1s)[0, 1]) if len(p0s) > 1 else 0.0

            metrics["timesteps"].append(t_so_far)
            metrics["actor_loss"].append(a_loss)
            metrics["critic_loss"].append(c_loss)
            metrics["price_gap"].append(float(abs(p0 - p1)))
            metrics["price_correlation"].append(corr)

            print(
                f"[{t_so_far:>8}]  "
                f"p0={p0:.2f}  p1={p1:.2f}  "
                f"gap={abs(p0-p1):.3f}  corr={corr:+.3f}  "
                f"a_loss={a_loss:.4f}  c_loss={c_loss:.4f}"
            )

        return metrics
