"""Gymnasium wrapper around the Overcooked-AI simulator (single-agent view).

One chef is controlled by the RL policy; the partner policy is injected
externally (self-play: same network; later: population sampling).

Observation: lossless state encoding (featurize_state_ppo), shape ~96-dim.
Action: discrete 6 (N/S/E/W/stay/interact) — both agents act jointly per step.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as OAIEnv
from overcooked_ai_py.agents.agent import AgentFromPolicy

N_ACTIONS = len(Action.ALL_ACTIONS)  # 6


class OvercookedSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layout_name="cramped_room", horizon=400,
                 rew_shaping_horizon=0, partner_policy=None,
                 agent_index=0, seed=None):
        super().__init__()
        self.layout_name = layout_name
        self.horizon = horizon
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OAIEnv.from_mdp(self.mdp, horizon=horizon,
                                        info_level=0)
        self.rew_shaping_horizon = rew_shaping_horizon
        self.partner_policy = partner_policy  # callable(obs_partner) -> action idx
        self.agent_index = agent_index
        self.action_space = spaces.Discrete(N_ACTIONS)
        dummy_state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(dummy_state)[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=obs.shape, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0
        self._ep_ret = 0.0
        self._ep_sparse = 0.0

    def _obs(self, state):
        return self.base_env.featurize_state_mdp(state)[self.agent_index].astype(np.float32)

    def set_partner_policy(self, policy):
        """policy: fn(obs) -> action index, or None for random."""
        self.partner_policy = policy

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()
        self.state = self.base_env.state
        self.t = 0
        self._ep_ret = 0.0
        self._ep_sparse = 0.0
        return self._obs(self.state), {}

    def step(self, action):
        partner_idx = 1 - self.agent_index
        if self.partner_policy is None:
            partner_action = self._rng.integers(N_ACTIONS)
        else:
            partner_obs = self.base_env.featurize_state_mdp(self.state)[partner_idx].astype(np.float32)
            partner_action = int(self.partner_policy(partner_obs))

        joint_action = [0, 0]
        joint_action[self.agent_index] = int(action)
        joint_action[partner_idx] = partner_action
        joint_action = tuple(Action.INDEX_TO_ACTION[a] for a in joint_action)

        next_state, reward, done, info = self.base_env.step(
            joint_action, display_phi=False)
        self.state = next_state
        self.t += 1

        sparse = reward
        shaped = info.get("shaped_r_by_agent", [0, 0])[self.agent_index] \
            if isinstance(info, dict) else 0
        rew = sparse + shaped

        truncated = (not done) and self.t >= self.horizon
        self._ep_ret += rew
        self._ep_sparse += sparse
        info = {"sparse_reward": sparse, "shaped_reward": shaped}
        if done or truncated:
            # SB3 convention: Monitor/VecEnv logs ep_rew_mean from this
            info["episode"] = {"r": self._ep_ret, "l": self.t,
                               "sparse": self._ep_sparse}
        return self._obs(self.state), rew, done, truncated, info
