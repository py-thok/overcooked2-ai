"""Policy population for FCP-style training.

A population is a directory of named SB3 PPO checkpoints plus a metadata
file (population.json) tracking provenance and evaluation scores.

    population/
      population.json
      sp_seed0_2M.zip
      sp_seed1_2M.zip
      bc_human.zip
      ...
"""
import json, os, time
import numpy as np
import torch as th


class PolicyAgent:
    """CPU-runnable frozen policy snapshot, picklable for SubprocVecEnv."""

    def __init__(self, state_dict=None, obs_dim=None, n_actions=6,
                 net_arch=(256, 256), deterministic=False, name="anon"):
        self.n_actions = n_actions
        self.net_arch = tuple(net_arch)
        self.deterministic = deterministic
        self.name = name
        self._state_dict = state_dict
        self._obs_dim = obs_dim
        self.policy = None
        if state_dict is not None and obs_dim is not None:
            self._build()

    def _build(self):
        from stable_baselines3.common.policies import ActorCriticPolicy
        from gymnasium import spaces
        self.policy = ActorCriticPolicy(
            observation_space=spaces.Box(-np.inf, np.inf,
                                         shape=(self._obs_dim,), dtype=np.float32),
            action_space=spaces.Discrete(self.n_actions),
            lr_schedule=lambda _: 3e-4,
            net_arch=list(self.net_arch),
        )
        self.policy.load_state_dict(self._state_dict)
        self.policy.eval()

    @classmethod
    def from_sb3_model(cls, model_or_path, name="anon", deterministic=False):
        from stable_baselines3 import PPO
        model = model_or_path
        if isinstance(model_or_path, (str, os.PathLike)):
            model = PPO.load(model_or_path, device="cpu")
        sd = {k: v.detach().cpu().clone()
              for k, v in model.policy.state_dict().items()}
        obs_dim = model.observation_space.shape[0]
        net_arch = tuple(model.policy.net_arch) if hasattr(model.policy, "net_arch") else (256, 256)
        return cls(sd, obs_dim, model.action_space.n, net_arch,
                   deterministic, name)

    def __call__(self, obs):
        if self.policy is None:
            self._build()
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            if self.deterministic:
                return int(self.policy.predict(obs_t, deterministic=True)[0][0])
            return int(self.policy.get_distribution(obs_t).sample().item())

    def __getstate__(self):
        return {"state_dict": self._state_dict, "obs_dim": self._obs_dim,
                "n_actions": self.n_actions, "net_arch": self.net_arch,
                "deterministic": self.deterministic, "name": self.name}

    def __setstate__(self, s):
        self.__init__(s["state_dict"], s["obs_dim"], s["n_actions"],
                      s["net_arch"], s["deterministic"], s["name"])


class RandomAgent:
    def __init__(self, n_actions=6, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.name = "random"

    def __call__(self, obs):
        return int(self.rng.integers(self.n_actions))


class Population:
    def __init__(self, root="population"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.meta_path = os.path.join(root, "population.json")
        if os.path.exists(self.meta_path):
            self.meta = json.load(open(self.meta_path))
        else:
            self.meta = {"members": []}

    def save_meta(self):
        json.dump(self.meta, open(self.meta_path, "w"), indent=2)

    def add(self, model_or_path, name, kind="sp", notes=""):
        """Save a model into the population. kind in {sp, bc, fcp, random}."""
        from stable_baselines3 import PPO
        dst = os.path.join(self.root, f"{name}.zip")
        if isinstance(model_or_path, (str, os.PathLike)):
            import shutil
            shutil.copy(model_or_path, dst)
        else:
            model_or_path.save(dst)
        entry = {"name": name, "kind": kind, "path": dst,
                 "notes": notes, "added": time.time(),
                 "elo": 1500.0}
        self.meta["members"] = [m for m in self.meta["members"]
                                if m["name"] != name] + [entry]
        self.save_meta()
        return entry

    def names(self):
        return [m["name"] for m in self.meta["members"]]

    def sample_agent(self, weights=None, deterministic=False, exclude=(),
                     seed=None):
        """Sample a population member as a PolicyAgent (or RandomAgent)."""
        members = [m for m in self.meta["members"] if m["name"] not in exclude]
        if not members:
            return RandomAgent()
        rng = np.random.default_rng(seed)
        w = np.ones(len(members)) if weights is None else np.asarray(
            [weights.get(m["name"], 1.0) for m in members], dtype=float)
        w = w / w.sum()
        m = members[rng.choice(len(members), p=w)]
        if m["kind"] == "random":
            return RandomAgent()
        return PolicyAgent.from_sb3_model(m["path"], name=m["name"],
                                          deterministic=deterministic)

    def load_agent(self, name, deterministic=False):
        m = next(m for m in self.meta["members"] if m["name"] == name)
        if m["kind"] == "random":
            return RandomAgent()
        return PolicyAgent.from_sb3_model(m["path"], name=name,
                                          deterministic=deterministic)

    def update_elo(self, name, delta):
        for m in self.meta["members"]:
            if m["name"] == name:
                m["elo"] += delta
        self.save_meta()
