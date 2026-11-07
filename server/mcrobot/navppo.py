"""PPO for the navigation policy on NavEnv: actor and critic MLPs, GAE with time-limit bootstrapping, atomic checkpoints
(latest: resumable; best: policy only; final: policy only). Simulation only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from .navenv import OBS_DIM


def mlp(i: int, o: int, hidden=(256, 256, 128)) -> nn.Sequential:
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = 3, init_std: float = 0.5):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        mu = self.actor(obs)
        if deterministic:
            return mu.clamp(-1, 1)
        return (mu + torch.randn_like(mu) * self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def save_atomic(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_policy(path, device="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    ac = ActorCritic().to(device)
    ac.load_state_dict(ck["model"])
    return ac


class Yield:
    """gpu-run asks a preemptible job to save and exit: SIGUSR1 and/or the yield file; SIGTERM (a /robot stop) also saves and exits."""

    def __init__(self):
        import signal

        self.flag = False
        self.file = os.environ.get("GPU_RUN_YIELD_FILE", "")
        for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *_: setattr(self, "flag", True))

    def requested(self) -> bool:
        return self.flag or bool(self.file and os.path.exists(self.file))


class Trainer:
    def __init__(self, env, out: Path, horizon: int = 32, lr: float = 3e-4, gamma: float = 0.995, lam: float = 0.95, epochs: int = 4,
                 minibatches: int = 8, clip: float = 0.2, ent: float = 0.002, reward_scale: float = 0.1, hook=None, log=print):
        self.env, self.out, self.dev = env, Path(out), env.dev
        self.out.mkdir(parents=True, exist_ok=True)
        self.T, self.gamma, self.lam, self.epochs, self.mb, self.clip, self.ent, self.rs = horizon, gamma, lam, epochs, minibatches, clip, ent, reward_scale
        self.ac = ActorCritic().to(self.dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.it, self.steps, self.best, self.hook, self.log = 0, 0, -1e9, hook, log
        self.hist = {"spawn_ok": [], "spawn_n": 0}
        self.obs = env.obs()
        self.resume()

    # ---- checkpoints ----------------------------------------------------------------------------------------------------------------
    def resume(self):
        p = self.out / "latest.pt"
        if p.exists():
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            self.ac.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.it, self.steps, self.best = ck["it"], ck["steps"], ck["best"]
            self.log(f"resumed from {p} at iteration {self.it}")

    def save_latest(self):
        save_atomic({"model": self.ac.state_dict(), "opt": self.opt.state_dict(), "it": self.it, "steps": self.steps, "best": self.best}, self.out / "latest.pt")

    def save_policy(self, name: str, extra=None):
        save_atomic({"model": self.ac.state_dict(), "it": self.it, "steps": self.steps, **(extra or {})}, self.out / name)

    # ---- one iteration --------------------------------------------------------------------------------------------------------------
    def rollout(self):
        env, T, n = self.env, self.T, self.env.n
        obs_b = torch.zeros(T, n, OBS_DIM, device=self.dev)
        act_b = torch.zeros(T, n, 3, device=self.dev)
        logp_b, val_b, rew_b, done_b = (torch.zeros(T, n, device=self.dev) for _ in range(4))
        stats = {"ep": 0, "success": 0, "lava": 0, "fell": 0, "timeout": 0, "ret": 0.0, "len": 0.0, "sp_ep": 0, "sp_ok": 0, "sp_time": 0.0, "plates": 0.0}
        obs = self.obs
        for t in range(T):
            with torch.no_grad():
                d = self.ac.dist(obs)
                a = d.sample()
                logp = d.log_prob(a).sum(-1)
                v = self.ac.value(obs)
            obs2, r, done, info = env.step(a)
            r = r * self.rs
            if done.any():
                trunc = info["timeout"]
                if trunc.any():
                    with torch.no_grad():
                        vf = self.ac.value(info["final_obs"])
                    r = r + self.gamma * vf * trunc
                m = done
                k = int(m.sum())
                stats["ep"] += k
                for key in ("success", "lava", "fell", "timeout"):
                    stats[key] += int(info[key][m].sum())
                stats["ret"] += float(info["ep_return"][m].sum())
                stats["len"] += float(info["ep_len"][m].sum())
                stats["plates"] += float(info["plates"][m].sum())
                sp = m & info["from_spawn"]
                stats["sp_ep"] += int(sp.sum())
                ok = sp & info["success"]
                stats["sp_ok"] += int(ok.sum())
                stats["sp_time"] += float(info["ep_len"][ok].sum()) * 0.1
            obs_b[t], act_b[t], logp_b[t], val_b[t], rew_b[t], done_b[t] = obs, a, logp, v, r, done.float()
            obs = obs2
            if self.hook:
                self.hook(env, self)
        self.obs = obs
        with torch.no_grad():
            last_v = self.ac.value(obs)
        adv = torch.zeros_like(rew_b)
        g = torch.zeros(n, device=self.dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + self.gamma * nv * nonterm - val_b[t]
            g = delta + self.gamma * self.lam * nonterm * g
            adv[t] = g
        return obs_b, act_b, logp_b, adv, adv + val_b, stats

    def update(self, obs_b, act_b, logp_b, adv, ret):
        T, n = obs_b.shape[:2]
        o, a, lp, ad, rt = (x.reshape(T * n, *x.shape[2:]) for x in (obs_b, act_b, logp_b, adv, ret))
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        N = o.shape[0]
        info = {}
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.dev)
            for chunk in perm.chunk(self.mb):
                d = self.ac.dist(o[chunk])
                nlp = d.log_prob(a[chunk]).sum(-1)
                ratio = (nlp - lp[chunk]).exp()
                s1, s2 = ratio * ad[chunk], ratio.clamp(1 - self.clip, 1 + self.clip) * ad[chunk]
                pl = -torch.minimum(s1, s2).mean()
                vl = 0.5 * (self.ac.value(o[chunk]) - rt[chunk]).pow(2).mean()
                loss = pl + 0.5 * vl - self.ent * d.entropy().sum(-1).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
            info = {"pl": float(pl), "vl": float(vl), "std": float(self.ac.log_std.detach().exp().mean())}
        return info

    def train(self, max_iters: int, max_seconds: float, y: Yield, ckpt_every_s: float = 300.0, log_path: Path | None = None) -> str:
        """Returns why it stopped: 'yield', 'done'."""
        t_start, t_ck = time.time(), time.time()
        log_f = open(log_path or self.out / "log.jsonl", "a")
        while self.it < max_iters and time.time() - t_start < max_seconds:
            if y.requested():
                self.save_latest()
                return "yield"
            t0 = time.time()
            obs_b, act_b, logp_b, adv, ret, st = self.rollout()
            t1 = time.time()
            up = self.update(obs_b, act_b, logp_b, adv, ret)
            self.it += 1
            self.steps += self.T * self.env.n
            ep = max(st["ep"], 1)
            rec = {"it": self.it, "steps": self.steps, "t": round(time.time() - t_start, 1), "rollout_s": round(t1 - t0, 2), "update_s": round(time.time() - t1, 2),
                   "env_steps_per_s": round(self.T * self.env.n / (t1 - t0), 1), "episodes": st["ep"], "success": st["success"] / ep, "lava": st["lava"] / ep,
                   "fell": st["fell"] / ep, "timeout": st["timeout"] / ep, "ret": st["ret"] / ep, "ep_len": st["len"] / ep, "plates": st["plates"] / ep,
                   "spawn_episodes": st["sp_ep"], "spawn_success": st["sp_ok"] / max(st["sp_ep"], 1), "spawn_time_s": st["sp_time"] / max(st["sp_ok"], 1), **up}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            self.log(f"it {self.it} steps {self.steps} fps {rec['env_steps_per_s']:.0f} succ {rec['success']:.2f} spawn {rec['spawn_success']:.2f} "
                     f"({st['sp_ep']}) lava {rec['lava']:.2f} fell {rec['fell']:.2f} ret {rec['ret']:.2f} std {up['std']:.2f}")
            self.hist["spawn_n"] += st["sp_ep"]
            if st["sp_ep"] >= 8:  # best = most spawn-start successes per episode, over a window with enough episodes
                score = rec["spawn_success"] + 0.01 * rec["success"] - 0.0001 * rec["spawn_time_s"]
                if score > self.best:
                    self.best = score
                    self.save_policy("best.pt", {"score": score})
            if time.time() - t_ck > ckpt_every_s:
                self.save_latest()
                t_ck = time.time()
        self.save_latest()
        return "done"
