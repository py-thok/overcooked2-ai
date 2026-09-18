"""Seed a population: train N independent self-play agents (different seeds)
and register them (plus a random agent) into a population directory.

Runs sequentially by default to avoid checkpoint name collisions and GPU
contention; use --parallel only if you have spare GPUs.

Usage:
    python seed_population.py --layout cramped_room --n-seeds 4 --timesteps 3000000
"""
import argparse, os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--population", default=None)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--parallel", action="store_true")
    args = p.parse_args()

    pop_dir = args.population or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "population", args.layout)
    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(here, "checkpoints", "pop_seeds")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(here, "logs"), exist_ok=True)

    procs = []
    for s in range(args.n_seeds):
        seed = s * 1000
        log_path = os.path.join(here, "logs", f"seed_{args.layout}_{s}.log")
        cmd = [py, os.path.join(here, "train_sp.py"),
               "--layout", args.layout,
               "--timesteps", str(args.timesteps),
               "--n-envs", str(args.n_envs),
               "--seed", str(seed),
               "--device", args.device,
               "--save-dir", ckpt_dir]
        print(f"[seed {s}] {' '.join(cmd)} -> {log_path}")
        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((s, proc, log))
        if not args.parallel:
            proc.wait()
            log.close()

    if args.parallel:
        for s, proc, log in procs:
            proc.wait()
            log.close()

    from population import Population
    pop = Population(pop_dir)
    for s in range(args.n_seeds):
        ckpt = os.path.join(
            ckpt_dir, f"ppo_sp_{args.layout}_seed{s*1000}_{args.timesteps}.zip")
        if os.path.exists(ckpt):
            pop.add(ckpt, f"sp_seed{s}", kind="sp",
                    notes=f"self-play seed {s*1000}, {args.timesteps} steps")
        else:
            print(f"WARN missing {ckpt}")
    if not any(m["kind"] == "random" for m in pop.meta["members"]):
        pop.meta["members"].append({"name": "random", "kind": "random",
                                    "path": "", "notes": "", "added": time.time(),
                                    "elo": 1500})
    pop.save_meta()
    print(f"population at {pop_dir}: {pop.names()}")


if __name__ == "__main__":
    main()
