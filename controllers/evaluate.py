r"""
Avaliacao do evader — preenche a TABELA IX do paper (T3 G9).

Metricas (por condicao):
  - Survival steps (mean +/- std)
  - Success rate (%)   -> chegou ao limite de tempo sem capture/collision/rollover
  - Capture rate (%)
  - Collision rate (%) -> obstacle_collision OU rollover (definicao do paper)
  - Mean pursuer distance (m)
  - Hidden ratio (%)   -> fracao de passos sem linha de visao perseguidor<->evader
  - Line-of-sight breaks (transicoes visivel->escondido por episodio)

CONDICOES (--condition):
  stopped   baseline: evader parado
  random    baseline: acoes aleatorias
  model     usa o checkpoint em --model

USO (raiz do projeto, Webots aberto + PLAY em fast):
  .\.venv\Scripts\python.exe .\controllers\evaluate.py --condition model --model "logs/checkpoints/MODELO" --algorithm auto --episodes 30 --limited-info-pursuer --enriched-random-obstacles --hide-reward-display --no-obstacle-safety
  .\.venv\Scripts\python.exe .\controllers\evaluate.py --condition stopped --episodes 30 --limited-info-pursuer --enriched-random-obstacles --hide-reward-display --no-obstacle-safety
  .\.venv\Scripts\python.exe .\controllers\evaluate.py --condition random --episodes 30 --limited-info-pursuer --enriched-random-obstacles --hide-reward-display --no-obstacle-safety

Paper aceita 20-30 episodios por condicao (100 se houver tempo).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import controllers.evader_env  # noqa: F401
from controllers.experiment_config import env_kwargs_from_config, load_experiment_config


def _load_model(path, algorithm):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    algos = {"ppo": PPO, "recurrent_ppo": RecurrentPPO}
    if algorithm != "auto":
        return algos[algorithm].load(path), algorithm
    errors = []
    for name, cls in algos.items():
        try:
            return cls.load(path), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Falha ao carregar checkpoint:\n" + "\n".join(errors))


def parse_args():
    p = argparse.ArgumentParser(description="Avaliacao do evader (Tabela IX).")
    p.add_argument("--condition", required=True, choices=["stopped", "random", "model"])
    p.add_argument("--model", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--algorithm", default="auto", choices=["auto", "ppo", "recurrent_ppo"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--robot-name", default=None)
    p.add_argument("--random-spawn", action="store_true")
    p.add_argument("--max-pursuer-distance", type=float, default=60.0)
    p.add_argument("--min-pursuer-distance", type=float, default=35.0)
    p.add_argument("--csv", default=None)
    p.add_argument("--limited-info-pursuer", action="store_true")
    p.add_argument("--random-obstacles", action="store_true")
    p.add_argument("--enriched-random-obstacles", action="store_true")
    p.add_argument("--center-spawn", action="store_true")
    p.add_argument("--hide-reward-display", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    obstacle_safety = p.add_mutually_exclusive_group()
    obstacle_safety.add_argument("--obstacle-safety", dest="obstacle_safety", action="store_true")
    obstacle_safety.add_argument("--no-obstacle-safety", dest="obstacle_safety", action="store_false")
    p.set_defaults(obstacle_safety=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.condition == "model" and not args.model:
        raise SystemExit("--condition model exige --model.")

    config = load_experiment_config(args.config)
    env_kwargs = env_kwargs_from_config(config)
    env_kwargs.update(
        robot_name=args.robot_name,
        show_reward_display=not args.hide_reward_display,
    )
    if args.obstacle_safety is not None:
        env_kwargs["obstacle_safety_enabled"] = args.obstacle_safety
    if args.limited_info_pursuer:
        env_kwargs["pursuer_behavior_mode"] = "limited_info_patrol"
        env_kwargs["pursuer_random_spawn"] = True
    if args.random_obstacles:
        env_kwargs["randomize_obstacles"] = True
    if args.enriched_random_obstacles:
        env_kwargs["randomize_obstacles"] = True
        env_kwargs["enriched_random_obstacles"] = True
    if args.center_spawn:
        env_kwargs["force_center_spawn"] = True
    if env_kwargs.get("randomize_obstacles") or env_kwargs.get("enriched_random_obstacles") or env_kwargs.get("force_center_spawn"):
        env_kwargs["center_spawn_when_random_obstacles"] = True
        print("Using random-obstacle training spawn.")
    if args.random_spawn:
        env_kwargs["pursuer_random_spawn"] = True
        env_kwargs["pursuer_random_spawn_min_evader_distance"] = args.min_pursuer_distance

    model, algo = (None, None)
    if args.condition == "model":
        print(f"Carregando modelo: {os.path.abspath(args.model)}")
        model, algo = _load_model(args.model, args.algorithm)
        print(f"Algoritmo carregado: {algo}")

    env = gym.make("Evader-v0", **env_kwargs)
    u = env.unwrapped

    if model is not None and model.observation_space != env.observation_space:
        raise RuntimeError("Espaco de observacao do checkpoint != env. Verifique os arquivos do evader_env.")

    from sb3_contrib import RecurrentPPO
    is_recurrent = isinstance(model, RecurrentPPO) if model is not None else False

    bounds = getattr(u, 'pursuer_random_spawn_bounds', (-150.0, 150.0, -170.0, 240.0))
    min_x, max_x, min_y, max_y = bounds
    rng = np.random.default_rng(args.seed)
    if args.random_spawn:
        try:
            env.reset(seed=args.seed)
            if hasattr(u, "_refresh_pursuer_obstacle_cache"):
                u._refresh_pursuer_obstacle_cache()
        except Exception:
            pass
    from controllers.evader_env.webots_runtime import SpawnPose as _SpawnPose

    def _safe_evader_xy():
        margin = max(getattr(u, 'pursuer_random_spawn_obstacle_margin', 8.0), 6.0)
        best, best_clear = None, -1e9
        for _ in range(1000):
            xy = np.array([rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)], dtype=np.float32)
            try:
                clear = u._pursuer_spawn_clearance(xy)
            except Exception:
                clear = float('inf')
            if clear > best_clear:
                best_clear, best = clear, xy
            if clear >= margin:
                return float(xy[0]), float(xy[1])
        return float(best[0]), float(best[1])

    def _safe_heading(ex, ey):
        best_h, best_clear = 0.0, -1e9
        for k in range(16):
            h = (k / 16.0) * 2.0 * np.pi
            fx, fy = ex + np.cos(h) * 15.0, ey + np.sin(h) * 15.0
            try:
                clear = u._pursuer_spawn_clearance(np.array([fx, fy], dtype=np.float32))
            except Exception:
                clear = float('inf')
            if clear > best_clear:
                best_clear, best_h = clear, h
            if clear >= 6.0:
                return float(h)
        return float(best_h)

    rows = []
    print(f"\nCondicao: {args.condition} | episodios: {args.episodes}\n")
    t0 = time.time()

    for ep in range(args.episodes):
        if args.random_spawn:
            ex, ey = _safe_evader_xy()
            u.spawn_poses = (_SpawnPose((ex, ey), (ex, ey), _safe_heading(ex, ey)),)
            u.force_center_spawn = False

        obs, _info = env.reset(seed=args.seed + ep)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        steps = visible_steps = los_breaks = 0
        dist_sum = 0.0
        prev_visible = None
        captured = collision = False

        while True:
            if args.condition == "stopped":
                action = np.array([2, 1])
            elif args.condition == "random":
                action = env.action_space.sample()
            elif is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=not args.stochastic,
                )
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)

            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            dist_sum += float(info.get("distance_to_pursuer", 0.0))

            visible = bool(info.get("pursuer_line_of_sight", info.get("pursuer_visible", 0.0)) >= 0.5)
            if visible:
                visible_steps += 1
            if prev_visible is not None and prev_visible and not visible:
                los_breaks += 1
            prev_visible = visible

            episode_starts = np.array([terminated or truncated], dtype=bool)
            if terminated or truncated:
                captured = bool(info.get("captured", False))
                collision = bool(info.get("obstacle_collision", False)) or bool(info.get("rollover", False))
                break

        success = not (captured or collision)
        rows.append({
            "episode": ep,
            "survival_steps": steps,
            "success": int(success),
            "captured": int(captured),
            "collision": int(collision),
            "mean_pursuer_distance": round(dist_sum / steps, 2) if steps else 0.0,
            "hidden_ratio": round(1.0 - visible_steps / steps, 3) if steps else 0.0,
            "los_breaks": los_breaks,
        })
        outcome = "SUCESSO" if success else ("CAPTURA" if captured else "COLISAO")
        print(f"ep {ep:3d} | passos={steps:5d} | {outcome:8s} | dist={rows[-1]['mean_pursuer_distance']:6.1f}m | escondido={rows[-1]['hidden_ratio']*100:4.1f}% | LOSbreaks={los_breaks}")

    env.close()

    n = len(rows)
    def col(k): return np.array([r[k] for r in rows], dtype=float)
    surv = col("survival_steps")
    print("\n" + "=" * 64)
    print(f"TABELA IX -- condicao: {args.condition} -- {n} episodios")
    print("=" * 64)
    print(f"  Survival steps (mean +/- std) : {surv.mean():.1f} +/- {surv.std():.1f}")
    print(f"  Success rate                  : {100*col('success').mean():.1f} %")
    print(f"  Capture rate                  : {100*col('captured').mean():.1f} %")
    print(f"  Collision rate                : {100*col('collision').mean():.1f} %")
    print(f"  Mean pursuer distance         : {col('mean_pursuer_distance').mean():.1f} m")
    print(f"  Hidden ratio                  : {100*col('hidden_ratio').mean():.1f} %")
    print(f"  Line-of-sight breaks (per ep) : {col('los_breaks').mean():.2f}")
    print("=" * 64)

    csv_path = args.csv
    if csv_path is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "logs", "eval_reports"), exist_ok=True)
        mode = "patrol" if (args.config and "patrol" in str(args.config)) else "chase"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fname = f"tableIX__{args.condition}__{mode}__ep{args.episodes}__{stamp}.csv"
        csv_path = os.path.join(PROJECT_ROOT, "logs", "eval_reports", fname)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV salvo em: {csv_path}")
    print(f"Tempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
