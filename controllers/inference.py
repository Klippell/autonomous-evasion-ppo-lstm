"""Run a trained recurrent PPO evader policy in Webots."""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm

import controllers.evader_env
from controllers.experiment_config import DEFAULT_CONFIG_PATH, env_kwargs_from_config, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained evader policy.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--model",
        default=os.path.join(PROJECT_ROOT, "logs", "evader_recurrent_ppo.zip"),
    )
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy, matching training-time behavior more closely.",
    )
    parser.add_argument("--hide-reward-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    print(f"Loading model: {os.path.abspath(args.model)}")
    print(f"Inference mode: {'stochastic' if args.stochastic else 'deterministic'}")
    model: BaseAlgorithm = RecurrentPPO.load(args.model)
    env_kwargs = env_kwargs_from_config(config)
    env_kwargs.update(
        robot_name=args.robot_name,
        show_reward_display=not args.hide_reward_display,
    )
    env: gym.Env = gym.make(
        "Evader-v0",
        **env_kwargs,
    )

    obs, _info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=not args.stochastic,
        )
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        if done:
            lstm_states = None
            obs, _info = env.reset()
        episode_starts = np.array([done], dtype=bool)


if __name__ == "__main__":
    main()
