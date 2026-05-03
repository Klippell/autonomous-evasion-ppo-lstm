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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained evader policy.")
    parser.add_argument(
        "--model",
        default=os.path.join(PROJECT_ROOT, "logs", "evader_recurrent_ppo.zip"),
    )
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model: BaseAlgorithm = RecurrentPPO.load(args.model)
    env: gym.Env = gym.make("Evader-v0", robot_name=args.robot_name)

    obs, _info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)
        if done:
            obs, _info = env.reset()


if __name__ == "__main__":
    main()
