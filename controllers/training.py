"""Train the Webots evader with recurrent PPO."""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback

import controllers.evader_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an evader that escapes a pursuer while avoiding obstacles.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs"))
    parser.add_argument("--save-name", default="evader_recurrent_ppo")
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    env: gym.Env = gym.make("Evader-v0", robot_name=args.robot_name)
    model: BaseAlgorithm = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard_logs"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix=args.save_name,
    )
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        tb_log_name=time.strftime("%Y%m%d-%H%M%S"),
        callback=checkpoint_callback,
    )
    model.save(os.path.join(args.log_dir, args.save_name))
    env.close()


if __name__ == "__main__":
    main()
