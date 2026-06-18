# Autonomous Evasion with PPO in Webots

University of Porto, Faculty of Sciences (FCUP)
CC3046 - Introduction to Intelligent Robotics, 2025/2026
Group T3-G9

- Guilherme Klippel (202300276)
- Pedro Paulo Basilio (202300926)
- Yan Coelho (202300916)

## Overview

This project trains an evader vehicle to escape one or more pursuers while
avoiding buildings and other obstacles in a Webots city. The learning
environment is implemented with Gymnasium and supports both PPO and recurrent
PPO from Stable-Baselines3.

The environment controls the evader through Webots' `<extern>` controller. The
pursuer, obstacle randomization, reward calculation, episode resets, and debug
visualization are managed by the Gym environment and Webots supervisor API.

## Current System

### Algorithms

- PPO with `MultiInputPolicy` is the default.
- Recurrent PPO with `MultiInputLstmPolicy` is enabled with
  `--algorithm recurrent_ppo`.
- Training can continue from a compatible checkpoint with `--resume-from`.
- Checkpoints include rolling reward and episode-length metrics in their names.

### Action space

The default action space is `MultiDiscrete([5, 5])`:

- Steering targets: `[-0.50, -0.25, 0.00, 0.25, 0.50]` radians.
- Drive targets: `[-0.50, 0.00, 0.35, 0.70, 1.00]`.

Each action selects an absolute target. Selecting straight steering therefore
returns the wheels toward zero instead of preserving their previous angle.

### Observation space

The policy receives a Gymnasium dictionary observation:

- `lidar` (12 values): three minimum-distance sectors from each of the front,
  left, back, and right LiDARs.
- `vision` (1 value): binary camera recognition of a pursuer.
- `pursuer` (4 values): normalized relative position, distance, and bearing
  derived from supervisor positions.
- `ego` (7 values): speed, acceleration, steering, yaw rate, heading sine and
  cosine, and touch contact.
- `avoidance` (5 values): front risk, safer turn direction, committed turn
  direction, heading progress, and remaining heading change.

### Pursuer and obstacles

- Direct-chase and limited-information pursuer modes are available.
- In limited-information mode, line of sight gives the pursuer the exact evader
  position. Without line of sight it follows the last periodically received
  position and falls back toward the map center when no target is known.
- The pursuer uses the supervisor's 2D obstacle map and A* path planning.
- Buildings can be randomized at every reset.
- Enriched obstacle training places blockers in front/right, front/left, or
  left/right arrangements around the centered evader spawn.
- The optional evader obstacle-safety teacher can assist training, but should
  be disabled for an unaided final evaluation.

## Requirements

- Windows 10 or 11.
- Webots R2025a installed in `C:\Program Files\Webots`, or `WEBOTS_HOME` set to
  the installation directory.
- Python 3.13 is the currently tested version.
- A CPU is sufficient; CUDA can be used when supported by the installed PyTorch
  build.

Webots supplies its own `controller` and `vehicle` Python modules. They are not
installed through `requirements.txt`.

## Installation

From PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open `worlds\my_city_traffic.wbt` once and confirm that the evader vehicle uses
the `<extern>` controller and is named `evader`.

## Training

### Standard PPO experiment

This command starts Webots automatically in fast, no-rendering batch mode:

```powershell
.\.venv\Scripts\python.exe .\controllers\training.py `
  --config .\configs\default_experiment.json `
  --algorithm ppo `
  --nowebots `
  --no-obstacle-safety `
  --save-name evader_full_ppo `
  --hide-reward-display `
  --limited-info-pursuer `
  --enriched-random-obstacles
```

### Recurrent PPO obstacle curriculum

```powershell
.\.venv\Scripts\python.exe .\controllers\training.py `
  --config .\configs\obstacle_curriculum.json `
  --algorithm recurrent_ppo `
  --log-dir logs\lstm_obstacle_runs `
  --save-name evader_lstm_obstacle_course `
  --nowebots `
  --no-obstacle-safety `
  --hide-reward-display `
  --enriched-random-obstacles
```

### Continue from a checkpoint

The checkpoint algorithm and observation space must match the new run:

```powershell
.\.venv\Scripts\python.exe .\controllers\training.py `
  --config .\configs\default_experiment.json `
  --algorithm recurrent_ppo `
  --resume-from logs\lstm_obstacle_runs\checkpoints\CHECKPOINT.zip `
  --log-dir logs\lstm_runs `
  --save-name evader_lstm_complete `
  --nowebots `
  --no-obstacle-safety `
  --hide-reward-display `
  --limited-info-pursuer `
  --enriched-random-obstacles
```

To run with an already open graphical Webots instance, omit `--nowebots`, load
the world, and start the simulation before launching the Python command.

## Inference

```powershell
.\.venv\Scripts\python.exe .\controllers\inference.py `
  --model logs\checkpoints\MODEL.zip `
  --algorithm auto `
  --stochastic `
  --limited-info-pursuer `
  --enriched-random-obstacles `
  --hide-reward-display `
  --no-obstacle-safety
```

Remove `--stochastic` for deterministic actions. Use the same environment flags
that were used during training to avoid evaluating the model under a different
spawn or obstacle distribution.

For detailed diagnostics:

```powershell
.\.venv\Scripts\python.exe .\controllers\inference.py `
  --model logs\checkpoints\MODEL.zip `
  --algorithm auto `
  --debug-report `
  --limited-info-pursuer `
  --enriched-random-obstacles `
  --no-obstacle-safety
```

Debug CSV files are written to `logs\debug_reports`. The optional
`--export-planner-map` flag exports the pursuer's planner grid, but frequent map
exports can substantially reduce simulation speed.

## Evaluation

Evaluate a trained model over 30 episodes:

```powershell
.\.venv\Scripts\python.exe .\controllers\evaluate.py `
  --condition model `
  --model logs\checkpoints\MODEL.zip `
  --algorithm auto `
  --episodes 30 `
  --limited-info-pursuer `
  --enriched-random-obstacles `
  --hide-reward-display `
  --no-obstacle-safety
```

Baseline conditions use the same script:

```powershell
.\.venv\Scripts\python.exe .\controllers\evaluate.py --condition stopped --episodes 30 --limited-info-pursuer --enriched-random-obstacles --hide-reward-display --no-obstacle-safety
.\.venv\Scripts\python.exe .\controllers\evaluate.py --condition random --episodes 30 --limited-info-pursuer --enriched-random-obstacles --hide-reward-display --no-obstacle-safety
```

Evaluation reports are written to `logs\eval_reports` unless `--csv` specifies
another destination.

## TensorBoard

Training writes event files below the selected log directory:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\logs
```

Then open the local URL printed by TensorBoard. Use `--no-tensorboard` when
event-file synchronization causes training interruptions.

## Configuration

- `configs\default_experiment.json`: full pursuit-evasion experiment and shared
  defaults.
- `configs\obstacle_curriculum.json`: obstacle-focused curriculum with the
  pursuer delayed beyond the episode duration.
- `configs\limited_info_patrol.json`: limited-information pursuer overrides.

An alternate configuration is merged over `default_experiment.json`, so an
experiment file only needs to contain values that differ from the defaults.
Command-line flags override the corresponding configuration values.

## Project Structure

```text
configs/
  default_experiment.json
  limited_info_patrol.json
  obstacle_curriculum.json
controllers/
  evader_env/
    __init__.py          Gym environment, sensors, pursuer, planner, and resets
    reward.py            Reward components and weights
    debug_display.py     Webots and terminal diagnostics
    webots_runtime.py    Webots paths and spawn metadata
  evaluate.py            Multi-episode experiment evaluation
  experiment_config.py   JSON configuration loading and merging
  inference.py           Trained-policy execution and debug reporting
  training.py            PPO/RecurrentPPO training and checkpointing
worlds/
  my_city_traffic.wbt    Webots city and vehicle definitions
logs/                    Generated models, TensorBoard data, and reports
```

## Generated Outputs

- Final models: directly under the selected `--log-dir`.
- Metric checkpoints: `<log-dir>\checkpoints`.
- TensorBoard runs: `<log-dir>\tensorboard_logs`.
- Inference diagnostics: `logs\debug_reports`.
- Evaluation CSV files: `logs\eval_reports`.

The observation structure is part of the saved policy contract. A checkpoint
trained before an observation-space change cannot be loaded safely by the
current environment and should be retrained or evaluated with its matching
code revision.
