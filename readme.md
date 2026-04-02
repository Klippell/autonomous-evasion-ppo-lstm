==========================================================================
UNIVERSITY OF PORTO - FACULTY OF SCIENCES (FCUP)
CC3046 - Introduction to Intelligent Robotics | Academic Year 2025/2026
PROJECT: Vision-Based Autonomous Pursuit-Evasion in Urban Environments
==========================================================================

GROUP: T3-G9
STUDENTS:
- Guilherme Klippel (202300276)
- Pedro Paulo Basilio (202300926)
- Yan Coelho (202300916)

--------------------------------------------------------------------------
1. PROJECT SUMMARY
--------------------------------------------------------------------------
This project explores Reinforcement Learning (RL) techniques, specifically
Proximal Policy Optimization (PPO), to train an autonomous evader vehicle
within a "cops and robbers" scenario in the Webots simulator.

The main objective is for the evader to learn how to navigate dynamic
urban environments using only onboard sensors to break the line-of-sight
(Line-of-Sight) and avoid capture by a pursuing agent.

--------------------------------------------------------------------------
2. TECHNICAL APPROACH
--------------------------------------------------------------------------
- Base Algorithm: PPO (Proximal Policy Optimization) for continuous control.
- Temporal Memory: LSTM layers to handle partial observability and
  temporary loss of visual contact with the pursuer.
- State Space (Sensors):
  * Camera: Visual detection of the pursuer vehicle.
  * LiDAR: Perception of obstacles and urban infrastructure.
  * GPS: Used by the pursuer (at frequency X) and for performance metrics.
- Action Space (Actuators):
  * Steering: Control of the steering angle of the front wheels.
  * Throttle: Control over acceleration and braking intensity.

--------------------------------------------------------------------------
3. REQUIREMENTS AND INSTALLATION
--------------------------------------------------------------------------
- Simulator: Webots R2023b or superior.
- Language: Python 3.10.x.
- Dependencies: stable-baselines3[extra], sb3-contrib, torch.

Quick Installation:
1. Create venv: python -m venv .venv
2. Activate venv: .venv\Scripts\activate
3. Install libraries: pip install -r requirements.txt

--------------------------------------------------------------------------
4. DIRECTORY STRUCTURE
--------------------------------------------------------------------------
/controllers/evader_controller/  -> Source code for the autonomous evader.
/worlds/                         -> .wbt files (e.g., "City Traffic" scene).
/models/                         -> (Future) Trained PPO+LSTM models.
README.txt                       -> Project documentation.
requirements.txt                 -> List of Python dependencies.

--------------------------------------------------------------------------
5. DEVELOPMENT MILESTONES
--------------------------------------------------------------------------
- Weeks 1-2: Initialization, simulation setup, and sensor testing. (COMPLETED)
- Weeks 3-5: PPO implementation and baseline pursuer behavior.
- Weeks 6-7: LSTM integration and reward function refinement.
- Weeks 8-10: Experiments and data collection (Survival Time/Escape Rate).
- Weeks 11-12: Final report completion and final demonstration.

--------------------------------------------------------------------------
6. EXECUTION
--------------------------------------------------------------------------
1. Open Webots with the "my_city_traffic.wbt" world.
2. Set the vehicle controller to <extern> mode.
3. Execute evader_controller.py from your IDE (e.g., PyCharm).
==========================================================================