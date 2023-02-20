# Baselines3 Module Based Algorithms

Clone the code, and then install the environment from root directory (where setup.py located):

```
pip install -e .
```

Then, in Python:

If GUI is using, then statement should be added when create envionrment, default is False

```
import gym 
import turtlebot_env
env = gym.make('Turtlebot-v0', use_gui=True) 
```

v0: X fixed and Y random target, but over the limitatitions of of the robot

v1: Random target, but dense reward

v2: Random target, with sparse reward

v3: Random target with distance related reward, but over the limitatitions of of the robot

