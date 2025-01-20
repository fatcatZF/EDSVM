import numpy as np 
import math 

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.envs.classic_control.cartpole import CartPoleEnv 
from gymnasium.wrappers import TransformReward, TransformObservation, TransformAction

from typing import Optional



def make_cartpole():
    class RewardShapingWrapper(Wrapper):
      def __init__(self, env):
        super().__init__(env)

      def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Modify the reward based on observation
        reward = self.modify_reward(observation, reward)
        return observation, reward, terminated, truncated, info

      def modify_reward(self, observation, reward):
        # Define custom reward modification logic
        # For example, add a penalty if the pole angle is too high
        pole_angle = observation[2]
        reward = np.cos(pole_angle)
        return reward
    class CartPoleEnvDrifted(CartPoleEnv):
      def __init__(self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None):
        super(CartPoleEnvDrifted, self).__init__()
        self.force_mag = 11.5 # increase the force mag from 10.0 to 11.5
    gym.register("CartPoleDrifted-v1",
                 CartPoleEnvDrifted, max_episode_steps=500) 
    
    env0 = RewardShapingWrapper(gym.make("CartPole-v1")) 
    env1 = RewardShapingWrapper(gym.make("CartPole-v1"))
    env2 = RewardShapingWrapper(gym.make("CartPoleDrifted-v1"))
    env3 = RewardShapingWrapper(TransformObservation(env1,
                                            lambda obs: obs+0.005*np.random.randn(obs.shape[-1]),
                                            env1.observation_space))

    return env0, env1, env2, env3   


def make_lunarlander():
    env0 = gym.make("LunarLander-v3")
    env1 = gym.make("LunarLander-v3") 
    env2 = gym.make("LunarLander-v3", 
                    enable_wind=True,
                    wind_power=0.7) # add wind with power=5.
    env3 = TransformObservation(env1,
                            lambda obs: obs+0.005*np.random.randn(obs.shape[-1]),
                            env1.observation_space) # Noisy Production Environment
    
    
    return env0,env1, env2, env3 



def make_hopper():
   env0 = gym.make("Hopper-v5")
   env1 = gym.make("Hopper-v5")
   env2 = TransformAction(env1, lambda a: a*np.random.uniform(low = 0.8, high = 1.,
                                                           size = (3,)),
                      env0.action_space)
   env3 = TransformObservation(env1,
                            lambda obs: obs + 0.005 * np.random.randn(obs.shape[-1]),
                            env1.observation_space)
   return env0, env1, env2, env3 



def make_halfcheetah():
    env0 = gym.make("HalfCheetah-v5")
    env1 = gym.make("HalfCheetah-v5")
    env2 = TransformAction(env1, lambda a: a*np.random.uniform(low = 0.95, high = 1.,
                                                           size = (6,)),
                      env1.action_space)
    env3 = TransformObservation(env1,
                            lambda obs: obs + 0.005 * np.random.randn(obs.shape[-1]),
                            env1.observation_space)
    
    return env0, env1, env2, env3 


def make_humanoid():
   env0 = gym.make("Humanoid-v5")
   env1 = gym.make("Humanoid-v5")
   env2 = TransformAction(env1, lambda a: a*np.random.uniform(low = 1.0, high = 1.2, size = (17,)),
                      env1.action_space)
   env3 = TransformObservation(env1,
                            lambda obs: obs+0.005*np.random.randn(obs.shape[-1]),
                            env1.observation_space)
   
   return env0, env1, env2, env3



def make_env(name="cartpole"):
    if (name=="cartpole"):
       env0, env1, env2, env3 = make_cartpole() 
    
    elif (name=="lunarlander"):
       env0, env1, env2, env3 = make_lunarlander() 

    elif (name=="hopper"):
       env0, env1, env2, env3 = make_hopper() 

    elif (name=="halfcheetah"):
       env0, env1, env2, env3 = make_halfcheetah() 
    
    elif (name=="humanoid"):
       env0, env1, env2, env3 = make_humanoid()

    else:
       print(f"The environment {name} has not been implemented.")
       return 

    return env0, env1, env2, env3 





if __name__ == "__main__":
   env0, env1, env2, env3 = make_env(name="cartpole")
   env0, env1, env2, env3 = make_env(name="lunarlander")
   env0, env1, env2, env3 = make_env(name="hopper")
   env0, env1, env2, env3 = make_env(name="halfcheetah")
   env0, env1, env2, env3 = make_env(name="humanoid")