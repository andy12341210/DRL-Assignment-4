import gymnasium as gym
import numpy as np
from model import SACAgent
import torch
from dmc import make_dmc_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        env = make_dmc_env("humanoid-walk", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.agent = SACAgent(state_dim,action_dim)
        checkpoint = torch.load("final.pth", weights_only=True)
        self.agent.actor.load_state_dict(checkpoint['actor'])

    def act(self, observation):
        return self.agent.select_action(observation)
    
