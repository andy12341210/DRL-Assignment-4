import gymnasium as gym
import numpy as np
from model import SACAgent
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        env = gym.make('Pendulum-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.agent = SACAgent(state_dim,action_dim)
        checkpoint = torch.load("modelQ1.pth", weights_only=True)
        self.agent.actor.load_state_dict(checkpoint['actor'])

    def act(self, observation):
        return self.agent.select_action(observation)*2