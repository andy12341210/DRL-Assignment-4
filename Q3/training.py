import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SACAgent
from tqdm import tqdm
from dmc import make_dmc_env

def train():
    env = make_dmc_env("humanoid-walk", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # print([state_dim, action_dim])
    
    agent = SACAgent(state_dim, action_dim)

    # checkpoint = torch.load("modelQ3.pth", weights_only=True)
    # agent.actor.load_state_dict(checkpoint['actor'])
    # agent.critic1.load_state_dict(checkpoint['critic1'])
    # agent.critic2.load_state_dict(checkpoint['critic2'])
    # agent.alpha = checkpoint['alpha']

    max_episodes = 10000
    max_steps = 1000
    update_interval = 50
    print_interval = 10
    checkpoint = 100
    progress_bar = tqdm(range(max_episodes))
    reward_history = []
    steps_history = []
    epsilon = 1
    decay = 0.999
    
    for episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            
            if np.random.rand() < epsilon:
                noise = np.random.normal(loc=0.0,scale=0.2,size=action_dim)
                action += noise
                action = np.clip(action, -1, 1)

            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            if done or truncated:
                steps_history.append(step)
                break

        reward_history.append(episode_reward)
        epsilon *= decay
        
        if episode % print_interval == 0 and episode != 0:
            moving_avg = np.mean(reward_history[-print_interval:])
            step_avg = np.mean(steps_history[-print_interval:])
            tqdm.write(f"Episode: {episode}, avg Reward: {moving_avg}, Alpha: {agent.alpha.item():.3f}, avg_steps: {step_avg:.1f}")
            
            rewards_array = np.array(reward_history)
            moving_avg_curve = np.convolve(rewards_array, np.ones(print_interval)/print_interval, mode='valid')

            fig = plt.figure()
            plt.plot(reward_history, label='Total Reward', alpha=0.3)
            plt.plot(range(print_interval-1, len(reward_history)), moving_avg_curve, label=f'{print_interval}-Episode Moving Avg', color='orange')
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Training History")
            plt.savefig("../picture/trainQ3_2", dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        if episode % checkpoint == 0 and episode != 0:
            torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic1': agent.critic1.state_dict(),
                    'critic2': agent.critic2.state_dict(),
                    'alpha': agent.alpha
                }, f"./modelQ3_2.pth")
            

    torch.save({
        'actor': agent.actor.state_dict(),
        'critic1': agent.critic1.state_dict(),
        'critic2': agent.critic2.state_dict(),
        'alpha': agent.alpha
    }, f"./modelQ3_2.pth")

if __name__ == '__main__':
    train()