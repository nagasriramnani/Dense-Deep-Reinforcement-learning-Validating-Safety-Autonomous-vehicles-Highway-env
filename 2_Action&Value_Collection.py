import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import highway_env
import numpy as np

# Constants
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}
MODEL_PATH = "C:\\Users\\Ram\\highway\\highway_ppo\\model.zip"
DATA_FILE = 'collected_data_ppo.csv'

def main():
    env = setup_environment()
    model = load_trained_model(env)

    observations, actions_taken, rewards_received, infos = collect_data(env, model)
    
    save_data_to_csv(observations, actions_taken, rewards_received, infos)

def setup_environment():
    env = gym.make("highway-fast-v0", render_mode="human")
    env.reset()
    return env

def load_trained_model(env):
    return PPO.load(MODEL_PATH, env=env)

def collect_data(env, model):
    observations = []
    actions_taken = []
    rewards_received = []
    infos = []

    for _ in range(1000):
        done = False
        obs, info = env.reset()
        while not done:
            action, obs_str, reward,done, info = perform_model_action(env, model, obs)
            actions_taken.append(ACTIONS_ALL[action])
            observations.append(obs_str)
            rewards_received.append(reward)
            infos.append(info)
            env.render()

    env.close()
    return observations, actions_taken, rewards_received, infos

def perform_model_action(env, model, obs):
    action, _states = model.predict(obs, deterministic=False)
    if isinstance(action, np.ndarray):
        action = action.item()

    obs_str = str(obs.tolist())
    obs, reward, done, info,_ = env.step(action)  # Only call env.step() once

    return action, obs_str, reward, done, info


def save_data_to_csv(observations, actions_taken, rewards_received, infos):
    df = pd.DataFrame({
        'Observations': observations,
        'Actions': actions_taken,
        'Rewards': rewards_received,
        'Info': infos
    })
    df.to_csv(DATA_FILE, index=False)

if __name__ == '__main__':
    main()
