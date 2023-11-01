import gymnasium as gym
from stable_baselines3 import PPO

# Constants
TRAIN_MODE = True
TOTAL_TIMESTEPS = int(2e4)
LOG_DIR = "highway_ppo/"
SAVE_DIR = "highway_ppo/model"

def create_highway_environment():
    """
    Initialize the highway environment.
    """
    environment = gym.make("highway-fast-v0", render_mode="human")
    initial_observation, _ = environment.reset()
    return environment, initial_observation

def setup_ppo_agent(environment):
    """
    Configure the PPO agent with given parameters.
    """
    agent = PPO('MlpPolicy', environment,
                policy_kwargs={'net_arch': [256, 256]},
                learning_rate=5e-4,
                gamma=0.8,
                n_steps=5,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=LOG_DIR)
    return agent

def train_agent(agent):
    """
    Train the PPO agent for a specified number of timesteps.
    """
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    agent.save(SAVE_DIR)

def run_ppo_simulation(agent, environment, episodes=10000):
    """
    Run a simulation using the trained PPO agent.
    """
    for episode in range(episodes):
        done = False
        observation, _ = environment.reset()
        while not done:
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, done, _,_ = environment.step(action)
            environment.render()

def main():
    environment, _ = create_highway_environment()
    ppo_agent = setup_ppo_agent(environment)
    
    if TRAIN_MODE:
        train_agent(ppo_agent)
        del ppo_agent
    
    # Load the trained model and run the simulation
    ppo_agent = PPO.load(SAVE_DIR, env=environment)
    run_ppo_simulation(ppo_agent, environment)

    environment.close()

if __name__ == '__main__':
    main()
