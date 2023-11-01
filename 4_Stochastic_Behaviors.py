import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense, Flatten
import gymnasium as gym

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

class BehaviorCloningPolicy:
    """A class for the behavior cloning policy based on a trained model."""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, ego_vehicle, vehicles):
        obs = [self.vehicle_to_observation(ego_vehicle)]
        for vehicle in vehicles:
            obs.append(self.vehicle_to_observation(vehicle))
        while len(obs) < 5:
            obs.append([0, 0, 0, 0, 0])  # Padding with "absence" observation
        obs = np.array(obs).reshape(1, 5, 5)
        action_probs = self.model.predict(obs)
        return np.argmax(action_probs[0])

    @staticmethod
    def vehicle_to_observation(vehicle):
        return [
            1.0,
            vehicle.position[0] / 100.0,
            vehicle.position[1] / 5.0,
            vehicle.speed / 30.0,
            vehicle.heading / (2 * np.pi)
        ]

def perturb_and_save_model(model, perturbation_factor=0.05, number_of_models=5):
    """Function to perturb a given model and save multiple instances."""
    
    if not os.path.exists("perturbed_models"):
        os.makedirs("perturbed_models")

    model_paths = []
    for i in range(number_of_models):
        perturbed_model = clone_model(model)
        perturbed_model.set_weights(model.get_weights())
        
        # Apply perturbation
        for layer in perturbed_model.layers:
            weights = layer.get_weights()
            perturbed_weights = [w + np.random.normal(0, perturbation_factor, size=w.shape) for w in weights]
            layer.set_weights(perturbed_weights)

        path = os.path.join("perturbed_models", "perturbed_model_{}.keras".format(i))
        perturbed_model.save(path)
        model_paths.append(path)

    return model_paths

def simulate_with_enhanced_agents(model_paths):
    """Function to simulate the environment using different agents (perturbed models)."""
    
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    agents = [BehaviorCloningPolicy(model_path) for model_path in model_paths]
    
    for _ in range(1000):
        done = False
        obs = env.reset()
        while not done:
            ego_vehicle = env.vehicle
            other_vehicles = list(np.random.choice(env.road.vehicles, size=4, replace=False))

            agent = np.random.choice(agents)
            agent_action = agent.predict(ego_vehicle, other_vehicles)
            obs, reward, done, _, _ = env.step(agent_action)
            env.render()

if __name__ == "__main__":
    original_model_path = "behavior_cloning_model.keras"
    original_model = load_model(original_model_path)
    model_paths = perturb_and_save_model(original_model)
    simulate_with_enhanced_agents(model_paths)
