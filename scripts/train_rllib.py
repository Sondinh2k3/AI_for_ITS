
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from src.env.sumo_env import SumoEnvironment
import os

def main():
    # --- 1. Initialize Ray ---
    ray.init()

    # --- 2. Register the custom environment ---
    # RLlib needs a function that returns an instance of the environment.
    # The lambda function here captures the arguments for our SumoEnvironment.
    net_file = os.path.abspath("network/grid4x4/grid4x4.net.xml")
    route_file = os.path.abspath("network/grid4x4/grid4x4.rou.xml")
    
    register_env(
        "sumo_multi_agent_v0",
        lambda config: SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            **config
        )
    )

    # --- 3. Configure the DQN Algorithm ---
    # Create a dummy env to get action and observation spaces
    dummy_env = SumoEnvironment(net_file=net_file, route_file=route_file)
    obs_space = dummy_env.observation_space(dummy_env.possible_agents[0])
    act_space = dummy_env.action_space(dummy_env.possible_agents[0])
    dummy_env.close()

    config = (
        DQNConfig()
        .environment(env="sumo_multi_agent_v0", env_config={"use_gui": False, "steps_per_episode": 1000})
        .framework("torch")
        .rollouts(num_rollout_workers=1) # Use 1 worker for local training
        .training(model={"fcnet_hiddens": [256, 256]})
        .multi_agent(
            # All agents share a single policy.
            policies={"shared_policy": (None, obs_space, act_space, {})},
            # Map all agent IDs to this single shared policy.
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy")
        )
        # Disable exploration for evaluation
        .evaluation(evaluation_config={"explore": False})
    )

    # --- 4. Build and Train the Algorithm ---
    algo = config.build()

    print("Starting training...")
    for i in range(10): # Train for 10 iterations
        result = algo.train()
        print(f"Iteration: {i + 1}, Episode Reward Mean: {result['episode_reward_mean']:.2f}, Total Timesteps: {result['timesteps_total']}")

    # --- 5. Save the trained model (checkpoint) ---
    checkpoint_dir = algo.save()
    print(f"Checkpoint saved in directory: {checkpoint_dir}")

    # --- 6. Shutdown Ray ---
    ray.shutdown()

if __name__ == "__main__":
    main()
