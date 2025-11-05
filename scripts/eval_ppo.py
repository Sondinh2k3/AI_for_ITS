"""
Evaluation Script for Trained PPO Models

This script loads a trained PPO checkpoint and evaluates it on the SUMO environment.
It can also render the traffic simulation.
"""

import os
import sys
import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.drl_algo.env import SumoEnvironment


def register_sumo_env(config_dict: dict):
    """Register SUMO environment with RLlib."""
    register_env(
        "sumo_traffic_signal_v0",
        lambda env_config: SumoEnvironment(**{**config_dict, **env_config})
    )


def evaluate_ppo(
    checkpoint_path: str,
    network_name: str = "grid4x4",
    num_episodes: int = 5,
    use_gui: bool = False,
    max_steps: int = 1000,
):
    """Evaluate a trained PPO model.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        network_name: Name of network
        num_episodes: Number of episodes to evaluate
        use_gui: Whether to use SUMO GUI
        max_steps: Maximum steps per episode
    """
    
    print("\n" + "="*80)
    print("PPO MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Network: {network_name}")
    print(f"Episodes: {num_episodes}")
    print("="*80 + "\n")
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(ignore_reinit_error=True, object_store_memory=int(1e9))
    
    try:
        # Prepare network files
        base_path = Path(__file__).parent.parent / "network" / network_name
        net_file = str(base_path / f"{network_name}.net.xml")
        route_file = str(base_path / f"{network_name}.rou.xml")
        
        # Environment configuration
        env_config = {
            "net_file": net_file,
            "route_file": route_file,
            "use_gui": use_gui,
            "render_mode": "rgb_array" if use_gui else None,
            "max_green": 60,
            "min_green": 5,
            "delta_time": 5,
            "yellow_time": 3,
        }
        
        # Register environment
        register_sumo_env(env_config)
        
        # Load trained algorithm
        print("Loading trained model...")
        algo = PPO.from_checkpoint(checkpoint_path)
        print("✓ Model loaded successfully\n")
        
        # Create environment for evaluation
        eval_env = SumoEnvironment(**env_config)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            obs, info = eval_env.reset()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < max_steps:
                # Get action from trained policy
                if isinstance(obs, dict):
                    # Multi-agent case
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        action = algo.compute_single_action(agent_obs, policy_id="default_policy")
                        actions[agent_id] = action
                    obs, reward, terminated, truncated, info = eval_env.step(actions)
                    episode_reward += sum(reward.values()) if isinstance(reward, dict) else reward
                else:
                    # Single-agent case
                    action = algo.compute_single_action(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Reward: {episode_reward:.2f}, Steps: {episode_length}\n")
        
        eval_env.close()
        
        # Print statistics
        print("="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Average Episode Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Average Episode Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
        print(f"Max Reward: {max(episode_rewards):.2f}")
        print(f"Min Reward: {min(episode_rewards):.2f}")
        print("="*80 + "\n")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="grid4x4",
        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
        help="Network name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO GUI"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    evaluate_ppo(
        checkpoint_path=args.checkpoint,
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        max_steps=args.max_steps,
    )
