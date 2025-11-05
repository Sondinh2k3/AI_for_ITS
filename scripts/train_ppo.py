"""
PPO (Proximal Policy Optimization) Training Script for Adaptive Traffic Signal Control

This script trains PPO agents to control traffic signals using the SUMO environment.
Each traffic signal is controlled by an independent agent that learns to optimize
traffic flow by adjusting signal phases.

Author: Son Dinh
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.tune.stopper import Stopper

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.drl_algo.env import SumoEnvironment


class CustomStopper(Stopper):
    """Custom stopper for training based on reward threshold or max iterations."""
    
    def __init__(self, max_iter: int = 1000, reward_threshold: float = None):
        """
        Args:
            max_iter: Maximum number of training iterations
            reward_threshold: Stop if mean reward exceeds this value
        """
        self.max_iter = max_iter
        self.reward_threshold = reward_threshold
        self.best_reward = float('-inf')
    
    def __call__(self, trial_id, result):
        """Called after each training iteration."""
        # Check max iterations
        if result["training_iteration"] >= self.max_iter:
            print(f"\n✓ Training completed: reached max iterations ({self.max_iter})")
            return True
        
        # Check reward threshold
        if self.reward_threshold is not None:
            mean_reward = result.get("env_runners", {}).get("episode_reward_mean", float('-inf'))
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
            
            if mean_reward > self.reward_threshold:
                print(f"\n✓ Training completed: reward threshold reached ({mean_reward:.2f})")
                return True
        
        return False
    
    def stop_all(self):
        """Return True to stop all trials."""
        return False


def register_sumo_env(config_dict: dict):
    """Register SUMO environment with RLlib.
    
    Args:
        config_dict: Dictionary containing network and route file paths
    """
    register_env(
        "sumo_traffic_signal_v0",
        lambda env_config: SumoEnvironment(**{**config_dict, **env_config})
    )


def create_ppo_config(
    env_config: dict,
    num_workers: int = 2,
    num_envs_per_worker: int = 1,
    learning_rate: float = 5e-5,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coeff: float = 0.01,
    clip_param: float = 0.3,
    use_gpu: bool = False,
) -> PPOConfig:
    """Create and configure PPO algorithm.
    
    Args:
        env_config: Environment configuration dictionary
        num_workers: Number of parallel workers
        num_envs_per_worker: Number of environments per worker
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        lambda_: GAE lambda parameter
        entropy_coeff: Entropy coefficient for exploration
        clip_param: PPO clip parameter
        use_gpu: Whether to use GPU
    
    Returns:
        Configured PPOConfig object
    """
    config = (
        PPOConfig()
        .environment(env="sumo_traffic_signal_v0", env_config=env_config)
        .framework("torch")
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=num_envs_per_worker,
            rollout_fragment_length="auto",
        )
        .training(
            lr=learning_rate,
            gamma=gamma,
            lambda_=lambda_,
            entropy_coeff=entropy_coeff,
            clip_param=clip_param,
            vf_clip_param=10.0,  # Value function clip parameter
            sgd_minibatch_size=128,
            train_batch_size=4096,
            num_sgd_iter=30,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256],  # Hidden layer sizes
                "fcnet_activation": "relu",
                "vf_share_layers": False,  # Separate value and policy networks
            },
        )
        .resources(num_gpus=1 if use_gpu else 0)
        .debugging(log_level="INFO")
    )
    
    return config


def train_ppo(
    network_name: str = "grid4x4",
    num_iterations: int = 100,
    num_workers: int = 2,
    checkpoint_interval: int = 10,
    reward_threshold: float = None,
    experiment_name: str = None,
    use_gui: bool = False,
    use_gpu: bool = False,
    seed: int = 42,
    output_dir: str = "./results",
):
    """Main training function for PPO.
    
    Args:
        network_name: Name of network (grid4x4, 4x4loop, etc.)
        num_iterations: Number of training iterations
        num_workers: Number of parallel workers
        checkpoint_interval: Save checkpoint every N iterations
        reward_threshold: Stop training if reward exceeds this
        experiment_name: Name for the experiment (for logging)
        use_gui: Whether to use SUMO GUI
        use_gpu: Whether to use GPU
        seed: Random seed
        output_dir: Directory to save results
    """
    
    # Create results directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set experiment name with timestamp
    if experiment_name is None:
        experiment_name = f"ppo_{network_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*80)
    print("PPO TRAINING FOR ADAPTIVE TRAFFIC SIGNAL CONTROL")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Network: {network_name}")
    print(f"Iterations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"GPU: {use_gpu}")
    print(f"Seed: {seed}")
    print("="*80 + "\n")
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(2e9),  # 2GB object store
    )
    
    try:
        # Set random seed
        import numpy as np
        np.random.seed(seed)
        
        # Prepare network files
        base_path = Path(__file__).parent.parent / "network" / network_name
        net_file = str(base_path / f"{network_name}.net.xml")
        route_file = str(base_path / f"{network_name}.rou.xml")
        
        if not Path(net_file).exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        if not Path(route_file).exists():
            raise FileNotFoundError(f"Route file not found: {route_file}")
        
        print(f"✓ Network file: {net_file}")
        print(f"✓ Route file: {route_file}\n")
        
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
        
        # Create PPO config
        ppo_config = create_ppo_config(
            env_config=env_config,
            num_workers=num_workers,
            learning_rate=5e-5,
            entropy_coeff=0.01,
            use_gpu=use_gpu,
        )
        
        # Create stopper
        stopper = CustomStopper(
            max_iter=num_iterations,
            reward_threshold=reward_threshold,
        )
        
        # Setup callbacks for logging
        callbacks_config = {
            "on_train_result": lambda algorithm, result: (
                print(
                    f"Iteration {result['training_iteration']:3d} | "
                    f"Episode Reward Mean: {result.get('env_runners', {}).get('episode_reward_mean', 0):8.2f} | "
                    f"Episode Len Mean: {result.get('env_runners', {}).get('episode_len_mean', 0):8.1f}"
                )
            ),
        }
        
        # Create Tuner for distributed training
        tuner = tune.Tuner(
            "PPO",
            param_space=ppo_config.to_dict(),
            run_config=air.RunConfig(
                name=experiment_name,
                local_dir=str(output_dir),
                stop=stopper,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=checkpoint_interval,
                    checkpoint_keep_num=3,  # Keep last 3 checkpoints
                    checkpoint_score_attr="episode_reward_mean",
                ),
                verbose=1,
                progress_reporter=tune.CLIReporter(
                    metric_columns=[
                        "episode_reward_mean",
                        "episode_len_mean",
                        "num_env_steps_trained",
                    ]
                ),
            ),
        )
        
        # Run training
        print("Starting PPO training...\n")
        results = tuner.fit()
        
        # Save training results
        best_checkpoint = results.get_best_checkpoint(metric="episode_reward_mean", mode="max")
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Best Checkpoint: {best_checkpoint}")
        print(f"Best Episode Reward Mean: {best_result.metrics.get('episode_reward_mean', 0):.2f}")
        print("="*80 + "\n")
        
        # Save configuration
        config_file = output_dir / experiment_name / "training_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({
                "experiment_name": experiment_name,
                "network_name": network_name,
                "num_iterations": num_iterations,
                "num_workers": num_workers,
                "checkpoint_interval": checkpoint_interval,
                "use_gpu": use_gpu,
                "seed": seed,
                "best_checkpoint": str(best_checkpoint),
                "best_reward": float(best_result.metrics.get('episode_reward_mean', 0)),
            }, f, indent=2)
        
        print(f"✓ Configuration saved to: {config_file}")
        print(f"✓ Results saved to: {output_dir / experiment_name}")
        
        return best_checkpoint, best_result
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agents for adaptive traffic signal control"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="grid4x4",
        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
        help="Network name"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Checkpoint interval"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Stop training if reward exceeds this threshold"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO GUI"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    train_ppo(
        network_name=args.network,
        num_iterations=args.iterations,
        num_workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        reward_threshold=args.reward_threshold,
        experiment_name=args.experiment_name,
        use_gui=args.gui,
        use_gpu=args.gpu,
        seed=args.seed,
        output_dir=args.output_dir,
    )
