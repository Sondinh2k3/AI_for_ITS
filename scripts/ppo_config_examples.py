"""
Advanced PPO Configuration Examples

This file contains example configurations for different scenarios.
Copy and modify these examples for your specific needs.
"""

from ray.rllib.algorithms.ppo import PPOConfig


# ============================================================================
# 1. SMALL NETWORK - QUICK TRAINING (Testing/Development)
# ============================================================================

def get_ppo_config_small():
    """Fast training for development/testing."""
    return (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,
            num_envs_per_worker=1,
        )
        .training(
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.3,
            sgd_minibatch_size=64,
            train_batch_size=2048,
            num_sgd_iter=20,
            model={
                "fcnet_hiddens": [128, 128],  # Smaller network
                "fcnet_activation": "relu",
            },
        )
    )


# ============================================================================
# 2. MEDIUM NETWORK - BALANCED TRAINING
# ============================================================================

def get_ppo_config_medium():
    """Balanced configuration for production."""
    return (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            num_envs_per_worker=1,
        )
        .training(
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.3,
            vf_clip_param=10.0,
            sgd_minibatch_size=128,
            train_batch_size=4096,
            num_sgd_iter=30,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            },
        )
    )


# ============================================================================
# 3. LARGE NETWORK - HIGH PERFORMANCE TRAINING
# ============================================================================

def get_ppo_config_large():
    """Large configuration for maximum performance."""
    return (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
            num_envs_per_worker=2,
        )
        .training(
            lr=1e-4,
            gamma=0.995,
            lambda_=0.98,
            entropy_coeff=0.005,
            clip_param=0.2,  # More conservative
            vf_clip_param=5.0,
            sgd_minibatch_size=256,
            train_batch_size=8192,
            num_sgd_iter=40,
            grad_clip=1.0,
            model={
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
                "free_log_std": True,
            },
        )
        .resources(num_gpus=1)  # Use GPU
    )


# ============================================================================
# 4. EXPLORATION-FOCUSED - FOR HARD PROBLEMS
# ============================================================================

def get_ppo_config_exploration():
    """High exploration for difficult environments."""
    return (
        PPOConfig()
        .framework("torch")
        .rollouts(num_rollout_workers=4)
        .training(
            lr=1e-4,
            entropy_coeff=0.05,  # High exploration
            clip_param=0.4,      # Generous clipping
            vf_clip_param=20.0,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
    )


# ============================================================================
# 5. STABILITY-FOCUSED - FOR CONVERGENCE
# ============================================================================

def get_ppo_config_stable():
    """Conservative configuration for stable convergence."""
    return (
        PPOConfig()
        .framework("torch")
        .rollouts(num_rollout_workers=2)
        .training(
            lr=1e-5,              # Low learning rate
            entropy_coeff=0.001,  # Low exploration
            clip_param=0.15,      # Tight clipping
            vf_clip_param=5.0,
            grad_clip=0.1,
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            },
        )
    )


# ============================================================================
# PARAMETER TUNING GUIDE
# ============================================================================

"""
Hyperparameter Tuning Recommendations:

1. LEARNING RATE (lr)
   - Start: 5e-5
   - If learning too fast (diverge): decrease to 1e-5
   - If learning too slow: increase to 1e-4
   - Typical range: [1e-5, 1e-3]

2. ENTROPY COEFFICIENT (entropy_coeff)
   - Start: 0.01
   - Too low (< 0.001): Agent memorizes → no generalization
   - Too high (> 0.1): Agent explores too much → no convergence
   - Typical range: [0.001, 0.1]

3. CLIP PARAMETER (clip_param)
   - Start: 0.3
   - Too low (< 0.1): Very conservative, slow learning
   - Too high (> 0.5): Can diverge
   - Typical range: [0.1, 0.5]

4. GAMMA (Discount factor)
   - 0.99: Good for long-horizon problems (traffic control)
   - 0.95: Shorter horizon, faster learning
   - 0.999: Very long horizon, might be too greedy

5. LAMBDA (GAE lambda)
   - 0.95-0.99: Standard, good bias-variance trade-off
   - Higher (0.99): Lower variance, higher bias
   - Lower (0.90): Higher variance, lower bias

6. BATCH SIZE
   - Small (2048): Faster update, noisier gradient
   - Large (8192): Slower update, cleaner gradient
   - SGD minibatch: Usually 1/4 to 1/32 of train_batch_size

7. NUMBER OF WORKERS
   - More workers = more parallel collection
   - Diminishing returns after CPU cores
   - Memory increases with workers

8. NETWORK SIZE (fcnet_hiddens)
   - Small [128, 128]: Fast, for simple tasks
   - Medium [256, 256]: Balanced (recommended)
   - Large [512, 512]: Slow, for complex tasks
   - Don't go too deep (> 3 layers) without reason
"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example of using different configurations:
    
    # In train_ppo.py, modify the create_ppo_config call:
    
    # Option 1: Use predefined config
    config = get_ppo_config_medium()
    
    # Option 2: Customize on the fly
    config = (
        get_ppo_config_medium()
        .training(lr=1e-4)  # Override learning rate
    )
    """
    pass
