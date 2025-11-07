"""Demo: Event-driven communication between Env and SUMO Simulator.

This demo shows how to use EventBus to decouple the RL environment
from the SUMO simulator, enabling:
- Asynchronous communication
- Multiple environments with one simulator
- Easy testing and debugging
- Distributed execution
"""

import ray
import time
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sim.event_bus import EventBus
from sim.event_driven_simulator import EventDrivenSumoSimulator
from environment.drl_algo.event_driven_env import EventDrivenEnvironment


def demo_eventbus_communication():
    """Demo basic EventBus communication between Env and Simulator."""
    
    print("=" * 70)
    print("Demo: EventBus-based Env ↔ Simulator Communication")
    print("=" * 70)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # 1. Create EventBus
        print("\n📡 Creating EventBus...")
        bus = EventBus.remote()
        
        # 2. Create Simulator
        print("\n🚦 Creating SUMO Simulator...")
        
        # SUMO configuration
        net_file = str(Path(__file__).parent.parent / "network" / "grid4x4" / "grid4x4.net.xml")
        route_file = str(Path(__file__).parent.parent / "network" / "grid4x4" / "grid4x4.rou.xml")
        
        sumo_config = {
            "net_file": net_file,
            "route_file": route_file,
            "use_gui": False,
            "num_seconds": 1000,
            "delta_time": 5,
        }
        
        simulator = EventDrivenSumoSimulator.remote(
            bus=bus,
            simulator_id="sumo_sim_1",
            **sumo_config
        )
        
        # Subscribe simulator to EventBus
        ray.get(simulator.subscribe_to_bus.remote(simulator))
        
        # 3. Create Environment
        print("\n🌍 Creating Environment...")
        env = EventDrivenEnvironment.remote(
            bus=bus,
            env_id="env_1",
            simulator_id="sumo_sim_1",
            timeout=10.0
        )
        
        # Subscribe env to EventBus
        ray.get(env.subscribe_to_bus.remote(env))
        
        # 4. Initialize Simulator
        print("\n🔧 Initializing Simulator...")
        init_result = ray.get(simulator.initialize.remote())
        print(f"   Init result: {init_result.get('success', False)}")
        
        # Wait for init message to propagate
        time.sleep(1)
        
        # 5. Reset Environment (through EventBus)
        print("\n🔄 Resetting Environment (via EventBus)...")
        observations = ray.get(env.reset.remote(seed=42))
        print(f"   Received {len(observations)} initial observations")
        print(f"   Agent IDs: {list(observations.keys())[:3]}...")  # Show first 3
        
        # 6. Run a few steps
        print("\n🎮 Running simulation steps (via EventBus)...")
        
        for step_num in range(5):
            # Create dummy actions (action 0 for all agents)
            actions = {agent_id: 0 for agent_id in observations.keys()}
            
            # Step environment (sends action via EventBus)
            obs, rewards, dones, info = ray.get(env.step.remote(actions))
            
            print(f"\n   Step {step_num + 1}:")
            print(f"      - Observations: {len(obs)} agents")
            print(f"      - Total reward: {sum(rewards.values()):.2f}")
            print(f"      - Done: {dones.get('__all__', False)}")
            print(f"      - Sim time: {info.get('step', 0)}")
            
            if dones.get("__all__", False):
                print("      ⚠️ Episode ended")
                break
        
        # 7. Check EventBus topics
        print("\n📋 EventBus Topics:")
        topics = ray.get(bus.list_topics.remote())
        for topic, subscribers in topics.items():
            print(f"   - {topic}: {subscribers}")
        
        # 8. Close environment
        print("\n🛑 Closing Environment...")
        ray.get(env.close.remote())
        
        print("\n✅ Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown Ray
        print("\n🔚 Shutting down Ray...")
        ray.shutdown()


def demo_multi_env_single_simulator():
    """Demo: Multiple environments sharing one simulator via EventBus."""
    
    print("=" * 70)
    print("Demo: Multiple Environments + Single Simulator")
    print("=" * 70)
    
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create EventBus
        bus = EventBus.remote()
        
        # Create one simulator
        net_file = str(Path(__file__).parent.parent / "network" / "grid4x4" / "grid4x4.net.xml")
        route_file = str(Path(__file__).parent.parent / "network" / "grid4x4" / "grid4x4.rou.xml")
        
        simulator = EventDrivenSumoSimulator.remote(
            bus=bus,
            simulator_id="shared_sim",
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=1000,
        )
        
        ray.get(simulator.subscribe_to_bus.remote(simulator))
        ray.get(simulator.initialize.remote())
        
        # Create multiple environments
        envs = []
        for i in range(3):
            env = EventDrivenEnvironment.remote(
                bus=bus,
                env_id=f"env_{i}",
                simulator_id="shared_sim",
            )
            ray.get(env.subscribe_to_bus.remote(env))
            envs.append(env)
        
        print(f"\n✅ Created {len(envs)} environments sharing one simulator")
        print("   Each environment communicates via EventBus")
        
        # Note: In real usage, you'd need to implement proper
        # turn-taking or scheduling for multiple envs
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    print("\n🚀 Starting EventBus Demo\n")
    
    # Run basic demo
    demo_eventbus_communication()
    
    # Uncomment to run multi-env demo
    # demo_multi_env_single_simulator()
