import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        for r, d in zip(rewards, dones):
            self.current_rewards.append(r)
            if d:
                ep_reward = sum(self.current_rewards)
                self.episode_rewards.append(ep_reward)
                self.current_rewards = []
        return True

class MaskedObservationWrapper(ObservationWrapper):
    def __init__(self, env, mask=None):
        super().__init__(env)
        self.monitor = Monitor(env)
        self.original_obs_space = env.observation_space
        self.mask = mask if mask is not None else np.ones(self.original_obs_space.shape)

    def observation(self, obs):
        return obs * self.mask

    def update_mask(self, new_mask):
        self.mask = new_mask

def evaluate(model):
    env = Monitor(gym.make("CartPole-v1"))
    mean, std = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    return mean, std

def plot_results(results):
    labels = [f"Run {i+1}" for i in range(len(results["Baseline PPO"]))]
    variants = ["Baseline PPO", "Static Mask", "Feedback Mask"]
    mean_rewards = {v: [r[0] for r in results[v]] for v in variants}
    std_devs = {v: [r[1] for r in results[v]] for v in variants}
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, variant in enumerate(variants):
        ax.bar(x + i * width, mean_rewards[variant], width, label=variant,
               yerr=std_devs[variant], capsize=5)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Evaluation Results by Agent Variant and Run')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

# Simulated feedback: apply feedback at multiple time steps and adjust soft attention mask smoothly
feedback_schedule = {
    2000: np.array([1.0, 1.0, 0.5, 1.0]),
    4000: np.array([1.0, 1.0, 0.2, 1.0]),
    6000: np.array([1.0, 1.0, 0.0, 1.0]),
}

def train_agent(name, mask=None, feedback_steps=None, seed=0):
    env = MaskedObservationWrapper(gym.make("CartPole-v1"), mask=mask)
    env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    callback = RewardLogger()

    total_timesteps = 10000
    feedback_interval = 1000

    for step in range(0, total_timesteps, feedback_interval):
        model.learn(total_timesteps=feedback_interval, reset_num_timesteps=False, callback=callback)
        if feedback_steps:
            if step in feedback_schedule:
                env.envs[0].env.update_mask(feedback_schedule[step])

    model.save(f"models/{name}_seed{seed}.zip")
    return model, callback.episode_rewards

def plot_training_curves(training_rewards):
    for variant in training_rewards.keys():
        plt.figure(figsize=(8, 4))
        for i, rewards in enumerate(training_rewards[variant]):
            plt.plot(rewards, label=f"{variant} - Run {i+1}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Training Reward Over Time ({variant})")
        plt.legend()

def compute_convergence_episodes(training_rewards, threshold=450):
    convergence_results = {}
    for variant, runs in training_rewards.items():
        convergence_results[variant] = []
        for rewards in runs:
            for i, r in enumerate(rewards):
                if r >= threshold:
                    convergence_results[variant].append(i + 1)
                    break
            else:
                convergence_results[variant].append(None)
    return convergence_results

def print_convergence_table(convergence_results):
    print("=== Convergence Episodes (Threshold: 450) ===")
    for variant, runs in convergence_results.items():
        print(f"{variant}:")
        for i, val in enumerate(runs):
            status = f"{val} episodes" if val is not None else "❌ did not converge"
            print(f"  Run {i+1}: {status}")
        converged = [v for v in runs if v is not None]
        if converged:
            avg = np.mean(converged)
            print(f"Avg (converged only): {avg:.1f} episodes")
        print()

def display_performance_table(results):
    print("Variant","Seed 1","Seed 2","Seed 3")
    for variant, rewards in results.items():
        print(f"{variant}:", end=" ")
        for reward in rewards:
            print(f"{reward}",end=" ")
        print()

def run_experiments():
    os.makedirs("models", exist_ok=True)
    seeds = [42, 100, 123]
    results = {"Baseline PPO": [], "Static Mask": [], "Feedback Mask": []}
    training_rewards = {"Baseline PPO": [], "Static Mask": [], "Feedback Mask": []}

    for seed in seeds:
        model_base, rewards_base = train_agent("ppo_baseline", mask=np.ones(4), seed=seed)
        reward, std = evaluate(model_base)
        results["Baseline PPO"].append((reward, std))
        training_rewards["Baseline PPO"].append(rewards_base)

        static_mask = np.array([0, 1, 1, 1])
        model_static, rewards_static = train_agent("ppo_static_mask", mask=static_mask, seed=seed)
        reward, std = evaluate(model_static)
        results["Static Mask"].append((reward, std))
        training_rewards["Static Mask"].append(rewards_static)

        model_feedback, rewards_feedback = train_agent("ppo_feedback_mask", mask=np.ones(4), feedback_steps=feedback_schedule, seed=seed)
        reward, std = evaluate(model_feedback)
        results["Feedback Mask"].append((reward, std))
        training_rewards["Feedback Mask"].append(rewards_feedback)

    display_performance_table(results)
    convergence_results = compute_convergence_episodes(training_rewards, threshold=450)
    print_convergence_table(convergence_results)
    plot_results(results)
    plot_training_curves(training_rewards)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiments()
