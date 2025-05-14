import gymnasium as gym
import numpy as np
import torch
import os
import wandb

class UCBalgorithm:
    def __init__(self, num_arms, burn_in = 1, min_range = -float("inf"), max_range = float("inf"), epsilon = 0, delta = .1):
        self.num_arms = num_arms
        self.mean_estimators = [0 for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.reward_sums = [0 for _ in range(num_arms)]
        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range
        self.epsilon = epsilon
        self.delta = delta
        self.global_time_step = 0

    def update_arm_statistics(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.reward_sums[arm_index] += reward
        self.mean_estimators[arm_index] = self.reward_sums[arm_index]/self.counts[arm_index] 
        self.global_time_step += 1

    def get_ucb_arm(self, confidence_radius, arm_info = None ):


        if sum(self.counts) <=  self.burn_in:
            #print("HERE")
            ucb_arm_index = random.choice(range(self.num_arms))
            ucb_arm_value = self.max_range
            lcb_arm_value = self.min_range
        else:
            ucb_bonuses = [confidence_radius*np.sqrt(np.log((self.global_time_step+1.0)/self.delta)/(count + .0000000001)) for count in self.counts ]
            ucb_arm_values = [min(self.mean_estimators[i] + ucb_bonuses[i], self.max_range) for i in range(self.num_arms)]
            #ucb_arm_index = np.argmax(ucb_arm_values)
            ucb_arm_values = np.array(ucb_arm_values)
            lcb_arm_values = [max(self.mean_estimators[i] - ucb_bonuses[i], self.min_range) for i in range(self.num_arms)]

            if np.random.random() <= self.epsilon:
                ucb_arm_index = np.random.choice(range(self.num_arms))
            else:
                ucb_arm_index = np.random.choice(np.flatnonzero(ucb_arm_values == ucb_arm_values.max()))
            
            ucb_arm_value = ucb_arm_values[ucb_arm_index]
            lcb_arm_value = lcb_arm_values[ucb_arm_index]
        return ucb_arm_index

class UCBHyperparam:
    wandb_project_name: str = "UCB_Hyperparams"

    def __init__(self, m, burn_in=1, confidence_radius=2, min_range=0, max_range=1, epsilon=0, track=False, wandb_project_name="UCB_Hyperparams", wandb_entity=None):
        self.ucb_algorithm = UCBalgorithm(m, burn_in=burn_in, min_range=min_range, max_range=max_range, epsilon=epsilon)
        self.m = m 
        self.confidence_radius = confidence_radius
        self.burn_in = burn_in
        self.T = 1
        self.base_probas = np.ones(self.m) / self.m
        self.track = track

        if self.track:
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                name="UCBHyperparam_Run",
                monitor_gym=True,
                save_code=True
            )

    def sample_base_index(self):
        index = self.ucb_algorithm.get_ucb_arm(self.confidence_radius)
        if self.T <= self.burn_in:
            self.base_probas = np.ones(self.m) / self.m
        else:
            self.base_probas = np.zeros(self.m)
            self.base_probas[index] = 1
        self.T += 1
        return index

    def get_distribution(self):
        return self.base_probas

    def update_distribution(self, arm_idx, reward, more_info=dict([])):
        self.ucb_algorithm.update_arm_statistics(arm_idx, reward)

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1)
learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
ucb_hyperparam = UCBHyperparam(m=len(learning_rates), track=True, wandb_project_name="UCB_Hyperparams")

num_timesteps = 1000000
for episode in range(num_timesteps):
    obs, info = env.reset()
    done = False
    total_reward = 0
    arm_index = ucb_hyperparam.sample_base_index()
    learning_rate = learning_rates[arm_index] 

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        obs = next_obs

    ucb_hyperparam.update_distribution(arm_index, total_reward)

    if ucb_hyperparam.track:
        wandb.log({"Episode": episode, "Total Reward": total_reward, "Learning Rate": learning_rate})

env.close()



class UCBalgorithm:
    def __init__(self, num_arms, burn_in = 1, min_range = -float("inf"), max_range = float("inf"), epsilon = 0, delta = .1):
        self.num_arms = num_arms
        self.mean_estimators = [0 for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.reward_sums = [0 for _ in range(num_arms)]
        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range
        self.epsilon = epsilon
        self.delta = delta
        self.global_time_step = 0

    def update_arm_statistics(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.reward_sums[arm_index] += reward
        self.mean_estimators[arm_index] = self.reward_sums[arm_index]/self.counts[arm_index] 
        self.global_time_step += 1

    def get_ucb_arm(self, confidence_radius, arm_info = None ):


        if sum(self.counts) <=  self.burn_in:
            #print("HERE")
            ucb_arm_index = random.choice(range(self.num_arms))
            ucb_arm_value = self.max_range
            lcb_arm_value = self.min_range
        else:
            ucb_bonuses = [confidence_radius*np.sqrt(np.log((self.global_time_step+1.0)/self.delta)/(count + .0000000001)) for count in self.counts ]
            ucb_arm_values = [min(self.mean_estimators[i] + ucb_bonuses[i], self.max_range) for i in range(self.num_arms)]
            #ucb_arm_index = np.argmax(ucb_arm_values)
            ucb_arm_values = np.array(ucb_arm_values)
            lcb_arm_values = [max(self.mean_estimators[i] - ucb_bonuses[i], self.min_range) for i in range(self.num_arms)]

            if np.random.random() <= self.epsilon:
                ucb_arm_index = np.random.choice(range(self.num_arms))
            else:
                ucb_arm_index = np.random.choice(np.flatnonzero(ucb_arm_values == ucb_arm_values.max()))
            
            ucb_arm_value = ucb_arm_values[ucb_arm_index]
            lcb_arm_value = lcb_arm_values[ucb_arm_index]
        return ucb_arm_index
