from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
import gym

env = gym.make('gym_socialgame:socialgame-v0')

model = SAC(MlpPolicy, env, verbose=1, learning_starts = 30, batch_size = 2, tensorboard_log="./rl_tensorboard_logs/")
model.learn(total_timesteps=1e6, log_interval=10)

