import argparse
import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env

import utils

def train(agent, num_steps):
    """
    Purpose: Train agent in env, and then call eval function to evaluate policy
    """
    #Train agent
    agent.learn(total_timesteps = num_steps, log_interval = 10)

def eval_policy(model, env, num_eval_episodes: int, list_reward_per_episode = False):
    """
    Purpose: Evaluate policy on environment over num_eval_episodes and print results

    Args:
        Model: Stable baselines model
        Env: Gym environment for evaluation
        num_eval_episodes: (Int) number of episodes to evaluate policy
        list_reward_per_episode: (Boolean) Whether or not to return a list containing rewards per episode (instead of mean reward over all episodes)
    
    """
    mean_reward, std_reward = evaluate_policy(model, env, num_eval_episodes, return_episode_rewards = list_reward_per_episode)

    print("Test Results: ")
    print("Mean Reward: {:.3f}".format(mean_reward))
    print("Std Reward: {:.3f}".format(std_reward))

def get_agent(env, args):
    """
    Purpose: Import algo, policy and create agent

    Returns: Agent

    Exceptions: Raises exception if args.algo unknown (not needed b/c we filter in the parser, but I added it for modularity)
    """
    if args.algo == 'sac':
        from stable_baselines import SAC
        from stable_baselines.sac.policies import MlpPolicy as policy
        return SAC(policy, env, batch_size = args.batch_size, learning_starts = 30, verbose = 0, tensorboard_log = './rl_tensorboard_logs/')
    
     #I (Akash) still need to study PPO to understand it, I implemented b/c I know Joe's work used PPO
    elif args.algo == 'ppo':
        from stable_baselines import PPO2
        
        if(args.policy_type == 'mlp'):
            from stable_baselines.common.policies import MlpPolicy as policy
        
        elif(args.policy_type == 'lstm'):
            from stable_baselines.common.policies import MlpLstmPolicy as policy
        
        return PPO2(policy, env, verbose = 0, tensorboard_log = './rl_tensorboard_logs/')

    else:
        raise NotImplementedError('Algorithm {} not supported. :( '.format(args.algo))


def args_convert_bool(args):
    """
    Purpose: Convert args which are specified as strings (e.g. yesterday, energy) into boolean to work with environment
    """
    args.yesterday = utils.string2bool(args.yesterday)
    args.energy = utils.string2bool(args.energy)

def get_environment(args):
    """
    Purpose: Create environment for algorithm given by args. algo

    Args:
        args
    
    Returns: Environment with action space compatible with algo
    """
    #Convert string args (which are supposed to be bool) into actual boolean values
    args_convert_bool(args)

    #SAC only works in continuous environment
    if(args.algo == 'sac'):
        action_space_string = 'continuous'
    
    #For algos (e.g. ppo) which can handle discrete or continuous case
    #Note: PPO typically uses normalized environment (#TODO)
    else:
        convert_action_space_str = lambda s: 'continuous' if s == 'c' else 'multidiscrete'
        action_space_string = convert_action_space_str(args.action_space)
    
    if(args.env_id == 'hourly'):
        env_id = '_hourly-v0'
    elif(args.env_id == 'monthly'):
        env_id = '_monthly-v0'
    else:
        env_id = '-v0'


    socialgame_env = gym.make('gym_socialgame:socialgame{}'.format(env_id), action_space_string = action_space_string, response_type_string = args.response,
                    one_price = args.one_day, number_of_participants = args.num_players, yesterday_in_state = args.yesterday, energy_in_state = args.energy)
    
    #Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(socialgame_env)

    #Using env_fn so we can create vectorized environment. This allows us to check if NaNs are outputted by the agent and what caused the NaNs
    env_fn = lambda: socialgame_env
    venv = DummyVecEnv([env_fn])
    env = VecNormalize(venv)
    return env

def parse_args():
    """
    Purpose: Parse arguments to run script
    """
    parser = argparse.ArgumentParser(description='Arguments for running Stable Baseline RL Algorithms on SocialGameEnv')
    parser.add_argument('--env_id', help = 'Environment ID for Gym Environment', type=str, choices = ['v0', 'monthly'], default = 'v0')
    parser.add_argument('algo', help = 'Stable Baselines Algorithm', type=str, choices = ['sac', 'ppo'] )
    parser.add_argument('--batch_size', help = 'Batch Size for sampling from replay buffer', type=int, default = 5, choices = [i for i in range(1,30)])
    parser.add_argument('--num_steps', help = 'Number of timesteps to train algo', type = int, default = 1000000)
    #Note: only some algos (e.g. PPO) can use LSTM Policy the feature below is for future testing
    parser.add_argument('--policy_type', help = 'Type of Policy (e.g. MLP, LSTM) for algo', default = 'mlp', choices = ['mlp', 'lstm'])
    parser.add_argument('--action_space', help = 'Action Space for Algo (only used for algos that are compatable with both discrete & cont', default = 'c',
                        choices = ['c','d'])
    parser.add_argument('--response',help = 'Player response function (l = linear, t = threshold_exponential, s = sinusoidal', type = str, default = 'l',
                        choices = ['l','t','s'])
    parser.add_argument('--one_day', help = 'Specific Day of the year to Train on (default = None, train over entire yr)', type=int,default = -1, 
                        choices = [i for i in range(-1, 366)])
    parser.add_argument('--num_players', help = 'Number of players ([1, 20]) in social game', type = int, default = 1, choices = [i for i in range(1, 21)])
    parser.add_argument('--yesterday', help = 'Whether to include yesterday in state (default = F)', type = str, default = 'F', choices = ['T', 'F'])
    parser.add_argument('--energy', help = 'Whether to include energy in state (default = F)', type=str, default = 'F', choices = ['T', 'F'])

    args = parser.parse_args()

    return args

def main():
    #Get args
    args = parse_args()

    #Print args for reference
    print(args)
    

    #Create environment
    env = get_environment(args)

    #Create Agent
    model = get_agent(env, args)
    
    #Train algo, (logging through Tensorboard)
    print("Beginning Testing!")
    train(model,args.num_steps)
    print("Training Completed! View TensorBoard logs at rl_tensorboard_logs/")

    #Print evaluation of policy
    print("Beginning Evaluation")
    #TODO: Define evaluation env (pointless if eval_env = env)
    eval_env = env
    eval_policy(model, eval_env, 10)

if __name__ == '__main__':
    main()