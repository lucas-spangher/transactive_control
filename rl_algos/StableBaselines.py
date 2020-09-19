import argparse
import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env

import numpy as np
import tensorflow as tf

import utils

def train(agent, num_steps):
    """
    Purpose: Train agent in env, and then call eval function to evaluate policy
    """
    #Train agent
    agent = agent.learn(total_timesteps = num_steps, log_interval = 1)
    return agent

def eval_policy_helper(args, model, num_eval_episodes, response=None,list_reward_per_episode=False):
    """
    Purpose: Helper function to compute the evaluation loop over environment

    Returns: Rewards from all episodes
    """
    rewards = np.zeros(num_eval_episodes)

    for episode_i in range(num_eval_episodes):
        env = get_environment(args, eval=True, response=response)
        curr_reward, _ = evaluate_policy(model, env, 1, return_episode_rewards=list_reward_per_episode)
        
        print("Episode Reward: {:.3f}".format(curr_reward[0][0]))

        rewards[episode_i] = curr_reward[0][0]
    
    #TODO: Improve text formatting
    np.savetxt('eval_rewards.txt',rewards)

    return rewards

def eval_policy(model, args, num_eval_episodes: int, list_reward_per_episode = False):
    """
    Purpose: Evaluate policy on environment over num_eval_episodes and print results

    Args:
        Model: Stable baselines model
        Args: Args (so we can instantiate eval env here)
        num_eval_episodes: (Int) number of episodes to evaluate policy
        list_reward_per_episode: (Boolean) Whether or not to return a list containing rewards per episode (instead of mean reward over all episodes)
    
    """
    #Since evaluate_policy goes over the first num_eval_episodes, if we train on one_yr we will always evaluate against the same n days
    if args.one_price == 0:
        args.one_price = -1

    if args.random:
        #If using DR we evaluate over linear, sin, and thresh env. 

        print("\n Evaluating Linear Env \n")
        eval_rewards = eval_policy_helper(args, model, 10, 'l', True)
        print("Rewards ")
        print(eval_rewards)
        print("Mean Linear Reward: {:.3f}".format(np.mean(eval_rewards)))
        print("Standard Deviation  of Linear Reward: {:.3f}".format(np.std(eval_rewards)))
        print("*****"*30)
        
        print("\n Evaluating Sin Env")
        eval_rewards = eval_policy_helper(args,model, 10, 's', True)
        print("Rewards ")
        print(eval_rewards)
        print("Mean Sin. Reward: {:.3f}".format(np.mean(eval_rewards)))
        print("Standard Deviation of Sin. Reward: {:.3f}".format(np.std(eval_rewards)))
        print("*****"*30)
        
        eval_rewards = eval_policy_helper(args, model, 10, 't', True)
        print("Rewards ")
        print(eval_rewards)
        print("Mean Thresh. Reward: {:.3f}".format(np.mean(eval_rewards)))
        print("Standard Deviation of Thresh. Reward: {:.3f}".format(np.std(eval_rewards)))
        print("*****"*30)
    
    else:

        eval_rewards = eval_policy_helper(args, model, 10, list_reward_per_episode = True)
        print("Rewards ")
        print(eval_rewards)
        print("Mean Reward: {:.3f}".format(np.mean(eval_rewards)))
        print("Standard Deviation of Reward: {:.3f}".format(np.std(eval_rewards)))
        print("*****"*30)

def get_agent(env, args):
    """
    Purpose: Import algo, policy and create agent

    Returns: Agent

    Exceptions: Raises exception if args.algo unknown (not needed b/c we filter in the parser, but I added it for modularity)
    """
    #TODO: DIFFERENTIATE DR LOGS!

    if args.algo == 'sac':
        from stable_baselines import SAC
        from stable_baselines.sac.policies import MlpPolicy as policy
        return SAC(policy, env, batch_size = args.batch_size, learning_starts = 30, verbose = 0, tensorboard_log = './rl_tensorboard_logs/')
    
    elif args.algo == 'ppo':
        from stable_baselines import PPO2
        
        if(args.policy_type == 'mlp'):
            from stable_baselines.common.policies import MlpPolicy as policy
        
        elif(args.policy_type == 'lstm'):
            from stable_baselines.common.policies import MlpLstmPolicy as policy
        
        return PPO2(policy, env,  nminibatches=1, verbose = 0, tensorboard_log = './rl_tensorboard_logs/')

    else:
        raise NotImplementedError('Algorithm {} not supported. :( '.format(args.algo))


def args_convert_bool(args):
    """
    Purpose: Convert args which are specified as strings (e.g. yesterday, energy) into boolean to work with environment
    """
    if not isinstance(args.yesterday, (bool)):
        args.yesterday = utils.string2bool(args.yesterday)
    if not isinstance(args.energy, (bool)):
        args.energy = utils.string2bool(args.energy)
    if not isinstance(args.random, (bool)):
        args.random = utils.string2bool(args.random)

def get_environment(args, eval=False, response = None):
    """
    Purpose: Create environment for algorithm given by args. algo

    Args:
        args
        eval: Boolean denoting whether or not to return evaluation env 
        response: For setting response for eval env^

    Returns: Environment with action space compatible with algo
    """
    #Convert string args (which are supposed to be bool) into actual boolean values
    args_convert_bool(args)

    #SAC only works in continuous environment
    if(args.algo == 'sac'):
        action_space_string = 'continuous'
    
    #For algos (e.g. ppo) which can handle discrete or continuous case

    else:
        convert_action_space_str = lambda s: 'continuous' if s == 'c' else 'multidiscrete'
        action_space_string = convert_action_space_str(args.action_space)
    


    if args.env_id == 'hourly':
        env_id = '_hourly-v0'
    elif args.env_id == 'monthly':
        env_id = '_monthly-v0'
    else:
        env_id = '-v0'

    if eval:
        if(response is None):
            response = args.response

        socialgame_env = gym.make('gym_socialgame:socialgame{}'.format(env_id), 
            action_space_string = action_space_string, 
            response_type_string = response,
            one_price = args.one_price, 
            number_of_participants = args.num_players, 
            yesterday_in_state = args.yesterday, 
            energy_in_state = args.energy)

    else:
        if args.random:
            socialgame_env = gym.make('gym_socialgame:socialgame_dr-v0',
            action_space_string = action_space_string, 
            response_type_string = args.response,
            one_price = args.one_price, 
            number_of_participants = args.num_players, 
            yesterday_in_state = args.yesterday, 
            energy_in_state = args.energy,
            low = args.low,
            high = args.high,
            distr = args.distr)
        
        else:
            socialgame_env = gym.make('gym_socialgame:socialgame{}'.format(env_id), 
                action_space_string = action_space_string, 
                response_type_string = args.response,
                one_price = args.one_price, 
                number_of_participants = args.num_players, 
                yesterday_in_state = args.yesterday, 
                energy_in_state = args.energy)
                    
    #Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(socialgame_env)

    #Using env_fn so we can create vectorized environment.
    env_fn = lambda: socialgame_env
    venv = DummyVecEnv([env_fn])
    env = VecNormalize(venv)
    return env

def parse_args():
    """
    Purpose: Parse arguments to run script
    """

    parser = argparse.ArgumentParser(description='Arguments for running Stable Baseline RL Algorithms on SocialGameEnv')
    
    #Algo Arguments
    parser.add_argument('--env_id', help = 'Environment ID for Gym Environment', type=str, choices = ['v0', 'monthly'], default = 'v0')
    parser.add_argument('algo', help = 'Stable Baselines Algorithm', type=str, choices = ['sac', 'ppo'] )
    parser.add_argument('--batch_size', help = 'Batch Size for sampling from replay buffer', type=int, default = 5, choices = [i for i in range(1,30)])
    parser.add_argument('--num_steps', help = 'Number of timesteps to train algo', type = int, default = 1000000)
    #Note: only some algos (e.g. PPO) can use LSTM Policy the feature below is for future testing
    parser.add_argument('--policy_type', help = 'Type of Policy (e.g. MLP, LSTM) for algo', default = 'mlp', choices = ['mlp', 'lstm'])

    # Basic Env Arguments
    parser.add_argument('--action_space', help = 'Action Space for Algo (only used for algos that are compatable with both discrete & cont', default = 'c',
                        choices = ['c','d'])
    parser.add_argument('--response',help = 'Player response function (l = linear, t = threshold_exponential, s = sinusoidal', type = str, default = 'l',
                        choices = ['l','t','s'])
    parser.add_argument('--one_price', help = 'Specific Day of the year to Train on (default = 0, train over entire yr)', type=int,default = 0, 
                        choices = [i for i in range(-1, 366)])
    parser.add_argument('--num_players', help = 'Number of players ([1, 20]) in social game', type = int, default = 10, choices = [i for i in range(1, 21)])
    parser.add_argument('--yesterday', help = 'Whether to include yesterday in state (default = F)', type = str, default = 'F', choices = ['T', 'F'])
    #TODO: Make energy default = True
    parser.add_argument('--energy', help = 'Whether to include energy in state (default = F)', type=str, default = 'F', choices = ['T', 'F'])

    #DR Env Arguments
    parser.add_argument('--random', help='Whether to use domain randomization (DR), default = F', type=str, default='F', choices=['T','F'])
    parser.add_argument('--low', help='Lower bound for uniform noise to response function, default = 0', type=int, default=0)
    parser.add_argument('--high', help='Upper bound for uniform noise to response function, default = 50', type=int, default = 50)
    parser.add_argument('--distr', help='Distribution for noise. Currently only Uniform distr. supported.', type=str, default='U', choices=['U','G'])

    #Get args  
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
    print("Beginning Training!")
    model = train(model,args.num_steps)
    print("Training Completed! View TensorBoard logs at rl_tensorboard_logs/")

    #Print evaluation of policy
    print("Beginning Evaluation")
    eval_policy(model, args, 10, list_reward_per_episode= True)

if __name__ == '__main__':
    main()