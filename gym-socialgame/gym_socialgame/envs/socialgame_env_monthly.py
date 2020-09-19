import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

class SocialGameEnvMonthly(SocialGameEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string = "continuous", response_type_string = "l", number_of_participants = 10,
                one_price = 0, energy_in_state = True, yesterday_in_state = False):
        """
        SocialGameEnv for an agent determining incentives in a social game. 
        
        Note: 30-step trajectory (i.e. agent submits a 10-dim vector containing incentives for each hour (8AM - 5PM) each day. 
            Then, environment advances one-day and episode completes after one month.

        Args:
            action_space_string: (String) either "continuous", or "multidiscrete"
            response_type_string: (String) either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_price: (Int) in range [-1,12] denoting which fixed day to train on . 
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,12] = Month of the Year
            energy_in_state: (Boolean) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (Boolean) denoting whether (or not) to append yesterday's price signal to the state

        """

        #Checking that one_price is valid 
        assert isinstance(one_price, int), "Variable one_price is not of type Int. Instead got type {}".format(type(one_price))
        assert 13 > one_price and one_price > -2, "Variable one_price out of range [-1,12]. Got one_price = {}".format(one_price)

        super().__init__(action_space_string=action_space_string,
                        response_type_string=response_type_string,
                        number_of_participants=number_of_participants,
                        one_price=one_price,
                        energy_in_state=energy_in_state, 
                        yesterday_in_state=yesterday_in_state)
        
        #Maintaining Reward over the month
        self.reward = 0
    
    def _find_one_price(self, one_price: int):
        """
        Purpose: Helper function to find one_price to train on (if applicable)

        Args:
            one_price: (Int) in range [-1,12]

        Returns:
            0 if one_price = 0
            one_price if one_price in range [1,12]
                    else random_number[1,12] if one_price = -1
        """
        
        if(one_price == -1):
            return np.random.randint(1, high=13)
        
        else:
            return one_price

 
    def _get_prices(self):
        """
        Purpose: Get grid price signals for the entire year (using past data from a building in Los Angeles as reference)

        Args:
            None
            
        Returns: Array containing 365 price signals, where array[day_number] = grid_price for day_number from 8AM - 5PM

        """

        all_prices = []
        if self.one_price != 0:
            # If one_price we repeat the price signals from a fixed month
            # Tweak one_price Price Signal HERE
            month = self.one_price - 1
            for i in range(1, 31):
                price = price_signal(30 * month + i, type_of_DR="time_of_use")
                print(price)
                price = np.array(price[8:18])
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)
            print(all_prices)
            assert 3==2
            all_prices = all_prices * 13 #Doing times 13 just in case, we loop around 365 days so it shouldn't be a concern

        else:
            for day in range(365):  
                price = price_signal(day + 1, type_of_DR="real_time_pricing")
                price = np.array(price[8:18])
                # put a floor on the prices so we don't have negative prices
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)

        
        return all_prices

    

    def step(self, action):
        """
        Purpose: Takes a step in the environment 

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)
        
        Returns: 
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        """
        #Checking that action is valid; If not, we clip (OpenAI algos don't take into account action space limits so we must do it ourselves)
        if(not self.action_space.contains(action)):
            action = np.asarray(action)
            if(self.action_space_string == 'continuous'):
                action = np.clip(action, 0, 10)

            elif(self.action_space_string == 'multidiscrete'):
                action = np.clip(action, 0, 2) 

        prev_price = self.prices[self.day]

        points = self._points_from_action(action)

        energy_consumptions = self._simulate_humans(points)
        
        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]
        
        #Getting reward
        self.reward += self._get_reward(prev_price, energy_consumptions)

        #Advancing day
        self.day = (self.day + 1) % 365
        self.cur_iter += 1

        #Getting next observation
        observation = self._get_observation()

        #Setting done and return reward
        if self.cur_iter % 30 == 0:
            done = True
            #TODO: To normalize or not to normalize?
            reward = self.reward

        else:
            done = False
            reward = 0.0
        
        #Setting info for baselines compatibility (no relevant info for us)
        info = {}

        return observation, reward, done, info
    
     #Keeping reset, render, close for clarity sake
    def reset(self):
        """ Resets the environment to day 0 (of yr or month depending on one_price init) """ 
        #Currently resetting based on current day to work with StableBaselines

        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
