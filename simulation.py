from agents import Person
from rewards import Reward
import pandas as pd
import utils

class Office():
	def __init__():

		self._start_timestamp = pd.Timestamp(year=2018,
                                         month=6,
                                         day=1,
                                         hour=0,
                                         minute=0)		

		self._end_timestamp = pd.Timestamp(year=2018,
                                         month=7,
                                         day=1,
                                         hour=0,
                                         minute=0)
		self._timestep= _start_timestamp

    def _create_agents_and_controller(self):
        """Initialize the market agents

        Args:
          None

        Return:
          agent_dict: dictionary of the agents
        """

        # controller initialize -- hyperparameters 
        # different types of controllers, and down the line, pick the one we use.

        controller = Controller()
        controller.initialize(hyperparameters = hyperparameters) 


        baseline_energy1 = pd.read_csv("wg1.txt", sep = "\t")
        baseline_energy2 = pd.read_csv("wg2.txt", sep = "\t")
        baseline_energy3 = pd.read_csv("wg3.txt", sep = "\t")

        be1 = change_wg_to_diff(baseline_energy1)
        be2 = change_wg_to_diff(baseline_energy2)
        be3 = change_wg_to_diff(baseline_energy3)

        players_dict = {}

        players_dict['player_0'] = Person(be1, points_multiplier = 1)
        players_dict['player_1'] = Person(be1, points_multiplier = 2)
        players_dict['player_2'] = Person(be1, points_multiplier = 4)
        players_dict['player_3'] = Person(be2, points_multiplier = 1)
        players_dict['player_3'] = Person(be2, points_multiplier = 2)
        players_dict['player_4'] = Person(be2, points_multiplier = 2.5)
        players_dict['player_7'] = Person(be3, points_multiplier = 1)
        players_dict['player_8'] = Person(be3, points_multiplier = .5)

        return players_dict


    def step(self, prices, controller):
    	""" 
    	- get what the controller would output
    	- controller.update to pass in reward
    	- controller initiatlization 
    	"""

    	# get controllers points 

    	controllers_points = controller.get_points(prices)

    	# get the people's energy consumption 

    	end = False 

    	while not end: 
    		energy_dict = {}
    		rewards_dict = {}
    		for player_name in players_dict:
    			player = players_dict.get(player_name)
    			player_energy = player.energy_output_simple_linear(controllers_points)
    			energy_dict[player_name] = player_energy
    			player_min_reward = player.get_min_demand()
    			player_max_reward = player.get_max_demand()
    			player_reward = Reward(player_energy, prices, player_min_demand, player_max_demand)
    			player_ideal_demands = player_reward.ideal_use_calculation()
    			distance_from_ideal = player_reward.distance_from_ideal(player_ideal_demands)
    			rewards_dict[player_name] = distance_from_ideal

    		total_distance = sum(rewards_dict.values())

    		# reward goes back into controller as controller update 
    		controller.update(reward = total_distance)

    		self._timestep = _timestep + time_interval

    		if self._timestep>self._end_timestamp:
    			end = True




    	



