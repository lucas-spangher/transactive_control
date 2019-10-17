from agents import Person
from rewards import Reward
import pandas as pd
import utils

class Office():
	def __init__():

		self._start_timestamp = pd.Timestamp(year=2018,
                                         month=6,
                                         day=1,
                                         hour=4,
                                         minute=30)
		self._timestep= _start_timestamp

    def _create_agents(self):
        """Initialize the market agents

        Args:
          None

        Return:
          agent_dict: dictionary of the agents
        """

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


    def step(self, controllers_points):

    	while not end:
    		energy_dict = {}
    		rewards_dict = {}
    		for player in players_dict:
    			player_energy = player.energy_output_simple_linear(controllers_points)
    			energy_dict[player] = player_energy
    			player_reward = Reward(player_energy, prices, min_demand, max_demand)
    			rewards_dict[player] =
    		self._timestep = _timestep + time_interval
