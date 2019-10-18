from agents import Person
from reward import Reward
from controller import BaseController
import pandas as pd
from utils import *
import csv  
import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.pyplot as plt

class Office():
	def __init__(self):
		self._start_timestamp = pd.Timestamp(year=2012,
                                         month=10,
                                         day=30,
                                         hour=0,
                                         minute=0)		
		self._end_timestamp = pd.Timestamp(year=2012,
                                         month=12,
                                         day=31,
                                         hour=0,
                                         minute=0)
		self._timestep= self._start_timestamp
		self._time_interval = timedelta(days=1)
		self.players_dict = self._create_agents()
		self.controller = self._create_controller()

	def _create_agents(self):
		"""Initialize the market agents
			Args:
			  None

			Return:
			  agent_dict: dictionary of the agents
        """ 	

		print("creating agents")
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

	def _create_controller(self):
		print("creating controller")
		# controller initialize -- hyperparameters 
		# different types of controllers, and down the line, pick the one we use.
		# controller.initialize(hyperparameters = hyperparameters) 

		controller = BaseController()

		return controller

	def get_timestep(self):
		return self._timestep

	def step(self, prices):
		""" 
		- get what the controller would output
		- controller.update to pass in reward
		- controller initiatlization 
		"""

		# get controllers points 
		controller = self.controller
		controllers_points = controller.get_points(prices)

		end = False 

		energy_dict = {}
		rewards_dict = {}
		for player_name in self.players_dict:
			
			# get the points output from players
			player = self.players_dict.get(player_name)
			player_energy = player.energy_output_simple_linear(controllers_points)
			energy_dict[player_name] = player_energy

			# get the reward from the player's output
			player_min_demand = player.get_min_demand()
			player_max_demand = player.get_max_demand()
			player_reward = Reward(player_energy, prices, player_min_demand, player_max_demand)
			player_ideal_demands = player_reward.ideal_use_calculation()
			distance_from_ideal = player_reward.neg_distance_from_ideal(player_ideal_demands)
			rewards_dict[player_name] = distance_from_ideal

		total_distance = sum(rewards_dict.values())

		# reward goes back into controller as controller update 

		# controller.update(reward = total_distance)

		self._timestep = self._timestep + self._time_interval

		if self._timestep>self._end_timestamp:
			end = True

		return total_distance, end

	def price_signal(self, day = 45):

		"""
		Utkarsha's work on price signal from a building with demand and solar 

		Input: Day = an int signifying a 24 hour period. 365 total, all of 2012, start at 1.
		Output: netdemand_price, a measure of how expensive energy is at each time in the day
			optionally, we can return the optimized demand, which is the building
			calculating where the net demand should be allocated 
		"""

		pv = np.array([])
		price = np.array([])
		demand = np.array([])

		with open('building_data.csv', encoding='utf8') as csvfile:
		    csvreader = csv.reader(csvfile, delimiter=',')
		    next(csvreader,None)
		    rowcount = 0
		    for row in csvreader:
		        pv = np.append(pv, 0.001*float(row[3])) # Converting Wh to kWh
		        price = np.append(price, float(row[2])) # Cost per kWh
		        val = row[5]
		        if val in (None,""): #How to treat missing values
		            val = 0
		        else:
		            val = float(val) # kWh
		        demand = np.append(demand, val)
		        rowcount+=1
		        # if rowcount>100:
		        #     break

		pvsize = 5 #Assumption

		netdemand = demand.copy()
		for i in range(len(demand)):
		    netdemand[i] = demand[i] - pvsize*pv[i]

		# Data starts at 5 am on Jan 1
		netdemand_24 = netdemand[24*day-5:24*day+19]
		price_24 = price[24*day-5:24*day+19]
		pv_24 = pv[24*day-5:24*day+19]
		demand_24 = demand[24*day-5:24*day+19]

		# Calculate optimal load scheduling. 90% of load is fixed, 10% is controllable.
		def optimise_24h(netdemand_24, price_24):
		    currentcost = netdemand_24*price_24
		    
		    fixed_load = 0.9*netdemand_24
		    controllable_load = sum(0.1*netdemand_24)
		    # fixed_load = 0*netdemand_24
		    # controllable_load = sum(netdemand_24)
		    
		    def objective(x):
		        load = fixed_load + x
		        cost = np.multiply(price_24,load)
		        # Negative demand means zero cost, not negative cost
		        # Adding L1 regularisation to penalise shifting of occupant demand
		        lambd = 0.005
		        return sum(np.maximum(cost,0)) + lambd*sum(abs(x-0.1*netdemand_24))

		    def constraint_sumofx(x):
		        return sum(x) - controllable_load
		    
		    def constraint_x_positive(x):
		        return x 

		    x0 = np.zeros(24)
		    cons = [
		        {'type':'eq', 'fun': constraint_sumofx},
		        {'type':'ineq', 'fun':constraint_x_positive}
		    ]
		    sol = minimize(objective, x0, constraints=cons)
		    return sol

		sol = optimise_24h(netdemand_24,price_24)
		x = sol['x']

		netdemand_price_24 = netdemand_24*price_24

		return(netdemand_price_24)

def main():
	test_office = Office()
	end = False
	distances = []
	day = 1
	while not end:
		timestep = test_office.get_timestep()
		print("--------" + str(timestep) + "-------")
		prices = test_office.price_signal(day)
		distance, end = test_office.step(prices)
		print(distance)
		day+=1
		distances.append(distance)

	plt.plot(distances)
	plt.show()



if __name__ == "__main__":
	main()

	

		   