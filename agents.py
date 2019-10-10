import pandas as pd
import numpy as np

#### file to make the simulation of people that we can work with 


class Person():
	""" Person (parent?) class -- will define how the person takes in a points signal and puts out an energy signal 
	baseline_energy = a list or dataframe of values. This is data from SinBerBEST 
	points_multiplier = an int which describes how sensitive each person is to points 

	"""

	def __init__(self, baseline_energy, points_multiplier = 1):
		self.baseline_energy = baseline_energy
		self.points_multiplier = points_multiplier

	def energy_output_simple_linear(self, points):
		"""Determines the energy output of the person, based on the formula:
		
		y[n] = -sum_{rolling window of 5} points + baseline_energy + noise

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 5 minute increments

		"""
		points_df = pd.DataFrame(points)
		
		points_effect = (
			points_df
				.rolling(
						window = 5,
						min_periods = 1)
				.mean()
			)

		time = points_effect.shape[0]
		energy_output= []

		for t in range(time):
			temp_energy = self.baseline_energy[t] - \
				points_effect.iloc[t]*self.points_multiplier + \
				np.random.normal(1)
			energy_output.append(temp_energy)
			
		return pd.DataFrame(energy_output)

class Person_with_hysteresis(Person):
	""" Wendy -- Determines the energy output of the person, based on the formula:
		
		y[n] = f(points) + baseline_energy + noise

		f: super special secret function that Wendy designs with hysteresis 

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 5 minute increments"""

	def __init__(self, baseline_energy, points_multiplier = 1):
		pass












