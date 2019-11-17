import cvxpy as cvx
import osqp
import numpy as np
from dtw import dtw

#### file to calculate the rewards. Meant to be modular:
#### class Rewards should have several different functions by Dec 2019


class Reward():
	def __init__(self, energy_use, prices, min_demand, max_demand):
		"""
		Args:
			energy_use: list returned by Person class signifying energy use
			prices: list returned by minigrid or building resources signifying cost throughout day
			min_demand: value computed by Person class signifying minimum energy use long term
			max_demand: value computed by Person class signifying maximum energy use long term
		"""

		self.energy_use = np.array(energy_use)
		self.prices = np.array(prices)
		self._num_timesteps = energy_use.shape[0]
		self.min_demand = min_demand
		self.max_demand = max_demand
		self.total_demand = np.sum(energy_use)

	def ideal_use_calculation(self):
		"""
		Computes an optimization of demand according to price

		returns: np.array of ideal energy demands given a price signal
		"""

		demands = cvx.Variable(self._num_timesteps)
		min_demand = cvx.Parameter()
		max_demand = cvx.Parameter()
		total_demand = cvx.Parameter()
		prices = cvx.Parameter(self._num_timesteps)

		min_demand = self.min_demand
		max_demand = self.max_demand
		total_demand = self.total_demand
		prices = self.prices

		constraints = [cvx.sum(demands, axis=0, keepdims=True) == total_demand]
		# constraints = [np.ones(self._num_timesteps).T * demands == total_demand]
		for i in range(self._num_timesteps):
			constraints += [demands[i] <= max_demand]
			constraints += [min_demand <= demands[i]]


		objective = cvx.Minimize(demands.T * prices)
		problem = cvx.Problem(objective, constraints)

		problem.solve(solver = cvx.OSQP, verbose=False)

		return np.array(demands.value)

	def neg_distance_from_ideal(self, demands):
		"""
		args:
			demands: np.array() of demands from ideal_use_calculation()

		returns:
			a numerical distance metric, negated
		"""

		return -((demands - self.energy_use)**2).sum()

	def cost_distance(self, demands):
		"""
		args:
			demands: np.array() of demands from ideal_use_calculation()

		returns:
			a cost-based distance metric, negated
		"""
		current_cost = self.prices*self.energy_use
		ideal_cost = self.prices*demands

		return (ideal_cost-current_cost).sum()

	def dynamic_time_warping(self, demands):
		"""
		args:
			demands: np.array() of demands from ideal_use_calculation()

		returns:
			a numerical distance metric, negated, computed using dynamic time warping
		"""
		total_points = len(self.energy_use)
		actual_use = self.energy_use.reshape(total_points)
		euclidean_norm = lambda x, y: (x-y)**2
		d, cost_matrix, acc_cost_matrix, path = dtw(demands, actual_use, dist=euclidean_norm)
		return -d

	def correlation_difference(self, demands):
		"""
		args:
			demands: np.array() of demands from ideal_use_calculation()

		returns:
			a numerical distance metric, negated, using pearson correlation
		"""
		total_points = len(self.energy_use)
		return np.corrcoef(demands, self.energy_use.reshape(total_points))[0][1]
