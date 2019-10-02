import cvxpy as cvx

#### file to calculate the rewards. Meant to be modular: 
#### class Rewards should have several different functions by Dec 2019


class Reward():
	def __init__(energy_use, prices, min_demand, max_demand, total_day_demand):
		"""
		Args: 
			energy_use: list returned by Person class signifying energy use 
			prices: list returned by minigrid or building resources signifying cost throughout day 
			min_demand: value computed by Person class signifying minimum energy use long term
			max_demand: value computed by Person class signifying maximum energy use long term
			total_day_demand: value compued by Person class signifying the sum of the day's demands 
		"""

		self.energy_use = np.array(energy_use)
		self.prices = np.array(prices) 
		self._num_timesteps = energy_use.shape[0]
		self.min_demand = min_demand
		self.max_demand = max_demand

	def ideal_use_calculation():
		"""
		Computes an optimization of demand according to price 

		returns: np.array of ideal energy demands given a price signal 
		"""

		demands = cvx.Variable(self._num_timesteps)
		min_demand = cvx.Parameter(1)
		max_demand = cvx.Parameter(1)
		total_demand = cvx.Parameter(1)
		prices = cvx.Parameter(self._num_timesteps)

		min_demand.value = self.min_demand
		max_demand.value = self.max_demand
		total_demand = self.total_demand
		prices = self.prices.values

		constraints = [cvx.sum(demands, axis=0, keepdims=True) == total_demand]
		for i in self._num_timesteps:
			constraints += [demands[i] <= max_demand]
			constraints += [demands[i] <= min_demand]	

		objective = cvx.Minimize(demands.T * prices)
		problem = cvx.Problem(objective, constraints)

		problem.solve()

		print(demands)

		return np.array(demands)

	def neg_distance_from_ideal(demands):
		"""
		args: 
			demands: np.array() of demands from ideal_use_calculation()

		returns: 
			a numerical distance metric, negated
		"""

		return -((demands - self.energy_use)**2).sum()









