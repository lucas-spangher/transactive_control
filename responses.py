from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Person(ABC):
    @abstractmethod
    def process_points(self, parameter_list):
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def BASELINE_ENERGY(self):
        raise NotImplementedError

    def energy_to_points(self):
        raise NotImplementedError

    @abstractmethod
    def energy_consumed(self, day, hour):
        raise NotImplementedError

    BASELINE_ENERGY = 300
    STARTING_POINTS = 500

    available_hours = [
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
    ]

    available_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


class Person1(Person):
    """
    This person uses energy consistently throughout the day, usually on computer, 
    apart from lunch between 12-1PM. They're the type of person that is mostly driven
    by points, so if the energy_unit_cost for that hour is high compared to the other 
    hours, their energy will be shifted relatively drastically from the baseline. """

    # TODO: analyze baseline, see if it's producing realistic values

    def __init__(
        self,
        mean,
        sd,
        lag_function,
        # points_differential_csv="sample_points_differential.csv",
        energy_cost_csv,
    ):
        self.noise_distribution = np.random.normal(mean, sd)
        # self.points_differential = pd.read_csv(points_differential_csv)
        self.lag_function = lag_function
        self.energy_cost = pd.read_csv(energy_cost_csv)
        # introducing this lag variable - determines the amount of time that it takes for the participant
        # to fully adjust their default behavior from closer to the baseline to closer to the intended behavior

        # self.lag = 4  # hours, this variable would probably decrease by one each week
        # self.points =

    # def energy_to_points(self, energy, day, hour, points_df):
    #     """
    #     Assuming that optimal energy usage involves significantly decreasing the energy usage during the afternoon
    #     hours (11AM - 3PM), by at least 30%, while decreasing the energy used at other hours by at least 10%.

    #     We also assume that the points received per hour, per day, are inversely proportional
    #     to the differential between the actual energy usage, and the optimal energy usage, times the differential multiplier.
    #     """

    #     # TODO: change this energy_to_points behavior based on more information about what optimal energy distributions look like

    #     baseline_energy = self.baseline(day, hour)
    #     optimal_energy_usage = (
    #         baseline_energy * 0.7
    #         if (hour > 11 and hour < 15)
    #         else baseline_energy * 0.9
    #     )
    #     multiplier_at_time = points_df[hour][day]
    #     return (
    #         abs(energy - optimal_energy_usage)
    #         / optimal_energy_usage
    #         * multiplier_at_time
    #     )

    def baseline(self, day, hour):
        """Baseline from 12-1PM is reduced, as they're not using their computer and lights. 
        The rest of the makeup of the function is based on a qualitative assessment of relative
        levels of productivity throughout the day. The specific values can be changed. """

        # TODO: Verify values based on research into energy usage per day

        day_multiplier = {
            "Monday": 1.1,
            "Tuesday": 1.15,
            "Wednesday": 1,
            "Thursday": 0.9,
            "Friday": 0.8,
        }
        hour_multiplier = {
            "08:00": 0.8,
            "09:00": 0.9,
            "10:00": 1,
            "11:00": 0.9,
            "12:00": 0,
            "13:00": 0.9,
            "14:00": 1.1,
            "15:00": 1.1,
            "16:00": 1.0,
            "17:00": 0.9,
            "18:00": 0.8,
        }

        return self.BASELINE_ENERGY * day_multiplier[day] * hour_multiplier[hour]

    def total_energy_consumed(self, day):
        return sum([self.baseline(day, hour) for hour in self.available_hours])

    def redistributed_energy(self, day, points):
        self.MAX_DIFFERENTIAL = 5
        import cvxpy as cvx

        energy_curve = cvx.Variable(11)
        objective = cvx.Minimize(energy_curve.T * points["Cost"])
        constraints = [
            cvx.sum(energy_curve, axis=0, keepdims=True)
            == self.total_energy_consumed(day)
        ]
        for hour in range(11):
            constraints += [energy_curve[hour] >= 0]

        for hour in range(1, 11):
            constraints += [
                abs(energy_curve[hour] - energy_curve[hour - 1])
                <= self.MAX_DIFFERENTIAL
            ]

        problem = cvx.Problem(objective, constraints)
        problem.solve()
        return energy_curve

    def predicted_energy_behavior(self, day):

        self.AFFINITY_TO_POINTS = 0.8
        self.ENERGY_STD_DEV = 5

        perfect_energy_use = self.redistributed_energy(day, self.energy_cost)
        baseline_energy_use = [
            self.baseline(day, hour) for hour in self.available_hours
        ]
        means = np.array(len(perfect_energy_use))
        for i in range(len(perfect_energy_use)):
            lesser, greater = (
                perfect_energy_use,
                baseline_energy_use
                if perfect_energy_use < baseline_energy_use
                else baseline_energy_use,
                perfect_energy_use,
            )
        means = np.append(means, lesser + 0.8 * (greater - lesser))
        return np.random.normal(means, self.ENERGY_STD_DEV)

    # def energy_consumed(self, day, hour):
    #     """
    #     Assume that the lag function is logistic, for person 1.
    #     """

    #     return w1 * self.lag * argmax(energy_to_points, key=energy) + w2 * baseline(
    #         day, hour
    #     )


class Person2(Person):
    """
    This person comes into the office very sporadically on any given day, there's a
    random (Bernoulli) variable denoting if they came to the ofice that day or no. If 
    they do come, then they have fairly uniform usage each day. However, on days in which
    they did not come, that will result in a usage of the baseline energy regardless. On
    days they do come, they react similarly to the first person. 
    """

    def __init__(self, behavior, points_csv):
        self.behavior = behavior

    def process_points(self, points_df):
        pass


class Person3(Person):
    """
    This person doesn't care much about the game in general, so their energy saving behavior 
    is aunffected, no matter what the points are (this is an edge case)
    """

    def __init__(self):
        pass

    def process_points(self, points_df):
        pass


class Person4(Person):
    """
    This person is one who is less driven by points than the first person. but has higher 
    self-efficacy, so she is less affected by the change in points than the average person.
    """

    def __init__(self, behavior, self_efficacy):
        "Self efficacy is measured as a number between 0 and 1."
        self.behavior = behavior


if __name__ == "__main__":
    # person = Person4()
    # person = Person1()
    pass
