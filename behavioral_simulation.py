import numpy as np
import random
from collections import defaultdict


class Person:
    """The Person class contains the bulk of the simulation -- the main
    purpose of the Workstation class is to help with the vicarous learning
    update.
    The initial starting states are dependent on the behavioral characteristic
     of the person -- depending on their behavioral class, different people in 
     the simulation will have different starting states.
     The flow weights between different weights, as well as the exogenous variables
     at each time step, have been assigned to be random values between 0 and 1."""

    def __init__(self, workstation, baseline, states=[], weights=[]):
        self.workstation = workstation
        self.curr_timestep = 0
        self.baseline = baseline

        # set the out of office value as a boolean, zero out the delta if you are out of the office
        # this would have to change the structure of the matrices as well

        # randomly initialize states between 0 and 1, if states are not passed in

        # fit in gaussian process on the person level

        # assume price signal is a twelve-key dataframe

        if not states:
            outcome_expectancy = random.random()
            self_efficacy = random.random()
            behavior = random.random()
            behavior_outcomes = random.random()
            states = np.array(
                [outcome_expectancy, self_efficacy, behavior, behavior_outcomes]
            )

        self.states = states

        # initialize state weights
        alpha = -0.5
        beta_41 = random.random()
        beta_32 = random.random()
        beta_13 = random.random()
        beta_23 = random.random()
        beta_34 = random.random()

        self.state_weights = np.array(
            [
                [alpha, 0, 0, beta_41],
                [0, alpha, beta_32, 0],
                [beta_13, beta_23, alpha, 0],
                [0, 0, beta_34, alpha],
            ]
        )

        w_1 = random.random()
        w_2 = random.random()
        w_3 = random.random()
        w_4 = random.random()
        w_5 = random.random()
        w_6 = random.random()
        w_7 = random.random()

        self.input_weights = np.array(
            [
                [1, w_1, 0, 0, 0, 0, 0, 0],
                [1, 0, w_2, 0, 0, 0, 0, 0],
                [0, 0, 0, w_3, w_4, w_5, w_6, 0],
                [0, 0, 0, 0, 0, 0, 0, w_7],
            ]
        )

    def exogenous_inputs(self, timestep):
        vicarious_learning = self.workstation.vicarious_learning_average(timestep - 1)
        weekly_poll = random.random()
        pretreatment_survey = random.random()
        points = random.random()
        email_indicator = random.random()

        # to be discussed - how out of office and energy saturation can be integrated as part of the workstation class

        out_of_office = random.random()
        energy_saturation_measure = random.random()

        predicted_energy = random.random()
        return np.array(
            [
                vicarious_learning,
                weekly_poll,
                pretreatment_survey,
                points,
                email_indicator,
                out_of_office,
                energy_saturation_measure,
                predicted_energy,
            ]
        )

    def update(self):
        """Final update of the form:
        state_vector_{t+1} = state_weights * state_vector_{t} + input_weights * exogenous_inputs_{t}"""
        self.states = np.dot(self.state_weights, self.states) + np.dot(
            self.input_weights, self.exogenous_inputs(self.curr_timestep)
        )

    def get_predicted_energy(self, baseline):
        """Gives the predicted energy distribution, as a function of baseline energy usage and a delta 
        function that is proportional to the behavior state. Will use the average of the three previous weeks"""

        # what does baseline refer to?

        return 0


class Workstation:
    """Each workstation consists of some number of people, set by default. 
When conducting the exogenous variable update, We use the list of people 
in a given workstation, along with the energy used at the previous timestep,
to determine the impact of vicarious learning."""

    def __init__(self, num_people=5, people_list=[]):
        self.people = [Person(self)] * num_people
        self.energy_used = defaultdict(dict)
        self.curr_timestep = 0

    def vicarious_learning_average(self, timestep):
        if timestep not in self.energy_used:
            return 0
        vicarious_learning_list = [
            self.energy_used[timestep - 1][person] for person in self.people
        ]
        return sum(vicarious_learning_list) / len(vicarious_learning_list)

    def update(self):
        for person in self.people:
            person.update()
            self.energy_used[self.curr_timestep][person] = person.get_predicted_energy(
                person.baseline
            )

        self.curr_timestep += 1
