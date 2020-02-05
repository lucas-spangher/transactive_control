#%%
import numpy as np
import random
from collections import defaultdict
import csv
import pandas as pd
from datetime import datetime, timedelta

# things to do:
# create functions for out of office, energy saturation weights
# fill in function for energy baseline, and predicted energy
# move away from workstation
# build in hourly updates


class Person:
    """The Person class contains the bulk of the simulation -- the main
    purpose of the Workstation class is to help with the vicarous learning
    update.
    The initial starting states are dependent on the behavioral characteristic
     of the person -- depending on their behavioral class, different people in 
     the simulation will have different starting states.
     The flow weights between different weights, as well as the exogenous variables
     at each time step, have been assigned to be random values between 0 and 1."""

    def __init__(self, workstation, name, df_data, states=[], weights=[]):
        self.workstation = workstation
        self.name = name
        self.states = []
        self.weights = []
        self.email_data = df_data[0]
        self.energy_data = df_data[1]
        self.points_data = df_data[2]
        self.presurvey_data = df_data[3]
        self.weekly_survey_data = df_data[4]
        self.baseline_data = df_data[5]
        self.ooo_data = df_data[6]

        # put filepaths of the CSV here

        # set the out of office value as a boolean, zero out the delta if you are out of the office
        # this would have to change the structure of the matrices as well

        # randomly initialize states between 0 and 1, if states are not passed in

        # fit in gaussian process on the person level

        # assume price signal is a twelve-key dataframe

        if not self.states:
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

    def exogenous_inputs(self, timestamp):
        vicarious_learning = self.workstation.vicarious_learning_average(timestamp)
        weekly_poll = self.get_weekly_poll(timestamp)
        pretreatment_survey = self.get_pretreatment_from_csv(timestamp)
        points = self.get_points(timestamp)
        email_indicator = self.get_email_indicator(timestamp)

        # to be discussed - how out of office and energy saturation can be integrated as part of the workstation class

        out_of_office = self.get_out_of_office_score(timestamp)
        energy_saturation_measure = self.get_energy_saturation_daily_baseline(timestamp)

        predicted_energy_baseline = self.get_hourly_baseline(timestamp)
        predicted_energy = self.get_predicted_energy(predicted_energy_baseline)
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

    def update(self, timestamp):
        """Final update of the form:
        state_vector_{t+1} = state_weights * state_vector_{t} + input_weights * exogenous_inputs_{t}"""
        self.states = np.dot(self.state_weights, self.states) + np.dot(
            self.input_weights, self.exogenous_inputs(timestamp)
        )

    def get_energy_at_time(self, date, hour):
        val = self.energy_data[
            (self.energy_data["Date"] == date) & (self.energy_data["Hour"] == hour)
        ]["HourlyEnergy"].iloc[0]
        return val

    def get_energy_saturation_daily_baseline(self, timestamp):
        baseline_times = []
        energy = []

        for _ in range(3):
            timestamp = timestamp - timedelta(weeks=1)
            baseline_times.append(self.extract_time_info(timestamp))

        for datetime in baseline_times:
            daily_sum = []
            for hour in range(8, 21):
                daily_sum.append(
                    self.energy_data[
                        (self.energy_data["Date"] == datetime[0])
                        & (self.energy_data["Hour"] == hour)
                    ]["HourlyEnergy"].iloc[0]
                )
            energy.append(sum(daily_sum))

        return sum(energy) / len(energy)

    def get_energy_saturation(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        cume_energy = []
        for hour in range(8, hour + 1):
            cume_energy.append(self.get_energy_at_time(date, hour))

        return sum(cume_energy) / self.get_energy_saturation_daily_baseline(timestamp)

    def get_predicted_energy(self, baseline):
        """Gives the predicted energy distribution, as a function of baseline energy usage and a delta 
        function that is proportional to the behavior state. Will use the average of the three previous weeks"""

        c = 0.5

        return baseline + self.states[2] * c

        # what does baseline refer to?
        # how to calculate the predicted energy relative to baseline

    def extract_time_info(self, timestamp):
        return timestamp.date(), timestamp.hour, int(timestamp.week)

    def get_weekly_poll(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.weekly_survey_data[self.weekly_survey_data["Week_Number"] == week][
            "WeeklySurvey"
        ].iloc[0]
        return val

    def get_pretreatment_from_csv(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.presurvey_data["PreSurvey"].iloc[0]
        return val

    def get_points(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.points_data[
            (self.points_data["Date"] == date) & (self.points_data["Hour"] == hour)
        ]["Points"].iloc[0]
        return val

    def get_email_indicator(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.email_data[
            (self.email_data["Date"] == date) & (self.email_data["Hour"] == hour)
        ]["Email"].iloc[0]
        return val

    def get_hourly_baseline(self, timestamp):
        baseline_times = []
        energy = []

        for _ in range(3):
            timestamp = timestamp - timedelta(weeks=1)
            baseline_times.append(self.extract_time_info(timestamp))

        for time in baseline_times:
            energy.append(
                self.energy_data[
                    (self.energy_data["Date"] == time[0])
                    & (self.energy_data["Hour"] == time[1])
                ]["HourlyEnergy"].iloc[0]
            )

        return sum(energy) / len(energy)

    def get_out_of_office_score(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.ooo_data[
            (self.ooo_data["Date"] == date) & (self.ooo_data["Hour"] == hour)
        ]["OutOfOffice"].iloc[0]
        return val


class Workstation:
    """Each workstation consists of some number of people, set by default. 
When conducting the exogenous variable update, We use the list of people 
in a given workstation, along with the energy used at the previous timestep,
to determine the impact of vicarious learning."""

    counter = 1

    def __init__(self, name):
        # self.people = [Person(self)] * num_people
        self.name = name
        # self.people_list = people_list
        self.energy_used = defaultdict(dict)
        self.curr_timestep = 0
        self.counter = 1

    def extract_time_info(self, timestamp):
        return timestamp.date(), timestamp.hour, int(timestamp.week)

    def vicarious_learning_average(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        print(date)
        print(hour - 1)
        prev_hour_energy = [
            person.get_energy_at_time(date, hour - 1) for person in self.people_list
        ]
        return sum(prev_hour_energy) / len(prev_hour_energy)

    def update(self, timestamp):
        for person in self.people_list:
            print(
                person.name,
                timestamp,
                person.get_predicted_energy(person.get_hourly_baseline(timestamp)),
                Workstation.counter,
            )
            person.update(timestamp)

            Workstation.counter += 1


class Simulation:
    def __init__(
        self,
        email_csv="email_dummy.csv",
        energy_csv="energy_dummy.csv",
        points_csv="points_dummy.csv",
        presurvey_csv="presurvey_dummy.csv",
        weekly_survey_csv="weekly_survey_dummy.csv",
        baseline_csv="BaselineHourlyData.csv",
        out_of_office_csv="OOO_dummy.csv",
    ):
        self.emails_df = pd.read_csv(email_csv)
        self.energy_df = pd.read_csv(energy_csv)
        self.points_df = pd.read_csv(points_csv)
        self.presurvey_df = pd.read_csv(presurvey_csv)
        self.weekly_survey_df = pd.read_csv(weekly_survey_csv)
        self.baseline_df = pd.read_csv(baseline_csv)
        self.ooo_df = pd.read_csv(out_of_office_csv)

        dfs = [
            self.emails_df,
            self.energy_df,
            self.points_df,
            self.presurvey_df,
            self.weekly_survey_df,
            self.baseline_df,
            self.ooo_df,
        ]

        for df in dfs:
            df.Date = pd.to_datetime(df.Date).dt.date

        self.workstations = []

        for workstation_name in self.emails_df["WorkGroup"].unique():
            curr_workstation = Workstation(workstation_name)
            filtered_df = self.emails_df[
                self.emails_df["WorkGroup"] == workstation_name
            ]
            curr_person_list = []
            for person_name in filtered_df["Name"].unique():
                curr_person_list.append(
                    Person(
                        workstation=curr_workstation,
                        name=person_name,
                        df_data=[
                            df[
                                (df["Name"] == person_name)
                                & (df["WorkGroup"] == workstation_name)
                            ]
                            for df in dfs
                        ],
                    )
                )

            curr_workstation.people_list = curr_person_list
            self.workstations.append(curr_workstation)

    def daily_update(self, starting_datetime):
        for hour in range(12):
            for workstation in self.workstations:
                workstation.update(starting_datetime + timedelta(hours=hour))


#%%
simulation = Simulation()
dummy_date = pd.Timestamp("2018-09-20T08")
simulation.daily_update(dummy_date)

# %%
