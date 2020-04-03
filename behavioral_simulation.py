#%%
import numpy as np
import random
from collections import defaultdict
import csv
import pandas as pd
from datetime import datetime, timedelta
import cvxpy as cvx
import IPython
from gekko import GEKKO
import math

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

    def __init__(
        self, workstation, name, df_data, starting_date, states=[], weights=[]
    ):
        self.workstation = workstation
        self.name = name
        self.states = []
        self.weights = []
        self.starting_date = (
            starting_date  ## TODO: Manan, is there a way to infer this from
        )
        # df_data instead? If so, please feel free to take it
        # out of the init and modify accordingly.
        self.email_data = df_data[0]
        self.energy_data = df_data[1]
        self.points_data = df_data[2]
        self.presurvey_data = df_data[3]
        self.weekly_survey_data = df_data[4]
        self.baseline_data = df_data[5]
        self.ooo_data = df_data[6]
        self.generic_energy_day = np.array(
            [0.000000, 
            0.960, 
            1.50, 
            1.210, 
            1.220, 
            1.290, 
            1.290, 
            1.20, 
            1.269,
            1.210001,
            114.290000,
            93.4100,
            97.730000,
            71.470000,
            112.520000,
            82.030000,
            77.800000,
            0.840000,
            86.760000, 
            59.820000, 
            24.230000, 
             0.330000, 
             0.330000, 
             0.400000
             ])

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

        ### A matrix
        self.state_weights = np.array(
            [
                [alpha,   0,       0,       beta_41],
                [0,       alpha,   beta_32, 0],
                [beta_13, beta_23, alpha,   0],
                [0,       0,       beta_34, alpha],
            ]
        )

        w_1 = random.random()
        w_2 = random.random()
        w_3 = random.random()
        w_4 = random.random()
        w_5 = random.random()
        w_6 = random.random()
        w_7 = random.random()

        ### B matrix
        self.input_weights = np.array(
            [
                [1, w_1, 0, 0, 0, 0, 0, 0],
                [1, 0, w_2, 0, 0, 0, 0, 0],
                [0, 0, 0, w_3, w_4, w_5, w_6, 0],
                [0, 0, 0, 0, 0, 0, 0, w_7],
            ]
        )

        # TODO: Manan great job on this code! It's fantastic, very professional.

    def exogenous_inputs(self, timestamp):

        ## Question: where does vicarious_learning_average get called when it references a prior hour?

        vicarious_learning = self.workstation.vicarious_learning_average(timestamp)
        weekly_poll = self.get_weekly_poll(timestamp)
        pretreatment_survey = self.get_pretreatment_from_csv(timestamp)
        points = self.get_points(timestamp)
        email_indicator = self.get_email_indicator(timestamp)

        # to be discussed - how out of office and energy saturation can be integrated as part of the workstation class

        out_of_office = self.get_out_of_office_score(timestamp)
        energy_saturation_measure = self.get_energy_saturation_daily_baseline(timestamp)

        if math.isnan(energy_saturation_measure):
            print("energy_saturation_measure is nan, replacing with 0")
            print(energy_saturation_measure)
            energy_saturation_measure = 0

        predicted_energy_baseline = self.get_hourly_baseline(timestamp)
        predicted_energy = self.get_predicted_energy(predicted_energy_baseline)

        # TODO: Manan, please eliminate this once predicted_energy is not returning nans 
        if math.isnan(predicted_energy):
            print("predicted_energy is nan, replacing with 0")
            predicted_energy = 0

        return np.array(
            [
                vicarious_learning,
                weekly_poll,
                pretreatment_survey,
                points,
                email_indicator,
                out_of_office,
                energy_saturation_measure,
                predicted_energy,  ## TODO: This is turning back nans :/
            ]
        )

    def step(self, timestamp):
        """ Function to step through the hours of the day based on the various 
        weight matrices 

        Note: used to be called "update", but I changed it because we 
        were overusing that word 

        Final update of the form:
        state_vector_{t+1} = state_weights * state_vector_{t} +
             input_weights * exogenous_inputs_{t}"""

        self.states = np.dot(self.state_weights, self.states) + np.dot(
            self.input_weights, self.exogenous_inputs(timestamp)
        )

    def daily_weight_fit(self, date):
        """ 
        Update function to the weights of the dynamic system, this will be called 
        once a day after the data has arrived. 
        """

        # # comments in cvx code
        # A = cvx.Variable(self.state_weights.shape)
        # B = cvx.Variable(self.input_weights.shape)

        m = GEKKO(remote = False)
        A = m.Array(
                m.Var, 
                (self.state_weights.shape),
                value = .5,
                lb = -1,
                ub = 1
            )

        ## diagonal entries
        # 0, 0
        A[0][0].value = -.5; A[0][0].lower = -1; A[0][0].upper = 1
        A[1][1].value = -.5; A[1][1].lower = -1; A[1][1].upper = 1
        A[2][2].value = -.5; A[2][2].lower = -1; A[2][2].upper = 1
        A[3][3].value = -.5; A[3][3].lower = -1; A[3][3].upper = 1

        ## non-zero entries
        A[0][3].value = random.random(); A[0][3].upper = 1; A[0][3].lower = -1 # for the beta labelled 4, 1 
        A[1][2].value = random.random(); A[1][2].upper = 1; A[1][2].lower = -1 # beta32
        A[2][0].value = random.random(); A[2][0].upper = 1; A[2][0].lower = -1 # beta13         
        A[1][2].value = random.random(); A[1][2].upper = 1; A[1][2].lower = -1 # beta23
        A[3][2].value = random.random(); A[3][2].upper = 1; A[3][2].lower = -1 # beta34

        ## zero entries 
        A[0][1].value = A[0][1].upper = A[0][1].lower = 0
        A[0][2].value = A[0][2].upper = A[0][2].lower = 0
        A[1][0].value = A[1][0].upper = A[1][0].lower = 0
        A[1][3].value = A[1][3].upper = A[1][3].lower = 0
        A[2][1].value = A[2][1].upper = A[2][1].lower = 0
        A[2][3].value = A[2][3].upper = A[2][3].lower = 0
        A[3][0].value = A[3][0].upper = A[3][0].lower = 0
        A[3][1].value = A[3][1].upper = A[3][1].lower = 0

        ############ B matrix
   
        B = m.Array(
                m.Var, 
                (self.input_weights.shape), 
            )

        # 1 entries: 
        # B_11 and B_21
        B[0][0].value = 1; B[0][0].upper = 1; B[0][0].lower = 1
        B[1][0].value = 1; B[1][0].upper = 1; B[1][0].lower = 1

        # weights w1, w2, w3, w4,..., w7
        B[0][1].value = random.random(); B[0][1].upper = 1; B[0][1].lower = -1
        B[1][2].value = random.random(); B[1][2].upper = 1; B[1][2].lower = -1
        B[2][3].value = random.random(); B[2][3].upper = 1; B[2][3].lower = -1
        B[2][4].value = random.random(); B[2][4].upper = 1; B[2][4].lower = -1
        B[2][5].value = random.random(); B[2][5].upper = 1; B[2][5].lower = -1
        B[2][6].value = random.random(); B[2][6].upper = 1; B[2][6].lower = -1
        B[3][7].value = random.random(); B[3][7].upper = 1; B[3][7].lower = -1

        # zeros
        B[0][2].value = B[0][2].upper = B[0][2].lower = 0
        B[0][3].value = B[0][3].upper = B[0][3].lower = 0
        B[0][4].value = B[0][4].upper = B[0][4].lower = 0
        B[0][5].value = B[0][5].upper = B[0][5].lower = 0
        B[0][6].value = B[0][6].upper = B[0][6].lower = 0
        B[0][7].value = B[0][7].upper = B[0][7].lower = 0
        B[1][1].value = B[1][1].upper = B[1][1].lower = 0
        B[1][3].value = B[1][3].upper = B[1][3].lower = 0
        B[1][4].value = B[1][4].upper = B[1][4].lower = 0
        B[1][5].value = B[1][5].upper = B[1][5].lower = 0
        B[1][6].value = B[1][6].upper = B[1][6].lower = 0
        B[1][7].value = B[1][7].upper = B[1][7].lower = 0
        B[2][0].value = B[2][0].upper = B[2][0].lower = 0
        B[2][1].value = B[2][1].upper = B[2][1].lower = 0
        B[2][2].value = B[2][2].upper = B[2][2].lower = 0
        B[2][7].value = B[2][7].upper = B[2][7].lower = 0
        B[3][0].value = B[3][0].upper = B[3][0].lower = 0
        B[3][1].value = B[3][1].upper = B[3][1].lower = 0
        B[3][2].value = B[3][2].upper = B[3][2].lower = 0
        B[3][3].value = B[3][3].upper = B[3][3].lower = 0
        B[3][4].value = B[3][4].upper = B[3][4].lower = 0
        B[3][5].value = B[3][5].upper = B[3][5].lower = 0
        B[3][6].value = B[3][6].upper = B[3][6].lower = 0

        # dates = pd.date_range(start=self.starting_date, end=date)
        dates = date
        hours = list(range(24))


        # y = [[self.get_days_energy(date = date)] for date in dates]
        y = self.get_days_energy(date = date)
        # baseline_energies = [self.get_hourly_baseline_for_day(date) for date in dates]
        baseline_energies = self.get_hourly_baseline_for_day(date)

        flat_baseline_energies = np.reshape(baseline_energies, -1)
        flat_y = np.reshape(y, -1)

        flat_diff = np.subtract(flat_y, flat_baseline_energies).values

        ## TODO: subtract the baseline energy for that day from y
        timesteps = len(flat_y)

        # TODO: ask Alex, should this be a Param? Or Const? Or nothing?  
        u = [self.get_exogenous_inputs_of_day(date) for hour in hours]
        # u = m.Array(m.Param, 
        #         (len(self.get_exogenous_inputs_of_day(date)), len(hours)), 
        #         [(self.get_exogenous_inputs_of_day(date) for hour in hours)]
        #     )
            

        # # z should be all latent states (check with Alex)
        # z = cvx.Variable((timesteps, 4))

        z = m.Array(m.Var, (timesteps, 4), lb = -1, ub = 1)

        # # c should be (0,0,c,0)
        
        C = np.array([0, 0, 1, 0])
        gammas = m.Array(m.Var, (timesteps-1), lb = -5, ub = 5)

        m.Obj(
            m.sqrt(
                m.sum([(flat_diff[i] - z[i][3])**2 for i in range(len(flat_y))] + 
                    [gammas[i] * 
                        (z[i+1][j] - np.dot(A, z[i])[j] - np.dot(B, u[i])[j])
                            for i in range(timesteps - 1)
                        for j in range(len(z[i])) 
                    ])
                )
            )

        m.options.solver = 2
        m.options.MAX_ITER = 1000
        m.solve()

        # IPython.embed()
        
        return A, B, z
        # return np.array(A.value), np.array(B.value), np.array(z.value)

    def step_helper(self, A, B, u):
        """ Function to step through the hours of the day based on the various 
        weight matrices 

        Note: used to be called "update", but I changed it because we 
        were overusing that word 

        Final update of the form:
        state_vector_{t+1} = state_weights * state_vector_{t} +
             input_weights * exogenous_inputs_{t}"""

        timesteps = len(u)

        z = np.zeros((timesteps, 4))

        # try: introducing non-linearity, like the A could be a probability vector 
            # like a square or a log term 
            # look into how hierarchical regression solves things 
            # make an RNN instead. Guide it so that 

        for i in range(timesteps-1):
            z[i+1] = np.dot(
                A, z[i]
            ) + np.dot(
                B, u[i]
            )

        return z 


    def parameter_fit_step_helper(self, A, B, u, flat_diff, timesteps):
        """ Function to step through the hours of the day based on the various 
        weight matrices and optimize these steps.

        Note: used to be called "update", but I changed it because we 
        were overusing that word 

        Final update of the form:
        state_vector_{t+1} = state_weights * state_vector_{t} +
             input_weights * exogenous_inputs_{t}"""

        z = cvx.Variable((timesteps, 4))

        errors = []

        for i in range(timesteps):
            errors.append(cvx.square(flat_diff[i] - z[i][2]))

        objective = cvx.Minimize(
                cvx.sum(
                        errors
                    )
                )


        constraints = []

        for i in range(timesteps-1):
            constraints.append(z[i + 1] == A @ z[i] + B @ u[i])

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver = cvx.OSQP, verbose=True)

        return z.value

    def parameter_fit_paramter_helper(self, z_old, u, flat_diff, flat_y, timesteps):
        """
        Helper function to resolve the parameters, once the z are computed 

        z_old: the z variables computed from the previous round, shape (timesteps, 4)
        u: exogenous inputs, shape (7, 24), (# exogenous inputs, hours)
        flat_diff: difference between actual energy and actual baseline
        flat_y: actual observed energy 
        timesteps: number of timesteps to feed in

        returns: parameters 

        """

        A = cvx.Variable(self.state_weights.shape)
        B = cvx.Variable(self.input_weights.shape)

        def objective_zs(A, B, z_old, timesteps):
            z = {}
            z[0] = [0,0,0,0]
            for i in range(timesteps-1):
                z[i + 1] = A @ z_old[i] + B @ u[i] 
            return z
        
        z_list = objective_zs(A, B, z_old, timesteps)

        errors = []

        for i in range(timesteps):
            errors.append(
                cvx.square(
                    flat_diff[i] - z_list[i][2]
                    )
            )

        objective = cvx.Minimize(
            cvx.sum(
                    errors
                )
            )
            
        constraints = [A[0][0] >= -.5, A[1][1] >= -.5, A[2][2] >= -.5, A[3][3] >= -.5, 
                A[0][1]  == 0,  A[0][2] == 0,  
                A[1][0] == 0,  A[1][3] == 0,  A[2][1] == 0, 
                A[2][3] == 0,  A[3][0] == 0,  A[3][1] == 0, 
                B[0][0] == 1,  B[1][0] == 1,  B[0][2] == 0,  
                B[0][3] == 0,  B[0][4] == 0,  B[0][5] == 0, 
                B[0][6] == 0,  B[0][7] == 0, 
                B[1][1] == 0,  B[1][3] == 0, 
                B[1][4] == 0,  B[1][5] == 0,  B[1][6] == 0, 
                B[1][7] == 0,  B[2][0] == 0, 
                B[2][1] == 0,  B[2][2] == 0,  B[2][7] == 0, 
                B[3][0] == 0,  B[3][1] == 0, 
                B[3][2] == 0,  B[3][3] == 0, 
                B[3][4] == 0,  B[3][5] == 0, B[3][6] == 0]

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver = cvx.OSQP, verbose=True)

        return A.value, B.value

    def iterative_convex_parameter_fit(self, date):
        """
        Attempting to update the A, B matrices with a convex objective
        
        """

        y = self.get_days_energy(date = date)
        baseline_energies = self.get_hourly_baseline_for_day(date)

        flat_baseline_energies = np.reshape(baseline_energies, -1)
        flat_y = np.reshape(y, -1)

        hours = list(range(24))

        u = [self.get_exogenous_inputs_of_day(date) for hour in hours]
        flat_diff = np.subtract(flat_y, flat_baseline_energies).values
        
        timesteps = len(flat_y)

        A = self.state_weights
        B = self.input_weights

        zs_to_save = []

        for i in range(50):

            ## try one single optimization 

            zs = self.parameter_fit_step_helper(A, B, u, flat_diff, timesteps)
            A, B = self.parameter_fit_paramter_helper(zs, u, flat_diff, flat_y, timesteps)
            zs_to_save.append(zs)

        IPython.embed()

        return A, B

    def convex_parameter_fit(self, date):
        """
        Attempting to update the A, B matrices with a convex objective
        
        """

        y = self.get_days_energy(date = date)
        baseline_energies = self.get_hourly_baseline_for_day(date)

        flat_baseline_energies = np.reshape(baseline_energies, -1)
        flat_y = np.reshape(y, -1)

        hours = list(range(24))

        u = [self.get_exogenous_inputs_of_day(date) for hour in hours]
        flat_diff = np.subtract(flat_y, flat_baseline_energies).values
        
        timesteps = len(flat_y)

        A = cvx.Variable(self.state_weights.shape)
        B = cvx.Variable(self.input_weights.shape)

        def objective_zs(A, B, timesteps):
            z = {}
            z[0] = [0,0,0,0]
            for i in range(timesteps-1):
                z[i + 1] = A @ z[i] + B @ u[i] 
            return z
        
        z_list = objective_zs(A, B, timesteps)

        errors = []

        for i in range(timesteps):
            errors.append(cvx.square(flat_diff[i] - z_list[i][2]))

        objective = cvx.Minimize(
            cvx.sum(
                    errors
                )
            )

        IPython.embed()

        constraints = []

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver = cvx.OSQP, verbose=True)

        return A, B


    def get_exogenous_inputs_of_day(self, date):

        # TODO: Manan, please help with this

        """
        queries the self.exogenous_inputs function for each timestamp in the 
        date given, and returns an 2D (or higher D?) array that can be stepped 
        through in the update function
        """

        val = self.exogenous_inputs(date)
        return val

    def get_energy_at_time(self, timestamp = None):
        date, hour, week = self.extract_time_info(timestamp)
        val = self.energy_data[
            (self.energy_data["Date"] == date)  # changed the format
            & (self.energy_data["Hour"] == hour)
        ]["HourlyEnergy"]
        return val.iloc[0]

    def get_days_energy(self, date = None):
        date, hour, week = self.extract_time_info(date)
        val = self.energy_data[
            self.energy_data["Date"] == date  # changed the format
        ]["HourlyEnergy"].fillna(value=0)
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
            energy.append(np.nansum(daily_sum))

        len_energy = np.count_nonzero(~np.isnan(energy))
        return np.nansum(energy) / len_energy

    def get_energy_saturation(self, timestamp):
        date, hour, week = self.extract_time_info(timestamp)
        cume_energy = []
        for hour in range(8, hour + 1):
            cume_energy.append(self.get_energy_at_time(timestamp))

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

    def get_hourly_baseline_for_day(self, timestamp):
        baseline_times = []
        energy = {}

        for _ in range(3):
            timestamp = timestamp - timedelta(weeks=1)
            d, _, _ = self.extract_time_info(timestamp)
            baseline_times.append(d)

        for i, date in enumerate(baseline_times):
            to_append = self.energy_data[
                    (self.energy_data["Date"] == date)
                ]["HourlyEnergy"].fillna(value = 0) 
            energy[i] = to_append
            if i == 0:
                summed_energy = to_append
            else:
                summed_energy = np.add(summed_energy, to_append)

        return summed_energy / len(energy)

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
        if hour == 0:
            return 0  ## TODO: Manan... is this ok? What did you return when hour was 0?
        prev_hour_energy = [
            person.get_energy_at_time(timestamp) for person in self.people_list
        ]
        return sum(prev_hour_energy) / len(prev_hour_energy)

    def predict(self, timestamp):
        errors = []
        for person in self.people_list:
            errors.append(
                abs(
                    person.get_energy_at_time(timestamp)
                    - person.get_predicted_energy(person.get_hourly_baseline(timestamp))
                )
                / person.get_predicted_energy(person.get_hourly_baseline(timestamp))
            )

        person.step(timestamp)
        Workstation.counter += 1
        print(errors)
        return sum(errors) / len(errors)

    def daily_weight_fit(self, date):
        """
        Wrapper for the workstation implementation of the person-level daily_weight_update
        function

        """

        for person in self.people_list:
            print(person.name)
            (
                person.state_weights,
                person.input_weights,
                person.states,
            ) = person.iterative_convex_parameter_fit(date)
            print(person.state_weights, person.input_weights, person.states)


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
                        starting_date=pd.Timestamp(
                            "2018-09-06"
                        ),  ## TODO: not hard coded
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

    def daily_prediction(self, starting_datetime):

        # TODO: Manan -- This is future looking?

        # steps through the simulation for the day
        global_errors = []
        for hour in range(12):
            for workstation in self.workstations:
                global_errors.append(
                    workstation.predict(starting_datetime + timedelta(hours=hour))
                )

        # return

    def daily_weight_fits(self, ending_datetime):
        for workstation in self.workstations:
            workstation.daily_weight_fit(ending_datetime)


# %%
simulation = Simulation()
dummy_date = pd.Timestamp("2018-09-20T08")
#simulation.daily_prediction(dummy_date)
simulation.daily_weight_fits(dummy_date)
