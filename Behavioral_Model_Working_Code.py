"""
Code for Behavioral Model

Final update of the form:
state_vector_{t+1} = state_weights * state_vector_{t} + input_weights * exogenous_inputs_{t}
"""

import numpy as np
import datetime as dt



def init():
    # Init States
    outcome_expectancy = 0
    self_efficacy = 0
    behavior = 0
    behavior_outcomes = 0

    state_vector = np.array(
        [outcome_expectancy, self_efficacy, behavior, behavior_outcomes]
    )

    # Init State Weights
    alpha = 0.5
    beta_41 = 0
    beta_32 = 0
    beta_13 = 0
    beta_23 = 0
    beta_34 = 0

    state_weights = np.array(
        [
            [alpha, 0, 0, beta_41],
            [0, alpha, beta_32, 0],
            [beta_13, beta_23, alpha, 0],
            [0, 0, beta_34, alpha],
        ]
    )

    # Init Exogenous Inputs

    init_inputs(
        vicarious_learning,
        weekly_poll,
        pretreatment_survey,
        points,
        email_indicator,
        out_of_office,
        energy_saturation_measure,
        predicted_energy,
    )

    exogenous_inputs = np.array(
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

    # Init Input Weights
    w_1 = 0
    w_2 = 0
    w_3 = 0
    w_4 = 0
    w_5 = 0
    w_6 = 0
    w_7 = 0

    input_weights = np.array(
        [
            [1, w_1, 0, 0, 0, 0, 0, 0],
            [1, 0, w_2, 0, 0, 0, 0, 0],
            [0, 0, 0, w_3, w_4, w_5, w_6, 0],
            [0, 0, 0, 0, 0, 0, 0, w_7],
        ]
    )


def init_inputs(
    vicarious_learning,
    weekly_poll,
    pretreatment_survey,
    points,
    email_indicator,
    out_of_office,
    energy_saturation_measure,
    predicted_energy,
):
    return


def fetch_inputs():
    return


def update():
    return


def main():
    curr_date = dt.datetime.today()

    init()


if __name__ == "__main__":
    main()
