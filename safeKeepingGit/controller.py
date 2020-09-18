import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.autograd import Variable


class BaseController():
    """
    Controller parent class -- defines how to interact with controllers
    """


    def __init__(self):
        pass

    def get_points(self, price_signal, **kwargs):
        """
        Inputs:
            price_signal -

        returns
            points - array of point values
            times - times for which the point values apply
        """
        return price_signal

    def update(self, reward):
        """
        Inputs:
            Some sort of reward that will updates the controller's policy.
        """
        pass


class BinController(BaseController):
    """

    """

    def __init__(self, action_dimension=3, scaling_factor=1, vertical_shift=10):
        self.ac_dim = action_dimension
        self.shift = vertical_shift
        self.scale = scaling_factor

    def _process_price_signal(self, price_signal):
        max_price = np.max(price_signal)
        min_price = np.min(price_signal)
        bins = np.linspace(min_price, max_price, self.ac_dim)
        binned_state = np.digitize(price_signal, bins) - np.ones(price_signal)
        return binned_state

    def get_points(price_signal):
        post_process = self._process_price_signal(price_signal)
        return self.policy(post_process)

    def policy(state):
        return self.scale * state + vertical_shift

    def update(reward):
        pass

class PGController(BaseController):
    def __init__(self, optimizer=None, policy=None, optimizer_params=None):
        if policy != None:
            self.policy_net = policy.float()
        else:
            self.policy_net = SimpleNet().float()

        if optimizer != None:
            self.optimizer = optimizer(self.policy_net.parameters(), optimizer_params)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-2)

        self.action_t_holder = None
        self.ac_dim = self.policy_net.ac_dim
        self.st_dim = self.policy_net.st_dim

    def policy_forward_pass(self, state_t):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)
        """
        return self.policy_net(state_t)

    def update(self, reward_t, state_t, action_t):
        normal_dist = MultivariateNormal(self.action_t_holder, torch.eye(self.ac_dim))
        log_probs = normal_dist.log_prob(action_t)

        loss = -log_probs * reward_t
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_points(self, price_signal, *other_state_vars):
        state = self.construct_state(price_signal)
        mean_action = self.policy_net(state.float())
        self.action_t_holder = mean_action
        # Continuous
        dist = MultivariateNormal(mean_action, torch.eye(self.ac_dim))
        output = dist.sample()
        return self.post_process_output(output)

    def get_points_without_sampling(self, price_signal, *other_state_vars):
        state = self.construct_state(price_signal)
        mean_action = self.policy_net(state.float())
        self.action_t_holder = mean_action

        dist = MultivariateNormal(mean_action, torch.eye(self.ac_dim)*1e-6)
        output = np.maximum(0,dist.sample())
        return self.post_process_output(output)

    def construct_state(self, price_signal):
        return Variable(torch.from_numpy(price_signal))

    def post_process_output(self, output):
        # returns action given NN output
        # currently naive
        return output

class SimpleNet(nn.Module):
    def __init__(self, ac_dim=24, st_dim=24):
        super(SimpleNet, self).__init__()
        self.ac_dim = ac_dim
        self.st_dim = st_dim
        self.fc1 = nn.Linear(self.st_dim, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, self.ac_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x






