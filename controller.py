import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

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

class PG_Controller(BaseController):
    def __init__(self):
        self.gamma = .95
        self.policy_net = SimpleNet()
        self.action_t_holder = 

    def policy_forward_pass(self, state_t):
        """Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)
        """
        self.policy_net()

    def train(reward_t, state_t, action_t):
        normal_dist = MultivariateNormal(self.action_t, torch.eye(12))
        log_probs = normal_dist.log_prob(action_t)
        grad_J = log_probs*reward


    def define_log_proc(self):





class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 12)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
